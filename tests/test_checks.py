from __future__ import annotations

import json
import tempfile
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from serving_bench.checks.logs import compilation_events, kernel_checks
from serving_bench.clients.vllm_bench import normalize
from serving_bench.common import BenchError, option_names
from serving_bench.config import resolve
from serving_bench.executors import docker
from serving_bench.locks import gpu_locks

ROOT = Path(__file__).resolve().parents[1]


class CheckTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.case = resolve(ROOT / "configs/campaigns/46-glm52-vllm-smoke.yaml")["cases"][0]

    def test_kernel_gate_and_warning_visibility(self):
        case = self.case
        text = "GlmMoeDsaForCausalLM FLASHINFER_CUTLASS FLASHINFER_MLA_SPARSE_SM120\nWARNING FP8 scale missing"
        result = kernel_checks(case, text)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(len(result["warnings"]), 1)
        self.assertEqual(kernel_checks(case, "")['status'], "FAIL")
        case["recipe"]["checks"]["forbidden"] = ["FP8 scale missing"]
        self.assertEqual(kernel_checks(case, text)["status"], "FAIL")
        case["recipe"]["checks"]["required"] = []
        case["recipe"]["checks"]["forbidden"] = []
        self.assertEqual(kernel_checks(case, "")['status'], "UNVERIFIED")

    def test_sglang_autotune_detected(self):
        case = resolve(ROOT / "configs/campaigns/48-dsv4-sglang-smoke.yaml")["cases"][0]
        self.assertTrue(compilation_events(case, "[AutoTuner]: Tuning sparse_mla_sm120_decode_dsv4"))
        self.assertTrue(compilation_events(case, "Running FlashInfer autotune with cache"))
        self.assertFalse(compilation_events(case, "Server is ready. FlashInfer autotune completed."))

    def test_result_validation_and_missing_metrics(self):
        path = self.root / "raw.json"
        value = {"completed": 2, "output_throughput": 10.0, "mean_ttft_ms": 20, "mean_tpot_ms": 5}
        path.write_text(json.dumps(value))
        result = normalize(path, 2)
        self.assertIsNone(result["metrics"]["p95_ttft_ms"])
        self.assertTrue(result["failure_count_inferred"])
        for changes in ({"completed": 1}, {"failed": 1}, {"mean_ttft_ms": float("nan")}, {"output_throughput": -1}, {"completed": True}):
            path.write_text(json.dumps({**value, **changes}))
            with self.subTest(changes=changes), self.assertRaises(BenchError):
                normalize(path, 2)

    def test_missing_shard_and_architecture_rejected(self):
        self.case["model_path"] = str(self.root)
        (self.root / "config.json").write_text(json.dumps({"architectures": [self.case["model"]["architecture"]]}))
        (self.root / "tokenizer_config.json").write_text("{}")
        (self.root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"weight": "missing.safetensors"}}))
        with self.assertRaisesRegex(BenchError, "Missing model file"):
            docker.model_info(self.case)
        (self.root / "missing.safetensors").write_bytes(b"test")
        self.assertEqual(len(docker.model_info(self.case)["shard_sizes"]), 1)
        (self.root / "config.json").write_text('{"architectures": ["wrong"]}')
        with self.assertRaisesRegex(BenchError, "architecture mismatch"):
            docker.model_info(self.case)

    def test_ignored_ignore_eos_fails_length_check(self):
        path = self.root / "short.json"
        value = {"completed": 2, "output_throughput": 10, "mean_ttft_ms": 20, "mean_tpot_ms": 5, "total_output_tokens": 30}
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(BenchError, "Fixed-length output mismatch"):
            normalize(path, 2, self.case["workloads"][0])

    def test_gpu_validation(self):
        rows = [{"index": str(i), "uuid": f"GPU-{i}", "name": "NVIDIA RTX 6000D", "memory_mib": "85651", "driver": "580", "compute_capability": "12.0"} for i in range(8)]
        self.assertEqual(len(docker.validate_gpus(self.case["target"], rows)), 8)
        rows[-1]["compute_capability"] = "9.0"
        with self.assertRaisesRegex(BenchError, "compute capability"):
            docker.validate_gpus(self.case["target"], rows)

    def test_owned_cleanup_only(self):
        info = {"Config": {"Labels": {docker.OWNER_LABEL: "someone-else"}}}
        with patch.object(docker, "container_info", return_value=info), patch.object(docker, "capture") as capture:
            with self.assertRaisesRegex(BenchError, "not owned"):
                docker.remove_owned("name", "my-run")
            capture.assert_not_called()

    def test_auto_removed_containers_accept_docker_error_casing(self):
        for text in ("Error: No such container: test", "error: no such object: test"):
            response = subprocess.CompletedProcess([], 1, "", text)
            with self.subTest(text=text), patch.object(docker, "capture", return_value=response):
                self.assertIsNone(docker.container_info("test"))
        response = subprocess.CompletedProcess([], 1, "", "permission denied")
        with patch.object(docker, "capture", return_value=response):
            with self.assertRaisesRegex(BenchError, "Cannot inspect"):
                docker.container_info("test")

    def test_help_timeout_still_removes_owned_probe_container(self):
        with patch.object(docker, "capture", side_effect=BenchError("timeout")), patch.object(docker, "remove_owned") as remove:
            with self.assertRaisesRegex(BenchError, "timeout"):
                docker.inspect_cli("image", "vllm", ["serve"], {})
            remove.assert_called_once()
            self.assertTrue(remove.call_args.args[0].startswith("sb-help-"))

    def test_server_help_can_expose_one_gpu_without_loading_a_model(self):
        response = subprocess.CompletedProcess([], 0, "help", "")
        with patch.object(docker, "capture", return_value=response) as capture, patch.object(docker, "remove_owned"):
            docker.inspect_cli("image", "vllm", ["serve"], {}, "--help=all", gpus=[3])
            argv = capture.call_args.args[0]
            self.assertEqual(argv[argv.index("--gpus") + 1], '"device=3"')
            self.assertEqual(argv[-1], "--help=all")
            self.assertNotIn("/model", argv)

    def test_busy_selected_gpu_rejected(self):
        response = subprocess.CompletedProcess([], 0, "GPU-0, 123, other-server\n", "")
        with patch.object(docker, "capture", return_value=response):
            with self.assertRaisesRegex(BenchError, "already have compute processes"):
                docker.check_idle([{"uuid": "GPU-0"}])
            docker.check_idle([{"uuid": "GPU-1"}])

    def test_cache_isolation_across_image_and_recipe(self):
        first = docker.cache_directory(self.case, "sha256:first")
        self.assertNotEqual(first, docker.cache_directory(self.case, "sha256:second"))
        self.case["recipe"]["options"]["max-num-seqs"] = 10
        self.assertNotEqual(first, docker.cache_directory(self.case, "sha256:first"))

    def test_gpu_locks_overlap_and_release(self):
        lock_dir = self.root / "locks"
        with gpu_locks([0, 1], lock_dir):
            with self.assertRaisesRegex(BenchError, "locked"):
                with gpu_locks([1, 2], lock_dir):
                    pass
            with gpu_locks([2], lock_dir):
                pass
        with gpu_locks([0, 1], lock_dir):
            pass

    def test_help_parsing_handles_explicit_negative_switches(self):
        self.assertEqual(option_names("--enable-prefix-caching, --no-enable-prefix-caching --tp-size N"),
                         {"enable-prefix-caching", "no-enable-prefix-caching", "tp-size"})


if __name__ == "__main__":
    unittest.main()
