import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from serving_bench.clients import decode_only
from serving_bench.common import BenchError


class DecodeOnlyTests(unittest.TestCase):
    def test_shared_prefix_dataset_and_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.jsonl"
            content = json.dumps({"id": "original", "prompt": "hello",
                                  "input_tokens": 16384, "output_tokens": 1024}) + "\n"
            source.write_text(content)
            digest = hashlib.sha256(content.encode()).hexdigest()
            rows, identity = decode_only.dataset_rows(source, digest, 4, 1024, 16384)
            self.assertEqual(identity, digest)
            self.assertEqual(len({row["prompt"] for row in rows}), 1)
            self.assertEqual(len({row["id"] for row in rows}), 4)
            with self.assertRaises(ValueError):
                decode_only.dataset_rows(source, "bad", 4, 1024, 16384)

    def test_metric_counters_reject_workload_or_series_mismatch(self):
        def metrics(hits, computed, success, preemptions):
            return decode_only.parse_metrics(
                f'vllm:prefix_cache_hits_total{{engine="0"}} {hits}\n'
                f'vllm:request_prefill_kv_computed_tokens_sum{{engine="0"}} {computed}\n'
                f'vllm:request_success_total{{engine="0"}} {success}\n'
                f'vllm:num_preemptions_total{{engine="0"}} {preemptions}\n',
                decode_only.COUNTERS,
            )

        counts = decode_only.deltas(metrics(10, 5, 2, 0),
                                    metrics(2 * 16128 + 10, 2 * 256 + 5, 4, 0))
        decode_only.check_round(counts, 2, 16384, 256)
        with self.assertRaisesRegex(BenchError, "cache hit deficit"):
            decode_only.check_round({**counts, "vllm:prefix_cache_hits_total": 1}, 2, 16384, 256)
        with self.assertRaisesRegex(BenchError, "Excess local prefill"):
            decode_only.check_round({**counts, "vllm:request_prefill_kv_computed_tokens_sum": 513},
                                    2, 16384, 256)
        with self.assertRaisesRegex(BenchError, "Missing or changed"):
            decode_only.deltas(metrics(0, 0, 0, 0), {})

    def test_dp_prime_must_hit_each_engine(self):
        def snapshot(hit0, hit1, success):
            lines = []
            for engine, hits in ((0, hit0), (1, hit1)):
                for name, value in (("prefix_cache_hits_total", hits),
                                    ("request_prefill_kv_computed_tokens_sum", 0),
                                    ("request_success_total", success // 2),
                                    ("num_preemptions_total", 0)):
                    lines.append(f'vllm:{name}{{engine="{engine}"}} {value}')
            return decode_only.parse_metrics("\n".join(lines), decode_only.COUNTERS)

        before = snapshot(0, 0, 0)
        decoded = decode_only.check_primes(before, snapshot(16128, 16128, 4), 2, 16384, 256)
        self.assertEqual(decoded["vllm:request_success_total"], 4)
        with self.assertRaisesRegex(BenchError, "all 2 engines"):
            decode_only.check_primes(before, snapshot(32256, 0, 4), 2, 16384, 256)
        with self.assertRaisesRegex(BenchError, "all 1 engines"):
            decode_only.check_primes(before, snapshot(16128, 16128, 4), 1, 16384, 256)

    def test_plan_is_offline_and_does_not_create_run_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.jsonl"
            content = json.dumps({"id": "source", "prompt": "test",
                                  "input_tokens": 16384, "output_tokens": 1024}) + "\n"
            source.write_text(content)
            result = root / "new-result"
            with patch.object(decode_only, "PREFIX_SHA256", hashlib.sha256(content.encode()).hexdigest()), \
                 patch.object(decode_only, "snapshot", side_effect=AssertionError("server accessed")), \
                 patch.object(decode_only, "image_info", side_effect=AssertionError("docker accessed")):
                self.assertEqual(0, decode_only.main([
                    "--source", str(source), "--model-dir", str(root), "--port", "31449",
                    "--run-root", str(result), "--plan",
                ]))
            self.assertFalse(result.exists())

    def test_client_command_uses_pinned_image_and_existing_jsonl_adapter(self):
        args = SimpleNamespace(client_cpus=None, client_mems=None, port=31449,
                               served_name="deepseek-v4-flash", model_dir=Path("/model-host"),
                               client_image="vllm/vllm-openai:v0.30.0",
                               client_image_id=decode_only.IMAGE_ID, concurrency=128)
        dataset = {"id": "shared", "dataset": {"name": "jsonl", "path": "/tmp/data.jsonl",
                                                "output_tokens": 1024, "sha256": "a", "max_input_tokens": 16384},
                   "traffic": {"request_rate": "inf"},
                   "sampling": {"seed": 0, "temperature": 0, "ignore_eos": True}}
        command = decode_only.client_command(args, dataset, 512, Path("/tmp/out"), "owner", "client")
        self.assertIn(decode_only.IMAGE_ID, command)
        self.assertIn("--no-oversample", command)
        self.assertIn("--disable-shuffle", command)
        self.assertEqual(command[command.index("--num-prompts") + 1], "512")
        self.assertEqual(command[command.index("--max-concurrency") + 1], "128")

    def test_profile_control_posts_and_records_completed_request(self):
        class Response:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

        with tempfile.TemporaryDirectory() as tmp, patch.object(decode_only.HTTP, "open", return_value=Response()) as open_:
            directory = Path(tmp)
            decode_only.profile_control(31332, "start", directory)
            request = open_.call_args.args[0]
            self.assertEqual(request.get_method(), "POST")
            self.assertEqual(request.full_url, "http://127.0.0.1:31332/start_profile")
            self.assertGreater(float((directory / "profile-start").read_text()), 0)

    def test_client_records_window_boundaries_even_if_docker_fails(self):
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(decode_only.subprocess, "run", side_effect=RuntimeError("client error")), \
             patch.object(decode_only, "remove_owned") as remove:
            directory = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, "client error"):
                decode_only.run_client(["docker", "run"], directory, "name", "owner", 10)
            self.assertLessEqual(int((directory / "client-start-ns").read_text()),
                                 int((directory / "client-finish-ns").read_text()))
            remove.assert_called_once_with("name", "owner")


if __name__ == "__main__":
    unittest.main()
