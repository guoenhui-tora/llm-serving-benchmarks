from __future__ import annotations

import copy
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from serving_bench.common import BenchError
from serving_bench.config import load_document, resolve
from serving_bench.engines import get_engine
from serving_bench.executors.docker import server_command
from serving_bench.clients.vllm_bench import command

ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "configs"
        shutil.copytree(ROOT / "tests/fixtures/configs", self.root)
        self.campaign = self.root / "campaigns/46-glm52-vllm-smoke.yaml"

    def edit(self, relative, update):
        path = self.root / relative
        value = yaml.safe_load(path.read_text())
        update(value)
        path.write_text(yaml.safe_dump(value))

    def test_fixture_campaigns_resolve_without_docker(self):
        with patch("subprocess.run", side_effect=AssertionError("Offline resolution executed a command")):
            for path in self.root.joinpath("campaigns").glob("*.yaml"):
                with self.subTest(path=path.name):
                    self.assertTrue(resolve(path)["cases"])

    def test_unknown_fields_fail(self):
        self.edit("runtimes/vllm-0.29.0.yaml", lambda v: v.update(engnie="vllm"))
        with self.assertRaisesRegex(BenchError, "unknown"):
            resolve(self.campaign)

    def test_duplicate_yaml_key_fails(self):
        path = self.root / "runtimes/vllm-0.29.0.yaml"
        path.write_text(path.read_text() + "engine: sglang\n")
        with self.assertRaisesRegex(BenchError, "Duplicate YAML"):
            resolve(self.campaign)

    def test_reference_escape_fails(self):
        self.edit("campaigns/46-glm52-vllm-smoke.yaml", lambda v: v.update(target="../outside.yaml"))
        with self.assertRaisesRegex(BenchError, "relative path"):
            resolve(self.campaign)

    def test_symlink_reference_escape_fails(self):
        path = self.root / "targets/rtx6000d-46.yaml"
        outside = Path(self.temp.name) / "outside.yaml"
        shutil.copy(path, outside)
        path.unlink()
        path.symlink_to(outside)
        with self.assertRaisesRegex(BenchError, "escapes root"):
            resolve(self.campaign)

    def test_cross_engine_recipe_rejected(self):
        self.edit("runtimes/vllm-0.29.0.yaml", lambda v: v.update(engine="sglang"))
        with self.assertRaisesRegex(BenchError, "engine mismatch"):
            resolve(self.campaign)

    def test_cross_hardware_recipe_rejected(self):
        self.edit("targets/rtx6000d-46.yaml", lambda v: v["gpu"].update(compute_capability="9.0"))
        with self.assertRaisesRegex(BenchError, "hardware mismatch"):
            resolve(self.campaign)

    def test_unknown_runtime_version_rejected(self):
        self.edit("runtimes/vllm-0.29.0.yaml", lambda v: v.update(version="0.30.0"))
        with self.assertRaisesRegex(BenchError, "version outside"):
            resolve(self.campaign)

    def test_missing_cache_disable_rejected(self):
        self.edit("recipes/glm52/vllm-sm120-smoke.yaml", lambda v: v["flags"].remove("no-enable-prefix-caching"))
        with self.assertRaisesRegex(BenchError, "cache=disabled"):
            resolve(self.campaign)

    def test_native_false_not_silently_inverted(self):
        self.edit("recipes/glm52/vllm-sm120-smoke.yaml", lambda v: v["options"].update({"enforce-eager": False}))
        with self.assertRaises(BenchError):
            resolve(self.campaign)

    def test_managed_flags_rejected(self):
        self.edit("recipes/glm52/vllm-sm120-smoke.yaml", lambda v: v["options"].update(port=1234))
        with self.assertRaisesRegex(BenchError, "managed"):
            resolve(self.campaign)

    def test_capacity_and_context_constraints(self):
        for key, value, pattern in [("tensor-parallel-size", 4, "GPU count"), ("max-model-len", 100, "context length")]:
            with self.subTest(key=key):
                case = resolve(self.campaign)["cases"][0]
                case["recipe"]["options"][key] = value
                if key == "tensor-parallel-size":
                    with self.assertRaisesRegex(BenchError, pattern):
                        get_engine("vllm").validate(case)
                else:
                    self.edit("recipes/glm52/vllm-sm120-smoke.yaml", lambda v: v["options"].update({key: value}))
                    with self.assertRaisesRegex(BenchError, pattern):
                        resolve(self.campaign)

    def test_smoke_cannot_masquerade_as_performance(self):
        self.edit("workloads/smoke-128-32.yaml", lambda v: v.update(purpose="performance"))
        with self.assertRaisesRegex(BenchError, "smoke recipe"):
            resolve(self.campaign)

    def test_case_selection_and_duplicate_ids(self):
        self.assertEqual(len(resolve(self.campaign, case_ids=["glm52-vllm"])["cases"]), 1)
        with self.assertRaisesRegex(BenchError, "Unknown case"):
            resolve(self.campaign, case_ids=["missing"])
        self.edit("campaigns/46-glm52-vllm-smoke.yaml", lambda v: v["cases"].append(copy.deepcopy(v["cases"][0])))
        with self.assertRaisesRegex(BenchError, "Duplicate case"):
            resolve(self.campaign)

    def test_native_commands_and_fixed_client_image(self):
        glm = resolve(self.campaign)["cases"][0]
        ds = resolve(self.root / "campaigns/48-dsv4-sglang-smoke.yaml")["cases"][0]
        for case in (glm, ds):
            argv = server_command(case, "test", "owner")
            self.assertIn("--entrypoint", argv)
            index = argv.index("--entrypoint")
            self.assertEqual(argv[index + 1], case["runtime"]["engine"])
            self.assertEqual(argv[index + 3], "serve")
            self.assertIn("--pull", argv)
            self.assertIn("127.0.0.1", argv)
            client = command(case, case["workloads"][0], 1, 2, Path("/results path"), "client", "owner")
            self.assertIn("vllm/vllm-openai:v0.29.0", client)
            self.assertNotIn("--gpus", client)
            self.assertEqual(client[client.index("--entrypoint") + 1], "python3")
            self.assertIn("vllm.benchmarks.serve", client[client.index("-c") + 1])
        self.assertNotIn("--model-path", server_command(glm, "test", "owner"))
        self.assertIn("--model-path", server_command(ds, "test", "owner"))

    def test_sglang_dp_attention_is_not_replica_multiplier(self):
        ds = resolve(self.root / "campaigns/48-dsv4-sglang-smoke.yaml")["cases"][0]
        ds["recipe"]["options"]["dp-size"] = 2
        with self.assertRaisesRegex(BenchError, "DP"):
            get_engine("sglang").validate(ds)
        ds["recipe"]["flags"].append("enable-dp-attention")
        self.assertEqual(get_engine("sglang").validate(ds), 4096)


if __name__ == "__main__":
    unittest.main()
