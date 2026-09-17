from __future__ import annotations

import copy
import json
import shutil
import socket
import subprocess
import tempfile
import threading
import unittest
from contextlib import ExitStack
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from serving_bench import runner
from serving_bench.common import BenchError, read_json, write_json
from serving_bench.config import resolve
from serving_bench.executors import docker
from serving_bench.locks import gpu_locks
from serving_bench.results.report import report

ROOT = Path(__file__).resolve().parents[1]


class FakeAPI(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        if self.path == "/health":
            self.wfile.write(b"OK")
        else:
            self.wfile.write(json.dumps({"data": [{"id": self.server.served_name}]}).encode())

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append(payload)
        self.send_response(200)
        self.end_headers()
        self.wfile.write(json.dumps({"choices": [{"message": {"content": self.server.reply}}]}).encode())


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.api = ThreadingHTTPServer(("127.0.0.1", 0), FakeAPI)
        self.api.served_name = "GLM-5.2-NVFP4"
        self.api.reply = "1加1等于2。"
        self.api.requests = []
        self.thread = threading.Thread(target=self.api.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.api.server_close)
        self.addCleanup(self.api.shutdown)
        self.plan = resolve(ROOT / "tests/fixtures/configs/campaigns/46-glm52-vllm-smoke.yaml")
        self.case = self.plan["cases"][0]
        self.case["target"]["port"] = self.api.server_port
        self.case["target"]["cache_root"] = str(self.root / "cache")
        self.containers = {}
        self.measurements = 0
        self.last_phase = ""
        self.fail_client = False
        self.interrupt_client = False
        self.compile_once = False
        self.break_kernel = False
        self.fail_preflight = False
        self.stop_before_start = False
        self.facts = {"hostname": socket.gethostname(), "gpus": [{"name": "fake GPU", "uuid": "test"}],
                      "model": {"identity": "test-model"}, "server_image": {"Id": "sha256:" + "a" * 64},
                      "client_image": {"Id": "sha256:" + "b" * 64}}

    def fake_capture(self, argv, **kwargs):
        if argv[:2] == ["docker", "run"]:
            name = argv[argv.index("--name") + 1]
            owner = argv[argv.index("--label") + 1].split("=", 1)[1]
            self.containers[name] = {"State": {"Running": True}, "Config": {"Labels": {docker.OWNER_LABEL: owner}}}
            if "--cpuset-cpus" in argv:
                self.containers[name]["HostConfig"] = {
                    "CpusetCpus": argv[argv.index("--cpuset-cpus") + 1],
                    "CpusetMems": argv[argv.index("--cpuset-mems") + 1]}
        elif argv[:2] == ["docker", "exec"]:
            binding = self.case["target"]["binding"]["server"]
            observed = {"threads": [{"tid": 1, "cpus": binding["cpus"], "mems": binding["mems"]}], "unreadable": []}
            return subprocess.CompletedProcess(argv, 0, json.dumps(observed), "")
        elif argv[:3] == ["docker", "rm", "-f"]:
            self.containers.pop(argv[-1], None)
        else:
            raise AssertionError(f"Unexpected command: {argv}")
        return subprocess.CompletedProcess(argv, 0, "ok", "")

    def fake_client(self, argv, log_path, timeout):
        log_path.write_text("Simulated load generator; this is not GPU performance data.\n")
        owner = argv[argv.index("--label") + 1].split("=", 1)[1]
        name = argv[argv.index("--name") + 1]
        self.containers[name] = {"Config": {"Labels": {docker.OWNER_LABEL: owner}}, "State": {"Running": True}}
        if "--cpuset-cpus" in argv:
            self.containers[name]["HostConfig"] = {
                "CpusetCpus": argv[argv.index("--cpuset-cpus") + 1],
                "CpusetMems": argv[argv.index("--cpuset-mems") + 1]}
        self.last_phase = log_path.parent.name
        if self.interrupt_client:
            raise KeyboardInterrupt
        if self.fail_client:
            raise BenchError("simulated client timeout")
        if self.last_phase.startswith("measurement"):
            self.measurements += 1
        count = int(argv[argv.index("--num-prompts") + 1])
        write_json(log_path.parent / "raw.json", {"completed": count, "output_throughput": 123,
                                                  "mean_ttft_ms": 20, "mean_tpot_ms": 3,
                                                  "p95_ttft_ms": 30, "total_input_tokens": count * 128, "total_output_tokens": count * 32})

    def fake_logs(self, name, since=None):
        if since:
            if self.compile_once and self.last_phase == "measurement-01" and self.measurements == 1:
                return "JIT compilation during inference"
            return "request completed"
        return "WARNING expected test warning" if self.break_kernel else "\n".join(self.case["recipe"]["checks"]["required"]) + "\nWARNING expected test warning"

    def fake_preflight(self, case):
        if self.fail_preflight:
            raise BenchError("simulated incompatible CLI")
        if self.stop_before_start:
            write_json(self.root / "run/stop-requested.json", {})
        return self.facts

    def execute(self, path=None):
        with ExitStack() as stack:
            stack.enter_context(patch("serving_bench.common.capture", side_effect=self.fake_capture))
            stack.enter_context(patch.object(docker, "capture", side_effect=self.fake_capture))
            stack.enter_context(patch.object(docker, "preflight", side_effect=self.fake_preflight))
            stack.enter_context(patch.object(docker, "environment_snapshot", return_value={"test": True}))
            stack.enter_context(patch.object(docker, "container_info", side_effect=lambda name: self.containers.get(name)))
            stack.enter_context(patch.object(docker, "run_client", side_effect=self.fake_client))
            stack.enter_context(patch.object(docker, "server_logs", side_effect=self.fake_logs))
            stack.enter_context(patch("serving_bench.runner.time.sleep"))
            stack.enter_context(patch("serving_bench.runner.gpu_locks", side_effect=lambda gpus: gpu_locks(gpus, self.root / "locks")))
            return runner.run(self.plan, path or self.root / "run")

    def test_full_run_real_http_portable_results_and_cleanup(self):
        state = self.execute()
        self.assertEqual(state["status"], "PASS")
        self.assertFalse(self.containers)
        self.assertEqual(self.api.requests[0]["chat_template_kwargs"], {"enable_thinking": False})
        case_dir = self.root / "run/cases/glm52-vllm"
        for name in ("case.json", "resolved.json", "server.log", "environment.json", "kernel-checks.json"):
            self.assertTrue((case_dir / name).is_file(), name)
        shutil.move(str(self.root / "run"), str(self.root / "relocated"))
        result = report([self.root / "relocated"])
        self.assertEqual(len(result["groups"]), 1)
        self.assertEqual(result["groups"][0]["n"], 1)
        self.assertIsNone(result["groups"][0]["statistics"]["output_throughput"]["sd"])

    def test_bound_run_records_actual_container_and_server_affinity(self):
        self.case["target"]["binding"] = {"server": {"cpus": "0-1", "mems": "0"},
                                          "client": {"cpus": "2-3", "mems": "1"}}
        self.assertEqual(self.execute()["status"], "PASS")
        self.assertFalse(self.containers)
        directory = self.root / "run/cases/glm52-vllm"
        self.assertTrue((directory / "server-affinity.json").is_file())
        trial = directory / "trials/smoke-128-32/c0001/r01/measurement-01"
        self.assertTrue((trial / "client-binding-inspect.json").is_file())
        self.assertTrue((trial / "server-affinity.json").is_file())

    def test_binding_mismatch_fails_case_and_still_cleans_containers(self):
        self.case["target"]["binding"] = {"server": {"cpus": "0-1", "mems": "0"},
                                          "client": {"cpus": "2-3", "mems": "1"}}
        real_client = self.fake_client
        def mismatch(argv, log_path, timeout):
            real_client(argv, log_path, timeout)
            name = argv[argv.index("--name") + 1]
            self.containers[name]["HostConfig"]["CpusetCpus"] = "2-4"
        self.fake_client = mismatch
        self.assertEqual(self.execute()["status"], "FAIL")
        self.assertFalse(self.containers)
        self.assertEqual(self.measurements, 0)
        self.assertFalse(report([self.root / "run"])["groups"])
        state = read_json(self.root / "run/cases/glm52-vllm/case.json")
        self.assertIn("CpusetCpus", state["error"])

    def test_bound_client_removed_during_stop_preserves_interrupted_state(self):
        self.case["target"]["binding"] = {"client": {"cpus": "2-3", "mems": "1"}}
        def stopped(argv, log_path, timeout):
            log_path.write_text("Client removed by stop")
            raise KeyboardInterrupt
        self.fake_client = stopped
        self.assertEqual(self.execute()["status"], "INTERRUPTED")
        self.assertFalse(self.containers)
        case_dir = self.root / "run/cases/glm52-vllm"
        self.assertEqual(read_json(case_dir / "case.json")["status"], "INTERRUPTED")
        self.assertTrue(list(case_dir.rglob("client-binding-error.json")))

    def test_bound_client_launch_failure_keeps_original_error(self):
        self.case["target"]["binding"] = {"client": {"cpus": "2-3", "mems": "1"}}
        def failed(argv, log_path, timeout):
            log_path.write_text("Client container could not start")
            raise BenchError("original client launch error")
        self.fake_client = failed
        self.assertEqual(self.execute()["status"], "FAIL")
        self.assertFalse(self.containers)
        case_dir = self.root / "run/cases/glm52-vllm"
        self.assertIn("original client launch error", read_json(case_dir / "case.json")["error"])
        self.assertTrue(list(case_dir.rglob("client-binding-error.json")))

    def test_compilation_discards_measurement_and_rewarms(self):
        self.compile_once = True
        self.assertEqual(self.execute()["status"], "PASS")
        self.assertEqual(self.measurements, 2)
        checks = read_json(self.root / "run/cases/glm52-vllm/trials/smoke-128-32/c0001/r01/measurement-checks.json")
        self.assertEqual([x["accepted"] for x in checks["attempts"]], [False, True])

    def test_client_timeout_preserves_logs_and_removes_containers(self):
        self.fail_client = True
        self.assertEqual(self.execute()["status"], "FAIL")
        self.assertFalse(self.containers)
        self.assertTrue((self.root / "run/cases/glm52-vllm/server.log").is_file())
        self.assertFalse(report([self.root / "run"])["groups"])

    def test_interrupt_preserves_terminal_state_and_cleanup(self):
        self.interrupt_client = True
        self.assertEqual(self.execute()["status"], "INTERRUPTED")
        self.assertFalse(self.containers)
        state = read_json(self.root / "run/cases/glm52-vllm/case.json")
        self.assertEqual(state["status"], "INTERRUPTED")

    def test_failed_probe_does_not_measure(self):
        self.api.reply = "wrong answer"
        self.assertEqual(self.execute()["status"], "FAIL")
        self.assertEqual(self.measurements, 0)
        self.assertFalse(self.containers)

    def test_kernel_failure_excludes_measurements(self):
        self.break_kernel = True
        self.assertEqual(self.execute()["status"], "FAIL")
        self.assertEqual(self.measurements, 1)
        self.assertFalse(report([self.root / "run"])["groups"])

    def test_preflight_failure_is_recorded_without_starting_service(self):
        self.fail_preflight = True
        self.assertEqual(self.execute()["status"], "FAIL")
        self.assertFalse(self.containers)
        self.assertEqual(self.measurements, 0)

    def test_stop_during_preflight_prevents_launch(self):
        self.stop_before_start = True
        self.assertEqual(self.execute()["status"], "INTERRUPTED")
        self.assertEqual(self.measurements, 0)
        self.assertFalse(self.containers)

    def test_mixed_concurrency_never_averaged_together(self):
        self.case["workloads"][0]["traffic"]["concurrency"] = [1, 2]
        self.assertEqual(self.execute()["status"], "PASS")
        result = report([self.root / "run"])
        self.assertEqual({x["concurrency"] for x in result["groups"]}, {1, 2})
        self.assertTrue(all(x["n"] == 1 for x in result["groups"]))

    def test_multi_case_sequential_and_recipe_groups_separate(self):
        second = copy.deepcopy(self.case)
        second["id"] = "second"
        second["recipe"]["id"] = "second-recipe"
        self.plan["cases"].append(second)
        self.assertEqual(self.execute()["status"], "PASS")
        self.assertFalse(self.containers)
        self.assertEqual(len(report([self.root / "run"])["groups"]), 2)

    def test_sglang_full_flow_uses_sglang_server_and_vllm_client(self):
        self.plan = resolve(ROOT / "tests/fixtures/configs/campaigns/48-dsv4-sglang-smoke.yaml")
        self.case = self.plan["cases"][0]
        self.case["target"]["port"] = self.api.server_port
        self.case["target"]["cache_root"] = str(self.root / "cache")
        self.api.served_name = "deepseek-v4-flash"
        self.assertEqual(self.execute()["status"], "PASS")
        self.assertEqual(self.api.requests[0]["chat_template_kwargs"], {"thinking": False})
        argv = read_json(self.root / "run/cases/dsv4-sglang/argv.json")
        self.assertEqual(argv[argv.index("--entrypoint") + 1], "sglang")

    def test_existing_run_is_not_overwritten(self):
        self.execute()
        with self.assertRaisesRegex(BenchError, "already exists"):
            self.execute()

    def test_repeated_or_copied_run_not_counted_twice(self):
        self.execute()
        shutil.copytree(self.root / "run", self.root / "copy")
        result = report([self.root / "run", self.root / "run", self.root / "copy"])
        self.assertEqual(result["groups"][0]["n"], 1)

    def test_later_case_runs_after_cleaned_failure(self):
        second = copy.deepcopy(self.case)
        second["id"] = "second"
        second["recipe"]["probe"]["expect_regex"] = "wrong"
        self.case["recipe"]["probe"]["expect_regex"] = "never-match"
        self.api.reply = "wrong answer"
        self.plan["cases"].append(second)
        state = self.execute()
        self.assertEqual(state["status"], "FAIL")
        self.assertEqual([c["status"] for c in state["cases"]], ["FAIL", "PASS"])
        self.assertFalse(self.containers)
        self.assertEqual(len(report([self.root / "run"])["groups"]), 1)


if __name__ == "__main__":
    unittest.main()
