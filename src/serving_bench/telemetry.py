"""Read-only resource time series for whole-node experiments."""
from __future__ import annotations

import json
import subprocess
import threading
import time
from pathlib import Path


class Recorder:
    def __init__(self, directory: Path, log, interval=5):
        self.path = directory / "resource-telemetry.jsonl"
        self.log = log
        self.interval = interval
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.collect, daemon=True)

    def start(self):
        self.thread.start()

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=20)
        if self.thread.is_alive():
            raise RuntimeError("Resource recorder did not stop")

    def collect(self):
        previous = None
        index = 0
        with self.path.open("w", buffering=1) as stream:
            while not self.stop_event.is_set():
                record = {"unix_s": time.time(), "monotonic_s": time.monotonic()}
                try:
                    cpus = {line.split()[0]: list(map(int, line.split()[1:9]))
                            for line in Path('/proc/stat').read_text().splitlines() if line.startswith('cpu')}
                    record['cpu_ticks'] = cpus
                    if previous:
                        delta = [b-a for a,b in zip(previous['cpu'],cpus['cpu'])]
                        record['host_cpu_busy_fraction'] = 1-(delta[3]+delta[4])/sum(delta) if sum(delta)>0 else None
                        if record['host_cpu_busy_fraction'] is not None and record['host_cpu_busy_fraction'] > 0.45:
                            self.log('RESOURCE WARNING: host CPU busy >45%; inspect background/cgroup telemetry before accepting resource comparability')
                    previous = cpus
                    record['memory'] = {line.split(':')[0]: line.split(':')[1].strip()
                                        for line in Path('/proc/meminfo').read_text().splitlines()
                                        if line.split(':')[0] in {'MemAvailable','SwapFree','SwapTotal','AnonPages','Cached'}}
                    for name, command in [('gpu',['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used,power.draw,temperature.gpu,clocks.sm,clocks.mem','--format=csv,noheader,nounits'])]:
                        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
                        record[name] = {'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr}
                    if index % 6 == 0:
                        cgroups = {}
                        for group in Path('/sys/fs/cgroup/system.slice').glob('*.service'):
                            try:
                                cgroups[group.name] = (group/'cpu.stat').read_text()
                            except OSError:
                                pass
                        record['system_service_cpu_stat'] = cgroups
                except Exception as exc:
                    record['observation_error'] = str(exc)
                stream.write(json.dumps(record)+'\n')
                index += 1
                self.stop_event.wait(self.interval)
