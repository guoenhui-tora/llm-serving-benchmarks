import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys
import json

spec=importlib.util.spec_from_file_location('paired_run',Path(__file__).with_name('paired_run.py'))
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)

class PairedTests(unittest.TestCase):
    def item(self):
        return {'case':{'id':'mock','workloads':[{'measurement':{'max_warmup_rounds':5,'min_warmup_rounds':2,'warmup_requests':64}}]}}
    def check_rounds(self,rates,events,expected=None):
        seq=[({'metrics':{'output_throughput':x}},['compile'] if e else []) for x,e in zip(rates,events)]
        with tempfile.TemporaryDirectory() as tmp,patch.object(r,'phase',side_effect=seq) as phase:
            if expected is None:
                with self.assertRaises(r.BenchError): r.warmup(self.item(),Path(tmp),1,lambda _:None)
            else:
                self.assertEqual(len(r.warmup(self.item(),Path(tmp),1,lambda _:None)),expected)
            self.assertLessEqual(phase.call_count,5)
    def test_compilation_resets_quiet_count(self): self.check_rounds([100,100,100,100],[False,True,False,False],4)
    def test_quiet_but_unstable_is_rejected(self): self.check_rounds([100,130,100,130,100],[False]*5)
    def test_stable_after_cold_round(self): self.check_rounds([50,100,102],[True,False,False],3)
    def test_binding_does_not_alter_engine_command(self):
        argv=['docker','run','--entrypoint','python3','image','-c','hello']
        out=r.bound(argv,'sglang',True)
        self.assertEqual(out[2:6],['--cpuset-cpus','46-47,110-111','--cpuset-mems','2'])
        self.assertEqual(out[6:],argv[2:])
    def test_real_compile_events_and_fallback(self):
        c=r.resolve(Path(__file__).parents[1]/'configs/campaigns/45-dsv4-vllm-tp4-aligned-c32.yaml')['cases'][0]
        self.assertTrue(r.logs.compilation_events(c,'TileLang begins to compile kernel foo'))
        self.assertTrue(r.logs.compilation_events(c,'JIT compilation during inference'))
        self.assertFalse(r.logs.compilation_events(c,'No tuned config covers this shape; consider tuning'))

if __name__=='__main__': unittest.main()
