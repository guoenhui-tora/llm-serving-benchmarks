import ast
from functools import cache
from pathlib import Path
from types import SimpleNamespace
import unittest
from warmup_plan import coverage, split_for, draft_query_bound, validate_topology

class CoverageTests(unittest.TestCase):
    def test_matches_pinned_split_dispatch_for_every_token_count(self):
        source = Path(__file__).parents[1] / 'image-source/model_executor/kernels/mhc/tilelang_kernels.py'
        if not source.exists():
            source = Path('/usr/local/lib/python3.12/dist-packages/vllm/model_executor/kernels/mhc/tilelang_kernels.py')
        tree = ast.parse(source.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'compute_num_split')
        for sms in (80, 156, 188):
            env = {'cache': cache, 'torch': SimpleNamespace(cuda=SimpleNamespace(get_device_properties=lambda _: SimpleNamespace(multi_processor_count=sms))),
                   'cdiv': lambda x,y: (x+y-1)//y}
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), env)
            sizes, groups = coverage(16384,4096,4,sms,(1,2,4,8,12,16,24,32))
            for hidden in (4096, 16384):
                actual = {env['compute_num_split'](64,hidden,(n+63)//64) for n in range(1,16385)}
                selected = {env['compute_num_split'](64,hidden,(n+63)//64) for n in sizes}
                self.assertEqual(actual, selected)
                for n in range(1,16385):
                    self.assertEqual(split_for(n,hidden,sms),env['compute_num_split'](64,hidden,(n+63)//64))
            self.assertIn(1,sizes); self.assertIn(8,sizes); self.assertIn(16,sizes); self.assertIn(17,sizes)
            self.assertIn(16384,sizes)

    def test_sequence_parallel_matches_pinned_topology(self):
        for values in ((2,2,False,1,False),(2,2,True,1,True),(4,1,False,1,False)):
            validate_topology(*values)
        for values in ((2,2,False,1,True),(2,2,True,1,False),(2,2,False,2,False),(8,1,False,1,False)):
            with self.assertRaises(RuntimeError): validate_topology(*values)

    def test_draft_capacity_and_graph_bound(self):
        self.assertEqual(draft_query_bound(96, 5, [480, 576]), 576)
        self.assertEqual(draft_query_bound(48, 5, [192]), 288)
        self.assertEqual(draft_query_bound(32, 5, [288]), 288)
        for seqs, captures in [(97, []), (96, [577])]:
            with self.assertRaises(RuntimeError):
                draft_query_bound(seqs, 5, captures)

    def test_rejects_out_of_scope_model_and_budget(self):
        for params in [(16385,4096,4,156),(16384,16384,4,156),(16384,4096,8,156),(0,4096,4,156)]:
            with self.assertRaises(ValueError):coverage(*params)

if __name__=='__main__': unittest.main()
