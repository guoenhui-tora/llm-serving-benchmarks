import ast
from functools import cache
from pathlib import Path
from types import SimpleNamespace
import unittest
from warmup_plan import coverage, split_for

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
            sizes, groups = coverage(8192,4096,4,sms,(1,2,4,8,12,16,24,32))
            for hidden in (4096, 16384):
                actual = {env['compute_num_split'](64,hidden,(n+63)//64) for n in range(1,8193)}
                selected = {env['compute_num_split'](64,hidden,(n+63)//64) for n in sizes}
                self.assertEqual(actual, selected)
                for n in range(1,8193):
                    self.assertEqual(split_for(n,hidden,sms),env['compute_num_split'](64,hidden,(n+63)//64))
            self.assertIn(1,sizes); self.assertIn(8,sizes); self.assertIn(16,sizes); self.assertIn(17,sizes)
            self.assertIn(8192,sizes)

    def test_rejects_out_of_scope_model_and_budget(self):
        for params in [(8193,4096,4,156),(8192,8192,4,156),(8192,4096,8,156),(0,4096,4,156)]:
            with self.assertRaises(ValueError):coverage(*params)

if __name__=='__main__': unittest.main()
