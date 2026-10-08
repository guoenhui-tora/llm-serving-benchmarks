"""CPU regression using the pinned image's actual propose/DPMetadata.make code.

Stubs represent tensors and collective results; no GPU/distributed execution.
"""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS

import vllm
from dspark_dp_profile import ProfileLoader, patch_source

ROOT = Path(__file__).resolve().parent
manifest = json.loads((ROOT / 'manifest.json').read_text())['dp_profile_backport']
vroot = Path(vllm.__file__).parent
path = vroot / 'v1/worker/gpu/spec_decode/dflash/speculator.py'
original = path.read_bytes()
patched = patch_source(original, manifest['original_sha256'])
assert hashlib.sha256(patched).hexdigest() == manifest['patched_sha256']
try:
    patch_source(original + b'\n', manifest['original_sha256'])
except RuntimeError:
    pass
else:
    raise AssertionError('Changed image source was accepted')

# Prove the only AST difference is the profiling call's DP argument.
before, after = ast.parse(original), ast.parse(patched)
changes = 0
for node in ast.walk(before):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == '_generate_draft':
        for kw in node.keywords:
            if kw.arg == 'num_tokens_across_dp' and isinstance(kw.value, ast.IfExp):
                kw.value = ast.Constant(None)
                changes += 1
assert changes == 1
assert ast.dump(before) == ast.dump(after), 'Changes leaked outside profiling argument'

class Tensor:
    def __init__(self, value=1):
        self.value = value
    def __getitem__(self, _):
        return self
    def copy_(self, _):
        return self
    def max(self):
        return self
    def item(self):
        return self.value


def function_from_source(source, class_name, name, namespace):
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    fn = copy.deepcopy(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name))
    fn.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), fn], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, '<actual-vllm-method>', 'exec'), namespace)
    return namespace[name]

make = function_from_source((vroot / 'forward_context.py').read_text(), 'DPMetadata', 'make', {'DPMetadata': lambda counts: counts})
old_propose = function_from_source(original, 'DFlashSpeculator', 'propose', {'CUDAGraphMode': NS(NONE=0)})
new_propose = function_from_source(patched, 'DFlashSpeculator', 'propose', {'CUDAGraphMode': NS(NONE=0)})


def run(propose, requests, k, rank, counts):
    observed = []
    def generate(nreqs, tokens, **kwargs):
        supplied = kwargs['num_tokens_across_dp']
        observed.append(supplied)
        # Simulate the existing fallback collective with each rank's own query
        # count; the original assertion is executed, not replaced by a test check.
        metadata = counts if supplied is None else supplied
        make(NS(data_parallel_size=2, data_parallel_rank=rank, is_moe_model=True), tokens, metadata)
        assert nreqs == requests and tokens == requests * k
    spec = NS(num_query_per_req=k, max_model_len=16384, hidden_states=Tensor(),
              context_positions=Tensor(), draft_tokens=Tensor(),
              model=NS(precompute_and_store_context_kv=lambda *args: None),
              _prepare_eplb_forward=lambda n: None, _generate_draft=generate)
    batch = NS(num_reqs=requests, num_tokens=8192, seq_lens_cpu_upper_bound=Tensor(8192))
    propose(spec, batch, None, None, Tensor(), None, None, None, None, None,
            None, None, dp_sync=NS(num_tokens_across_dp=[8192,8192]),
            dummy_run=True, skip_attn_for_dummy_run=True, is_profile=True)
    assert observed == [None]

try:
    run(old_propose, 16, 5, 0, [80,80])
except AssertionError as error:
    assert str(error) == '8192 80', str(error)
else:
    raise AssertionError('Original regression was not reproduced')

cases = 0
for k in range(1,8):
    for requests_by_rank in ((16,16),(16,3)):
        counts = [b*k for b in requests_by_rank]
        for rank,b in enumerate(requests_by_rank):
            run(new_propose,b,k,rank,counts)
            cases += 1

# Exercise the actual opt-in finder and loader without importing GPU modules.
import sitecustomize
finder = next(f for f in __import__('sys').meta_path if isinstance(f, sitecustomize.DraftOverlay))
flag = 'SERVING_BENCH_DSPARK_DP_PROFILE_FIX'
previous = os.environ.get(flag)
try:
    os.environ.pop(flag, None)
    assert finder.find_spec(manifest['module'], [str(path.parent)]) is None
    os.environ[flag] = '1'
    spec = finder.find_spec(manifest['module'], [str(path.parent)])
    assert isinstance(spec.loader, ProfileLoader)
    spec.loader.get_code(manifest['module'])
    # JIT source inspection must see the patched line mapping, not the image's
    # old file. Execute only the actual kernel definition with decorators removed.
    import inspect, linecache, sys, types
    kernel = copy.deepcopy(next(n for n in ast.parse(patched).body
                                if isinstance(n, ast.FunctionDef)
                                and n.name == '_prepare_dflash_inputs_kernel'))
    kernel.decorator_list = []
    module_name = '_dp_profile_source_regression'
    module = types.ModuleType(module_name)
    module.__file__ = str(path)
    module.__loader__ = spec.loader
    sys.modules[module_name] = module
    tree = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), kernel], type_ignores=[])
    ast.fix_missing_locations(tree)
    exec(compile(tree, spec.loader.source_filename, 'exec'), module.__dict__)
    fn = module.__dict__[kernel.name]
    for clear in (False, True):
        if clear:
            linecache.clearcache()
        found = ast.parse(inspect.getsource(fn))
        assert found.body[0].name == kernel.name
        assert ast.dump(ast.Module(body=found.body[0].body, type_ignores=[])) == ast.dump(ast.Module(body=kernel.body, type_ignores=[]))
    sys.modules.pop(module_name)
finally:
    if previous is None:
        os.environ.pop(flag, None)
    else:
        os.environ[flag] = previous
print(json.dumps({'status':'PASS','original_assertion':'8192 80',
                  'patched_profiling_cases':cases,'scope':'CPU methods + simulated DP metadata; no GPU/collective validation'}))
