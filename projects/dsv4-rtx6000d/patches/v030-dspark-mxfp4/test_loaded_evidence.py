"""Verify runtime observer accepts three MXFP4 modules and rejects wrong/missing ones."""
from types import SimpleNamespace
from unittest.mock import patch
import json
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4MoEMethod
from dspark_native_mxfp4 import verify_loaded_draft

items = []
for i in range(3):
    module = RoutedExperts.__new__(RoutedExperts)
    method = Mxfp4MoEMethod.__new__(Mxfp4MoEMethod)
    method.moe_kernel = object()
    method.mxfp4_backend = SimpleNamespace(value='TEST_BACKEND')
    object.__setattr__(module,'quant_method',method)
    items.append((f'layers.{i}.ffn.experts',module))
model = SimpleNamespace(named_modules=lambda:items)
with patch('vllm.logger.init_logger'):
    assert len(verify_loaded_draft(model)) == 3
    good = items[0][1].quant_method
    object.__setattr__(items[0][1],'quant_method',object())
    try: verify_loaded_draft(model)
    except RuntimeError: pass
    else: raise AssertionError('Wrong quantization method accepted')
    object.__setattr__(items[0][1],'quant_method',good)
    good.moe_kernel = None
    try: verify_loaded_draft(model)
    except RuntimeError: pass
    else: raise AssertionError('Missing kernel accepted')
    good.moe_kernel = object()
    items.pop()
    try: verify_loaded_draft(model)
    except RuntimeError: pass
    else: raise AssertionError('Missing draft layer accepted')
print(json.dumps({'status':'PASS','positive':1,'negative':3,'scope':'runtime observer logic'}))
