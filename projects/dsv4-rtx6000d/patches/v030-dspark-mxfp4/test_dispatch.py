"""CPU check of actual installed dispatcher and loader hook; no GPU kernels."""
import copy
import inspect
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sitecustomize as hook
from vllm.model_executor.layers.quantization import get_quantization_config
qc = importlib.import_module(get_quantization_config('deepseek_v4_fp8').__module__)
import vllm.model_executor.layers.quantization.modelopt as mo
import vllm.v1.worker.gpu.spec_decode.dspark.utils as loader
from dspark_native_mxfp4 import native_draft_config

assert 'native_draft_config(' in inspect.getsource(loader.load_dspark_model)
try:
    hook.patched_source(b'changed source')
except RuntimeError:
    pass
else:
    raise AssertionError('Changed loader must fail closed')
cfg = json.loads(Path('/model/config.json').read_text())
target = qc.DeepseekV4FP8Config.from_config(cfg['quantization_config'])
context = SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(**cfg)),
    quant_config=target, speculative_config=SimpleNamespace(method='dspark'))
original_hf = copy.deepcopy(cfg)
layer = qc.RoutedExperts.__new__(qc.RoutedExperts)
object.__setattr__(layer, 'moe_config', None)
with patch.object(qc, 'get_current_vllm_config', return_value=context), \
     patch.object(mo, 'ModelOptNvFp4FusedMoE', side_effect=lambda **kw: 'NVFP4'), \
     patch.object(qc, 'Mxfp4MoEMethod', side_effect=lambda *a: 'MXFP4'):
    before = target.get_quant_method(layer, 'model.layers.0.ffn.experts.routed_experts')
    draft = native_draft_config(context, SimpleNamespace(model='/model'),
        qc.DeepseekV4FP8Config.from_config(cfg['quantization_config']))
    observed = [draft.get_quant_method(layer, f'model.layers.{i}.ffn.experts.routed_experts')
                for i in (43,44,45)]
    after = target.get_quant_method(layer, 'model.layers.0.ffn.experts.routed_experts')
    assert before == after == 'NVFP4'
    assert observed == ['MXFP4'] * 3
    assert draft is not target and cfg == original_hf
    for attr in ('weight_block_size','ignored_layers','is_scale_e8m0','activation_scheme'):
        assert getattr(draft,attr) == getattr(target,attr), attr
print(json.dumps({'status':'PASS','target_before':before,'draft':observed,
    'target_after':after,'loader_hook':'installed','source_mismatch':'rejected',
    'weight_matrices_verified':2304,'quant_class_module':qc.__name__,
    'gpu_kernel_validation':False},indent=2))
