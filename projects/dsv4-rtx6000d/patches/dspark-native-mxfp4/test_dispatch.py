import json
from types import SimpleNamespace
from unittest.mock import patch
import vllm.models.deepseek_v4.quant_config as qc
import vllm.model_executor.layers.quantization.modelopt as mo
from dspark_native_mxfp4 import native_draft_config
import vllm.v1.worker.gpu.spec_decode.dspark.utils as utils
assert '/audit/utils.py' == utils.__file__
cfg=json.load(open('/model/config.json'))
target=qc.DeepseekV4FP8Config.from_config(cfg['quantization_config'])
context=SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(**cfg)),quant_config=target,speculative_config=SimpleNamespace(method='dspark'))
layer=qc.RoutedExperts.__new__(qc.RoutedExperts)
object.__setattr__(layer,'moe_config',None)
with patch.object(qc,'get_current_vllm_config',return_value=context), patch.object(mo,'ModelOptNvFp4FusedMoE',side_effect=lambda **kw:'NVFP4'), patch.object(qc,'Mxfp4MoEMethod',side_effect=lambda *a:'MXFP4'):
    before=target.get_quant_method(layer,'model.layers.0.ffn.experts.routed_experts')
    draft=native_draft_config(context,SimpleNamespace(model='/model'),qc.DeepseekV4FP8Config.from_config(cfg['quantization_config']))
    observed=[draft.get_quant_method(layer,f'model.layers.{i}.ffn.experts.routed_experts') for i in (43,44,45)]
    after=target.get_quant_method(layer,'model.layers.0.ffn.experts.routed_experts')
    assert before==after=='NVFP4'
    assert observed==['MXFP4']*3
    assert draft.weight_block_size==target.weight_block_size
    assert draft.ignored_layers==target.ignored_layers
    assert draft.is_scale_e8m0==target.is_scale_e8m0
    print(json.dumps({'target_before':before,'draft':observed,'target_after':after,'weight_block_size':draft.weight_block_size,'status':'PASS'}))
