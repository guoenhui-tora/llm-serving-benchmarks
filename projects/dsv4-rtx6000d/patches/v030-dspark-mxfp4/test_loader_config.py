"""Exercise actual CUDA registry and EngineArgs/DSpark loader; no model weights loaded."""
import importlib
import json
from unittest.mock import patch
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.models.utils import get_draft_quant_config
import vllm.model_executor.model_loader as model_loader
import vllm.model_executor.layers.quantization.modelopt as mo
import vllm.v1.worker.gpu.spec_decode.dspark.utils as loader

v = EngineArgs(model='/model',tokenizer_mode='deepseek_v4',trust_remote_code=True,
    tensor_parallel_size=2,pipeline_parallel_size=2,distributed_executor_backend='mp',
    max_model_len=32768,max_num_seqs=32,max_num_batched_tokens=8192,
    enable_expert_parallel=True,
    speculative_config={'method':'dspark','num_speculative_tokens':5}).create_engine_config()
q = get_draft_quant_config(v)
qc = importlib.import_module(type(q).__module__)
layer = qc.RoutedExperts.__new__(qc.RoutedExperts)
object.__setattr__(layer,'moe_config',None)
observed = {}

class StopBeforeModel(Exception):
    pass

def check_model(*, vllm_config, model_config):
    draft = vllm_config.quant_config
    assert vllm_config.model_config is v.model_config
    assert model_config is v.speculative_config.draft_model_config
    with patch.object(qc,'get_current_vllm_config',return_value=vllm_config):
        observed['draft'] = [draft.get_quant_method(layer,f'model.layers.{i}.ffn.experts.routed_experts')
                             for i in (43,44,45)]
    assert observed['draft'] == ['MXFP4']*3
    assert draft is not v.quant_config
    assert draft.weight_block_size == v.quant_config.weight_block_size
    raise StopBeforeModel

with patch.object(mo,'ModelOptNvFp4FusedMoE',side_effect=lambda **kw:'NVFP4'), \
     patch.object(qc,'Mxfp4MoEMethod',side_effect=lambda *a:'MXFP4'), \
     patch.object(qc,'get_current_vllm_config',return_value=v):
    # Verify the CUDA registry path reproduces the original bug as well.
    assert q.get_quant_method(layer,'model.layers.43.ffn.experts.routed_experts') == 'NVFP4'
    observed['target_before'] = v.quant_config.get_quant_method(layer,'model.layers.0.ffn.experts.routed_experts')
    # Process groups are not created in this configuration-only check.
    with patch.object(loader,'get_pp_safe_draft_load_config',side_effect=lambda x:x), \
         patch.object(model_loader,'get_model',side_effect=check_model):
        try:
            loader.load_dspark_model(object(),v)
        except StopBeforeModel:
            pass
        else:
            raise AssertionError('Expected to stop before model allocation')
    observed['target_after'] = v.quant_config.get_quant_method(layer,'model.layers.0.ffn.experts.routed_experts')
    assert observed['target_before'] == observed['target_after'] == 'NVFP4'
print(json.dumps({'status':'PASS','quant_module':qc.__name__,
    'actual_engine_config':True,'actual_loader_hook':True,'model_loaded':False,
    **observed},indent=2))
