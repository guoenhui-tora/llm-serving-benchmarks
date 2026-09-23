"""Version-pinned diagnostic for the locally audited NVIDIA 0731 checkpoint.

Keep FP8 linear handling and target layer metadata; correct only draft MoE
format selection. This is not a general upstream patch or a precision conversion.
"""
import copy
import hashlib
import json
from pathlib import Path
import struct

CONFIG_SHA = 'bb0d2286d6761439e41d3cef31d16489411b816ed8688922f59730bbd5567cdb'
INDEX_SHA = '5d2ad3076e04081d6c0728cb4b004dc832850ec5ae732f3adb06cf87c4b437a5'


def native_draft_config(vllm_config, draft_config, quant_config):
    from vllm.model_executor.layers.quantization import get_quantization_config
    from vllm.logger import init_logger

    assert vllm_config.speculative_config.method == 'dspark'
    # CUDA v0.30 registers the V4.1-compatible class for both V4 and V4.1.
    # Use the actual registry class; the CPU fallback class is a different type.
    assert isinstance(quant_config, get_quantization_config('deepseek_v4_fp8'))
    model = Path(draft_config.model)
    assert hashlib.sha256((model / 'config.json').read_bytes()).hexdigest() == CONFIG_SHA
    index_bytes = (model / 'model.safetensors.index.json').read_bytes()
    assert hashlib.sha256(index_bytes).hexdigest() == INDEX_SHA
    index = json.loads(index_bytes)['weight_map']
    headers = {}
    for stage in range(3):
        for expert in range(256):
            for matrix in ('w1', 'w2', 'w3'):
                base = f'mtp.{stage}.ffn.experts.{expert}.{matrix}'
                assert base + '.weight_scale_2' not in index
                assert base + '.input_scale' not in index
                pair = []
                for suffix in ('.weight', '.scale'):
                    name = base + suffix
                    shard = index[name]
                    if shard not in headers:
                        with (model / shard).open('rb') as stream:
                            size = struct.unpack('<Q', stream.read(8))[0]
                            headers[shard] = json.loads(stream.read(size))
                    pair.append(headers[shard][name])
                weight, scale = pair
                assert weight['dtype'] == 'I8' and scale['dtype'] == 'F8_E8M0'
                assert scale['shape'] == [weight['shape'][0], weight['shape'][1] // 16]
    # Never mutate the target's config. Pin lazy MoE properties on a separate
    # draft instance, since current model_config intentionally remains target.
    result = copy.copy(quant_config)
    result._resolved_expert_dtype = 'fp4'
    result._resolved_moe_quant_algo = ''
    result._nvfp4_config = None
    assert result is not vllm_config.quant_config
    init_logger(__name__).warning(
        'LOCAL_DSPARK_FORMAT_FIX: validated 768 native MXFP4 draft experts; '
        'draft=Mxfp4MoEMethod group32 E8M0; target NVFP4 unchanged; '
        'FP8 linear configuration retained.'
    )
    return result


def verify_loaded_draft(draft_model):
    """Observe real loaded modules on each draft rank, including nonzero PP ranks."""
    from vllm.model_executor.layers.fused_moe import RoutedExperts
    from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4MoEMethod
    from vllm.logger import init_logger

    rows = []
    for name, module in draft_model.named_modules():
        if not isinstance(module, RoutedExperts):
            continue
        method = module.quant_method
        if not isinstance(method, Mxfp4MoEMethod):
            raise RuntimeError(f'Draft {name} uses {type(method).__name__}, expected MXFP4')
        if method.moe_kernel is None:
            raise RuntimeError(f'Draft {name} has no loaded MXFP4 kernel')
        rows.append({'layer':name,'method':type(method).__name__,
                     'backend':method.mxfp4_backend.value})
    if len(rows) != 3:
        raise RuntimeError(f'Expected 3 loaded draft MoE modules, got {len(rows)}')
    # Default oracle info_once(scope="local") is silent on the last PP stage.
    # Explicitly log the loaded objects on every draft worker, without changing
    # backend selection, tensors, forward execution, or any warning filter.
    # This external module is outside vllm's INFO logger namespace. Emit the
    # audit record directly so root WARNING level cannot silently hide it.
    print('LOCAL_DSPARK_LOADED: verified 3 Mxfp4 MoE backend modules: '
          + json.dumps(rows, sort_keys=True), flush=True)
    return rows
