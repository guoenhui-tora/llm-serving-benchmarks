"""Startup-only mHC coverage for the pinned vLLM 0.29.0 NVIDIA DSV4 worker.

No forward method, kernel implementation, or numerical option is replaced.
"""
import hashlib
import json
import os
import time
from pathlib import Path

from warmup_plan import coverage


def verify_source():
    import vllm
    root = Path(vllm.__file__).parent
    expected = json.loads(Path(__file__).with_name('pinned-source.json').read_text())
    for name, digest in expected.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Pinned source mismatch: {name}')


verify_source()
from vllm.v1.worker.gpu_worker import Worker as BaseWorker


class Worker(BaseWorker):
    def compile_or_warm_up_model(self):
        # Use the same real worker and loaded model, before ordinary graph
        # capture/JIT monitor activation. Base warmup and RNG reset still run.
        self._dsv4_mhc_coverage_warmup()
        return super().compile_or_warm_up_model()

    def _dsv4_mhc_coverage_warmup(self):
        import torch
        from vllm.model_executor.kernels.mhc import tilelang as ops
        from vllm.model_executor.kernels.mhc import tilelang_kernels as kernels
        from vllm.utils.deep_gemm import is_deep_gemm_supported
        model = self.get_model()
        layers = [m for m in model.modules()
                  if type(m).__name__ == 'DeepseekV4DecoderLayer'
                  and type(m).__module__ == 'vllm.models.deepseek_v4.nvidia.model']
        if not layers or not is_deep_gemm_supported():
            raise RuntimeError('Expected pinned NVIDIA DSV4 with DeepGEMM dispatch')
        if (self.parallel_config.tensor_parallel_size != 4
                or self.parallel_config.pipeline_parallel_size != 1
                or self.parallel_config.data_parallel_size != 1
                or self.vllm_config.speculative_config is not None):
            raise RuntimeError('Warmup scope requires TP4/PP1/DP1, DSpark off')
        layer = layers[0]
        configs = {(m.hidden_size, m.hc_mult, m.rms_norm_eps, m.hc_eps,
                    m.hc_post_alpha, m.hc_sinkhorn_iters,
                    m.attn_norm.variance_epsilon, m.ffn_norm.variance_epsilon)
                   for m in layers}
        if len(configs) != 1 or layer.hc_attn_fn_broadcast is None:
            raise RuntimeError('Uncovered layer parameters or broadcast weights')
        device = layer.hc_attn_fn.device
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        sizes, groups = coverage(self.scheduler_config.max_num_batched_tokens,
                                 layer.hidden_size, layer.hc_mult, sms,
                                 self.vllm_config.compilation_config.cudagraph_capture_sizes or [])
        monitored = {n: getattr(kernels, n) for n in (
            'mhc_pre_big_fuse_broadcast_with_norm_tilelang',
            'mhc_pre_big_fuse_with_norm_tilelang')}
        def keys():
            return {n: set(f._kernel_cache) for n, f in monitored.items()}
        started = time.monotonic()
        budget = 480
        record = dict(rank=self.rank, pid=os.getpid(), sms=sms, token_sizes=sizes,
                      dispatch_groups=groups, source_verified=True, phase='STARTED')
        print('DSV4_TARGETED_WARMUP ' + json.dumps(record), flush=True)
        before = keys()
        memory_before = torch.cuda.memory_allocated(device)
        passes = []
        with torch.inference_mode():
            x = torch.zeros((max(sizes), layer.hidden_size), dtype=torch.bfloat16, device=device)
            for phase in range(2):
                phase_start = time.monotonic()
                for size in sizes:
                    if time.monotonic() - started > budget:
                        raise TimeoutError('Targeted mHC warmup exceeded 480 seconds')
                    residual, post, mix, hidden = ops.mhc_pre_broadcast_tilelang(
                        x[:size], layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base,
                        layer.rms_norm_eps, layer.hc_eps, layer.hc_eps,
                        layer.hc_post_alpha, layer.hc_sinkhorn_iters,
                        norm_weight=layer.attn_norm.weight.data,
                        norm_eps=layer.attn_norm.variance_epsilon,
                        fn_broadcast=layer.hc_attn_fn_broadcast)
                    outputs = ops.mhc_fused_post_pre_tilelang(
                        hidden, residual, post, mix, layer.hc_ffn_fn,
                        layer.hc_ffn_scale, layer.hc_ffn_base,
                        layer.rms_norm_eps, layer.hc_eps, layer.hc_eps,
                        layer.hc_post_alpha, layer.hc_sinkhorn_iters,
                        n_splits=1, tile_n=1, norm_weight=layer.ffn_norm.weight.data,
                        norm_eps=layer.ffn_norm.variance_epsilon)
                    if not all(bool(torch.isfinite(t).all()) for t in outputs):
                        raise RuntimeError(f'Nonfinite warmup output at {size} tokens')
                    torch.cuda.synchronize(device)
                    del outputs, residual, post, mix, hidden
                current = keys()
                passes.append(dict(seconds=time.monotonic()-phase_start,
                                   cache_entries={n:len(k) for n,k in current.items()}))
                if phase == 0:
                    loaded = current
                elif current != loaded:
                    raise RuntimeError('Second identical warmup pass added mHC cache keys')
            del x
        expected_broadcast = {g['signature'][0] for g in groups}
        expected_fused = {g['signature'][2] for g in groups}
        expected = [expected_broadcast, expected_fused]
        actual = {n: {k[0][7] for k in ks} for n,ks in loaded.items()}
        for (n, values), need in zip(actual.items(), expected):
            if not need <= values:
                raise RuntimeError(f'Incomplete split coverage for {n}: {need-values}')
        record.update(phase='COMPLETE', seconds=time.monotonic()-started, passes=passes,
                      cache_before={n:len(k) for n,k in before.items()},
                      covered_splits={n:sorted(v) for n,v in actual.items()},
                      memory_allocated_delta=torch.cuda.memory_allocated(device)-memory_before)
        print('DSV4_TARGETED_WARMUP ' + json.dumps(record), flush=True)
