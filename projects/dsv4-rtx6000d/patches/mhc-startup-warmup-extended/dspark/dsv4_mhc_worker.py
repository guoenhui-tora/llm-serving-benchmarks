"""Startup-only mHC coverage for the pinned vLLM 0.29.0 NVIDIA DSV4 worker.

No forward method, kernel implementation, or numerical option is replaced.
"""
import hashlib
import json
import os
import time
from pathlib import Path

from warmup_plan import coverage, draft_query_bound, validate_topology


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
        self._dsv4_draft_coverage_warmup()
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
        topology = (self.parallel_config.tensor_parallel_size,
                    self.parallel_config.data_parallel_size,
                    self.parallel_config.enable_expert_parallel)
        for actual_layer in layers:
            validate_topology(*topology, self.parallel_config.pipeline_parallel_size,
                              actual_layer.use_sequence_parallel)
        if (self.vllm_config.speculative_config is None
                or self.vllm_config.speculative_config.method != "dspark"
                or self.vllm_config.speculative_config.num_speculative_tokens not in (3, 5)):
            raise RuntimeError('Warmup scope requires TP4 DP1 EPoff or TP2 DP2 EPon/off, PP1, DSpark K3/K5')
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
        record = dict(rank=self.rank, dp_rank=self.parallel_config.data_parallel_index, topology=topology, pid=os.getpid(), sequence_parallel=layer.use_sequence_parallel, sms=sms, token_sizes=sizes,
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

    def _dsv4_draft_coverage_warmup(self):
        import torch
        from vllm.model_executor.kernels.mhc import tilelang as ops
        from vllm.model_executor.kernels.mhc import tilelang_kernels as kernels
        draft = self.get_draft_model()
        if type(draft).__name__ != 'DSparkDeepseekV4ForCausalLM':
            raise RuntimeError('Unexpected draft implementation')
        model = draft.model
        layers = list(model.layers)
        if len(layers) != 3 or list(model.target_layer_ids) != [40, 41, 42]:
            raise RuntimeError('Unexpected draft layers or auxiliary target layers')
        configs = {(m.hidden_size, m.hc_mult, m.rms_norm_eps, m.hc_eps,
                    m.hc_post_alpha, m.hc_sinkhorn_iters,
                    m.attn_norm.variance_epsilon, m.ffn_norm.variance_epsilon)
                   for m in layers}
        layer = layers[0]
        if len(configs) != 1 or (layer.hidden_size, layer.hc_mult) != (4096, 4):
            raise RuntimeError('Uncovered draft geometry')
        k = self.vllm_config.speculative_config.num_speculative_tokens
        captures = self.vllm_config.compilation_config.cudagraph_capture_sizes or []
        bound = draft_query_bound(self.scheduler_config.max_num_seqs, k, captures)
        device = layer.hc_attn_fn.device
        names = draft.get_draft_kv_cache_layer_names()
        group_map = self.model_runner.kv_cache_config.transfer_group_index_by_layer
        groups = {n: group_map.get(n) for n in names}
        registration = None
        if self.vllm_config.kv_transfer_config is not None:
            worker = self.model_runner.kv_connector.kv_connector.connector_worker
            if (any(g is None for g in groups.values())
                    or any(n not in worker._layer_specs for n in names)
                    or not worker.src_xfer_handles_by_block_size):
                raise RuntimeError('Draft SWA KV missing from active NIXL registration')
            registration = dict(engine_id=worker.engine_id, compat_hash=worker.compat_hash,
                                num_regions=worker.num_regions,
                                local_transfer_handles=len(worker.src_xfer_handles_by_block_size))
        record = dict(phase='STARTED', rank=self.rank,
                      dp_rank=self.parallel_config.data_parallel_index,
                      pid=os.getpid(), k=k, token_min=1, token_max=bound,
                      draft_kv_groups=groups, nixl_registration=registration,
                      cache_tensors=[dict(name=n, shape=list(m.attn.swa_cache_layer.kv_cache.shape),
                                          dtype=str(m.attn.swa_cache_layer.kv_cache.dtype))
                                     for n,m in zip(names,layers)])
        print('DSV4_DSPARK_WARMUP ' + json.dumps(record), flush=True)
        monitored = {n: getattr(kernels,n) for n in (
            'mhc_pre_big_fuse_with_norm_tilelang', 'mhc_fused_tilelang',
            'mhc_post_tilelang', 'hc_head_fuse_tilelang')}
        def keys():
            return {n:set(f._kernel_cache) for n,f in monitored.items()}
        started = time.monotonic()
        memory_before = torch.cuda.memory_allocated(device)
        passes = []
        with torch.inference_mode():
            x = torch.zeros((bound, layer.hc_mult, layer.hidden_size),
                            dtype=torch.bfloat16, device=device)
            for phase in range(2):
                phase_start = time.monotonic()
                for size in range(1, bound + 1):
                    if time.monotonic() - started > 480:
                        raise TimeoutError('Draft coverage exceeded 480 seconds')
                    post, mix, hidden = ops.mhc_pre_tilelang(
                        x[:size], layer.hc_attn_fn, layer.hc_attn_scale,
                        layer.hc_attn_base, layer.rms_norm_eps, layer.hc_eps,
                        layer.hc_eps, layer.hc_post_alpha, layer.hc_sinkhorn_iters,
                        norm_weight=layer.attn_norm.weight.data,
                        norm_eps=layer.attn_norm.variance_epsilon)
                    residual2, post2, mix2, hidden2 = ops.mhc_fused_post_pre_tilelang(
                        hidden, x[:size], post, mix, layer.hc_ffn_fn,
                        layer.hc_ffn_scale, layer.hc_ffn_base, layer.rms_norm_eps,
                        layer.hc_eps, layer.hc_eps, layer.hc_post_alpha,
                        layer.hc_sinkhorn_iters, n_splits=1, tile_n=1,
                        norm_weight=layer.ffn_norm.weight.data,
                        norm_eps=layer.ffn_norm.variance_epsilon)
                    residual3 = ops.mhc_post_tilelang(hidden2, residual2, post2, mix2)
                    head = ops.hc_head_fused_kernel_tilelang(
                        residual3, model.hc_head_fn, model.hc_head_scale,
                        model.hc_head_base, model.rms_norm_eps, model.hc_eps)
                    if not all(bool(torch.isfinite(t).all()) for t in (
                            post,mix,hidden,residual2,post2,mix2,hidden2,residual3,head)):
                        raise RuntimeError(f'Nonfinite draft warmup at {size}')
                    torch.cuda.synchronize(device)
                    del post,mix,hidden,residual2,post2,mix2,hidden2,residual3,head
                current = keys()
                passes.append(dict(seconds=time.monotonic()-phase_start,
                                   cache_entries={n:len(v) for n,v in current.items()}))
                if phase == 0:
                    loaded = current
                elif current != loaded:
                    raise RuntimeError('Repeated draft warmup added mHC cache keys')
            del x
        delta = torch.cuda.memory_allocated(device)-memory_before
        if delta != 0:
            raise RuntimeError(f'Draft warmup retains tensor memory: {delta}')
        record.update(phase='COMPLETE', seconds=time.monotonic()-started,
                      passes=passes, memory_allocated_delta=delta)
        print('DSV4_DSPARK_WARMUP ' + json.dumps(record), flush=True)
