"""Compose input-kernel coverage with the previously validated mHC worker."""
import json
import os
from input_warmup import run_coverage, topk_geometries, verify_source
from dsv4_mhc_worker import Worker as MHCWorker


class Worker(MHCWorker):
    def compile_or_warm_up_model(self):
        verify_source()
        runner = self.model_runner
        if (self.model_config.max_model_len not in (16384, 32768)
                or self.parallel_config.pipeline_parallel_size != 1):
            raise RuntimeError('Input warmup scope: 16K/32K max context, PP1')
        topk = topk_geometries(runner, self.model_config.max_model_len)
        draft = set()
        spec = runner.speculator
        if self.vllm_config.speculative_config is not None:
            if type(spec).__name__ != 'DSparkSpeculator' or not spec.sample_from_anchor:
                raise RuntimeError('Only pinned DSpark anchor sampling is covered')
            tables = spec.block_tables
            if (tables.cp_size, tables.cp_rank, tables.cp_interleave) != (1, 0, 1):
                raise RuntimeError('Context parallelism is outside tested input coverage')
            for gid in spec.draft_kv_cache_group_ids:
                draft.add((spec.max_num_reqs, spec.max_num_tokens, spec.max_model_len,
                           spec.num_query_per_req, spec.num_speculative_steps,
                           spec.parallel_drafting_token_id,
                           tables.input_block_tables[gid].stride(0), tables.kernel_block_sizes[gid]))
        record = dict(phase='STARTED', rank=self.rank,
                      dp_rank=self.parallel_config.data_parallel_index, pid=os.getpid())
        print('DSV4_INPUT_KERNEL_WARMUP ' + json.dumps(record), flush=True)
        record.update(run_coverage(sorted(topk), sorted(draft), self.device))
        record['phase'] = 'COMPLETE'
        print('DSV4_INPUT_KERNEL_WARMUP ' + json.dumps(record), flush=True)
        # Original mHC, graph capture, and RNG reset remain in the base chain.
        return super().compile_or_warm_up_model()
