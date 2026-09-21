"""GPU regression: real pinned kernels, scratch-only correctness and cache reuse."""
import json
import unittest


class InputWarmupTest(unittest.TestCase):
    def test_builder_selection_excludes_indexer(self):
        from types import SimpleNamespace as NS
        from input_warmup import topk_geometries
        from vllm.models.deepseek_v4.sparse_mla import DeepseekV4SparseMLAMetadataBuilder as Builder
        c4, c128 = Builder.__new__(Builder), Builder.__new__(Builder)
        c4.compress_ratio, c4.topk_tokens = 4, 512
        c128.compress_ratio = 128
        for b in (c4,c128):b.kv_cache_spec=NS(block_size=256)
        c128.c128a_prefill_buffer=NS(stride=lambda dim:256)
        indexer=NS(compress_ratio=4)  # Real failure: no topk_tokens on indexer.
        def group(b):return NS(get_metadata_builder=lambda index:b)
        runner=NS(attn_groups=[[group(indexer),group(c4)],[group(c128)]],
                  block_tables=NS(input_block_tables=[NS(stride=lambda dim:128)]*2))
        self.assertEqual(topk_geometries(runner,32768),
                         [(128,256,128,2),(256,256,128,2),(512,512,128,64)])
        runner.attn_groups=[[group(indexer)]]
        with self.assertRaisesRegex(RuntimeError,'Missing actual C4/C128'):
            topk_geometries(runner,32768)

    def test_runtime_shapes_and_values(self):
        import torch
        from input_warmup import run_coverage, draft_case, topk_case, cache_keys, verify_source
        from vllm.models.deepseek_v4.common.ops.cache_utils import _compute_global_topk_indices_and_lens_kernel as topk
        from vllm.v1.worker.gpu.spec_decode.dflash.speculator import _prepare_dflash_inputs_kernel as draft
        verify_source()
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)
        geometries = [(512,512,64,64),(128,128,64,2),
                      (512,512,128,64),(128,256,128,2),(256,256,128,2)]
        drafts = [(16,8192,16384,5,5,0,64,256), (32,16384,32768,3,3,0,128,256)]
        record = run_coverage(geometries,drafts,device,budget_s=450)
        before = (cache_keys(topk),cache_keys(draft))
        # Replay different interior spans, arbitrary sliced mixed-token offsets,
        # and near-context-limit/null-block cases not just warmup endpoints.
        for g in drafts:
            for span in [2,5,7,13,31,47,65,91,129,255,257,1023,4095]:
                draft_case(g,span,device,near_limit=True)
        for g in geometries:
            for offset in [3,7,8,11,15]:
                topk_case(g,offset,device)
        self.assertEqual(before,(cache_keys(topk),cache_keys(draft)))
        print('INPUT_KERNEL_GPU_TEST '+json.dumps(record),flush=True)


if __name__ == '__main__':
    unittest.main()
