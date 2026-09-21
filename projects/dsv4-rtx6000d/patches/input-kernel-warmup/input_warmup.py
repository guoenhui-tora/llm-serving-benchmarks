"""Startup coverage of pinned DSV4 input kernels; only private scratch tensors."""
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace


def verify_source():
    import vllm
    root = Path(vllm.__file__).parent
    expected = json.loads(Path(__file__).with_name('input-pinned-source.json').read_text())
    for path, digest in expected.items():
        if hashlib.sha256((root / path).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Input warmup source mismatch: {path}')


def topk_geometries(runner, max_model_len):
    from vllm.models.deepseek_v4.sparse_mla import DeepseekV4SparseMLAMetadataBuilder
    geometries, ratios = set(), set()
    for gid, groups in enumerate(runner.attn_groups):
        for group in groups:
            builder = group.get_metadata_builder(0)
            # Indexer metadata also has compress_ratio=4, but never calls this
            # attention slot-remapping kernel. Select the pinned builder type.
            if not isinstance(builder, DeepseekV4SparseMLAMetadataBuilder):
                continue
            ratio = builder.compress_ratio
            if ratio not in (4, 128):
                raise RuntimeError(f'Uncovered sparse attention compression: {ratio}')
            ratios.add(ratio)
            stride = runner.block_tables.input_block_tables[gid].stride(0)
            size = builder.kv_cache_spec.block_size // ratio
            if ratio == 4:
                width = builder.topk_tokens
                geometries.add((width, width, stride, size))
            else:
                capacity = builder.c128a_prefill_buffer.stride(0)
                for length in range(1, max_model_len+1):
                    width = min(max(1 << (max(length//128, 1)-1).bit_length(), 128), capacity)
                    geometries.add((width, capacity, stride, size))
    if ratios != {4, 128}:
        raise RuntimeError(f'Missing actual C4/C128 attention builders: {ratios}')
    return sorted(geometries)


def draft_spans(max_tokens, query_tokens):
    if not 1 <= max_tokens <= 16384 or query_tokens not in (3, 5):
        raise ValueError('Scope: DSV4 K3/K5, budget<=16384')
    groups = {}
    for n in range(1, max_tokens + 1):
        block = min(256, 1 << (n + query_tokens - 1).bit_length())
        groups.setdefault(block, []).append(n)
    return sorted({n for group in groups.values() for n in (group[0], group[-1])})


def cache_keys(kernel):
    import torch
    return set(kernel.device_caches[torch.cuda.current_device()][0])


def topk_case(geometry, offset, device):
    import torch
    from vllm.models.deepseek_v4.common.ops.cache_utils import compute_global_topk_indices_and_lens
    width, row_stride, block_stride, block_size = geometry
    backing = torch.full((4, row_stride), -1, dtype=torch.int32, device=device)
    local = backing[:, :width]
    local[:, :3] = torch.tensor([0, block_size - 1, block_size], device=device)
    req = torch.arange(offset + 4, dtype=torch.int32, device=device) % 2
    valid = torch.arange(offset + 4, device=device) % 2 == 0
    table = (torch.arange(2 * block_stride, device=device, dtype=torch.int32)
             .reshape(2, block_stride) + 7)
    mapped, lens = compute_global_topk_indices_and_lens(
        local, req[offset:], table, block_size, valid[offset:])
    # Independent integer block-table reference, including -1 and padding rows.
    cpu_local, cpu_req, cpu_table = local.cpu(), req[offset:].cpu(), table.cpu()
    expected = torch.full((4, width), -1, dtype=torch.int32)
    for i in range(4):
        for j in range(3):
            block, remainder = divmod(int(cpu_local[i, j]), block_size)
            expected[i, j] = int(cpu_table[int(cpu_req[i]), block]) * block_size + remainder
    assert torch.equal(mapped.cpu(), expected), ('topk slots', geometry, offset)
    assert torch.equal(lens.cpu(), valid[offset:].cpu().int() * 3), ('topk padding', geometry, offset)


def draft_case(geometry, span, device, near_limit=False):
    import numpy as np
    import torch
    from vllm.v1.worker.gpu.input_batch import InputBuffers
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import prepare_dflash_inputs
    nmax, tmax, model_len, q, steps, token_id, block_stride, block_size = geometry
    lengths = [span, 1] if span < tmax and nmax >= 2 else [span]
    n = len(lengths)
    starts = [0]
    positions = []
    for length in lengths:
        base = model_len - length - 1 if near_limit else 0
        positions += list(range(base, base + length))
        starts.append(starts[-1] + length)
    rejected = [int(span > 2)] + ([0] if n == 2 else [])
    sampled = [1] + ([0] if n == 2 else [])
    def tensor(values, dtype=torch.int32):
        return torch.tensor(values, dtype=dtype, device=device)
    def zeros(size, dtype=torch.int64):
        return torch.zeros(size, dtype=dtype, device=device)
    batch = SimpleNamespace(num_reqs=n, num_scheduled_tokens=np.array(lengths, dtype=np.int32),
                            positions=tensor(positions, torch.int64), query_start_loc=tensor(starts),
                            idx_mapping=tensor(list(reversed(range(n))), torch.int64))
    buffers = InputBuffers(nmax, tmax, device)
    query_slots, context_pos, context_slots = zeros(tmax), zeros(tmax), zeros(tmax)
    sample_indices, sample_pos = zeros(nmax * steps), zeros(nmax * steps)
    sample_map = zeros(nmax * steps, torch.int32)
    out_temp, out_seeds = zeros(nmax, torch.float32), zeros(nmax)
    last = tensor(list(range(100, 100 + nmax)), torch.int64)
    next_token = tensor(list(range(200, 200 + nmax)))
    temp = tensor([0.25] * nmax, torch.float32)
    seeds = tensor(list(range(nmax)), torch.int64)
    table = torch.arange(1, nmax * block_stride + 1, dtype=torch.int32, device=device).reshape(nmax, block_stride)
    table[:, 0] = 0  # Sliding-window/null blocks must never become writable slots.
    prepare_dflash_inputs(buffers, query_slots, context_pos, context_slots,
                         sample_indices, sample_pos, sample_map, out_temp, out_seeds,
                         batch, tensor(sampled), tensor(rejected), last, next_token,
                         temp, seeds, table, block_size, 0, 1, 1, token_id,
                         q, steps, nmax, tmax, model_len, sample_from_anchor=True)
    table_cpu = table.cpu()
    def slot(row, pos):
        b = int(table_cpu[row, min(pos // block_size, block_stride - 1)])
        return b * block_size + pos % block_size if b else -1
    expected_query, expected_qslots, expected_ids, expected_samples = [], [], [], []
    expected_context, expected_cslots = [], []
    for i, length in enumerate(lengths):
        state = n - 1 - i
        good = length - rejected[i]
        last_pos = positions[starts[i] + good - 1]
        for j in range(length):
            p = positions[starts[i] + j]
            expected_context.append(p if j < good else 0)
            expected_cslots.append(slot(i, p) if j < good else -1)
        for j in range(q):
            pos = last_pos + 1 + j
            expected_query.append(min(pos, model_len - 1))
            expected_qslots.append(slot(i, pos))
            expected_ids.append((100 if sampled[i] else 200) + state if j == 0 else token_id)
            expected_samples.append(pos + 1)
        assert int(buffers.seq_lens[i]) == min(last_pos + 1 + q, model_len)
    pairs = [(buffers.positions[:n*q], expected_query), (query_slots[:n*q], expected_qslots),
             (buffers.input_ids[:n*q], expected_ids), (sample_pos[:n*steps], expected_samples),
             (context_pos[:starts[-1]], expected_context), (context_slots[:starts[-1]], expected_cslots)]
    for actual, expected in pairs:
        assert actual.cpu().tolist() == expected, ('draft values', geometry, span, near_limit)
    assert sample_indices[:n*steps].cpu().tolist() == list(range(n*q))
    assert sample_map[:n*steps].cpu().tolist() == [state for state in reversed(range(n)) for _ in range(steps)]
    assert buffers.query_start_loc.cpu().tolist() == [min(i, n)*q for i in range(nmax+1)]
    assert bool((buffers.seq_lens[n:] == 0).all())
    assert bool((query_slots[n*q:] == -1).all())
    assert bool((sample_indices[n*steps:] == 0).all()) and bool((sample_map[n*steps:] == -1).all())
    assert out_temp[:n].cpu().tolist() == [0.25]*n and out_seeds[:n].cpu().tolist() == list(range(n))


def run_coverage(topk_geometries, draft_geometries, device, budget_s=300):
    import torch
    from vllm.models.deepseek_v4.common.ops.cache_utils import _compute_global_topk_indices_and_lens_kernel as topk
    kernels = {'topk': topk}
    if draft_geometries:
        from vllm.v1.worker.gpu.spec_decode.dflash.speculator import _prepare_dflash_inputs_kernel
        kernels['draft_inputs'] = _prepare_dflash_inputs_kernel
    started = time.monotonic()
    memory = torch.cuda.memory_allocated(device)
    passes = []
    with torch.inference_mode():
        for phase in range(2):
            phase_start = time.monotonic()
            for geometry in sorted(set(topk_geometries)):
                for offset in range(16):
                    if time.monotonic() - started > budget_s:
                        raise TimeoutError('Input kernel warmup budget exceeded')
                    topk_case(geometry, offset, device)
            for geometry in sorted(set(draft_geometries)):
                for span in draft_spans(geometry[1], geometry[3]):
                    for near_limit in (False, True):
                        if time.monotonic() - started > budget_s:
                            raise TimeoutError('Input kernel warmup budget exceeded')
                        draft_case(geometry, span, device, near_limit)
            torch.cuda.synchronize(device)
            current = {k:cache_keys(v) for k,v in kernels.items()}
            passes.append(dict(seconds=time.monotonic()-phase_start,
                               cache_entries={k:len(v) for k,v in current.items()}))
            if phase == 0:
                loaded = current
            elif current != loaded:
                raise RuntimeError('Second input warmup pass added specialization keys')
    delta = torch.cuda.memory_allocated(device)-memory
    if delta:
        raise RuntimeError(f'Input warmup retained scratch memory: {delta}')
    return dict(topk_geometries=topk_geometries, pointer_offsets=list(range(16)),
                draft_geometries=draft_geometries,
                draft_spans={str(g):draft_spans(g[1],g[3]) for g in draft_geometries},
                passes=passes, memory_allocated_delta=delta, seconds=time.monotonic()-started,
                specialization_keys={k:sorted(str(x) for x in v) for k,v in loaded.items()},
                specialization_key_sha256={k:sorted(hashlib.sha256(str(x).encode()).hexdigest() for x in v) for k,v in loaded.items()})
