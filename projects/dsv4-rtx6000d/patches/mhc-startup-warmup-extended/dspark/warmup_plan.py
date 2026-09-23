"""Coverage of the pinned NVIDIA DSV4 mHC dispatch, without importing CUDA."""
def split_for(tokens, hidden, sms):
    return max(1, min(sms // ((tokens + 63) // 64), (hidden + 63) // 64 // 4))


def signature(tokens, hidden, hc_mult, sms):
    broadcast = split_for(tokens, hidden, sms)
    if tokens <= 16:
        fused = ('small', 8 if tokens < 8 and hidden <= 4096 else 4,
                 2 if tokens < 8 else 3)
    else:
        fused = ('large', split_for(tokens, hidden * hc_mult, sms), 1)
    return (broadcast, *fused)


def coverage(max_tokens, hidden, hc_mult, sms, captures=()):
    if not (1 <= max_tokens <= 16384 and hidden == 4096 and hc_mult == 4 and sms > 0):
        raise ValueError('This warmup is scoped to DSV4 Flash, max_tokens <= 16384')
    groups = {}
    for n in range(1, max_tokens + 1):
        groups.setdefault(signature(n, hidden, hc_mult, sms), []).append(n)
    # Both boundaries exercise dynamic token sizes within each dispatch class.
    sizes = sorted({n for v in groups.values() for n in (v[0], v[-1])}
                   | {n for n in captures if 1 <= n <= max_tokens})
    return sizes, [{'signature': list(k), 'first': v[0], 'last': v[-1]}
                   for k, v in groups.items()]


def draft_query_bound(max_num_seqs, speculative_tokens, captures):
    """Bound startup scratch work to DP-engine S96 / K5 (576 queries)."""
    bound = max(max_num_seqs * (speculative_tokens + 1), max(captures, default=1))
    if not 1 <= bound <= 576:
        raise RuntimeError('Draft query coverage exceeds bounded scope (576)')
    return bound


def validate_topology(tp, dp, ep, pp, sequence_parallel):
    if pp != 1 or (tp, dp, ep) not in ((4, 1, False), (2, 2, True), (2, 2, False)):
        raise RuntimeError('Warmup scope requires TP4 DP1 EPoff or TP2 DP2 EPon/off, PP1')
    if sequence_parallel != ep:
        raise RuntimeError('Actual sequence parallel path differs from pinned topology')
