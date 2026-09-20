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
    if not (1 <= max_tokens <= 8192 and hidden == 4096 and hc_mult == 4 and sms > 0):
        raise ValueError('This warmup is scoped to DSV4 Flash, max_tokens <= 8192')
    groups = {}
    for n in range(1, max_tokens + 1):
        groups.setdefault(signature(n, hidden, hc_mult, sms), []).append(n)
    # Both boundaries exercise dynamic token sizes within each dispatch class.
    sizes = sorted({n for v in groups.values() for n in (v[0], v[-1])}
                   | {n for n in captures if 1 <= n <= max_tokens})
    return sizes, [{'signature': list(k), 'first': v[0], 'last': v[-1]}
                   for k, v in groups.items()]
