"""Export exact official-client request observations after its timed window."""
import inspect
import json
import math
from pathlib import Path
import vllm.benchmarks.serve as serve

original = serve.calculate_metrics
signature = inspect.signature(original)

def capture(*args, **kwargs):
    result = original(*args, **kwargs)
    bound = signature.bind(*args, **kwargs).arguments
    outputs = bound['outputs']
    lengths = result[1]
    rows = []
    for i, (output, length) in enumerate(zip(outputs, lengths, strict=True)):
        tpot = (output.latency - output.ttft) / (length - 1) if length > 1 else 0.0
        rows.append(dict(index=i, success=output.success, input_tokens=output.prompt_len,
                         output_tokens=length, start_time=output.start_time,
                         ttft_s=output.ttft, tpot_s=tpot, e2el_s=output.latency,
                         slo_pass=bool(output.success and output.ttft <= 10 and tpot <= 0.05)))
    good = sum(row['slo_pass'] for row in rows)
    duration = bound['dur_s']
    assert math.isclose(good / duration, result[0].request_goodput, rel_tol=1e-10, abs_tol=1e-12)
    with Path('/results/requests.jsonl').open('x') as f:
        for row in rows:
            f.write(json.dumps(row) + '\n')
    Path('/results/slo.json').write_text(json.dumps(dict(
        ttft_limit_s=10, tpot_limit_s=.05, target_joint_pass_rate=.95,
        requests=len(rows), good_requests=good, joint_pass_rate=good/len(rows),
        request_goodput=good/duration, duration=duration), indent=2) + '\n')
    return result

serve.calculate_metrics = capture
