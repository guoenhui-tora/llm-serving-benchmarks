"""Stdlib JSONL loading and evidence, also mounted into the pinned client."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def read_records(path, output_tokens, max_input_tokens, expected_sha256=None):
    content = Path(path).read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("JSONL SHA256 mismatch; dataset changed")
    records, ids = [], set()
    for line_number, line in enumerate(content.decode("utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise ValueError(f"JSONL line {line_number}: invalid JSON") from exc
        if not isinstance(row, dict) or set(row) - {"id", "prompt", "input_tokens", "output_tokens"}:
            raise ValueError(f"JSONL line {line_number}: expected id/prompt and optional token counts")
        for key in ("id", "prompt"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"JSONL line {line_number}: {key} must be a nonempty string")
        if row["id"] in ids:
            raise ValueError(f"JSONL duplicate id: {row['id']}")
        ids.add(row["id"])
        for key in ("input_tokens", "output_tokens"):
            if key in row and (type(row[key]) is not int or row[key] <= 0):
                raise ValueError(f"JSONL {row['id']}: {key} must be a positive integer")
        if row.get("input_tokens", 1) > max_input_tokens:
            raise ValueError(f"JSONL {row['id']}: declared input exceeds max_input_tokens")
        if row.get("output_tokens", output_tokens) != output_tokens:
            raise ValueError(f"JSONL {row['id']}: output_tokens differs from workload")
        records.append(row)
    if not records:
        raise ValueError("JSONL dataset is empty")
    return records, digest


def install(serve, dataset, output: Path, index=0, replicas=1):
    """Keep official CustomDataset.sample, transport and clocks unchanged."""
    from vllm.benchmarks.datasets import CustomDataset

    records, digest = read_records("/dataset.jsonl", dataset["output_tokens"],
                                   dataset["max_input_tokens"], dataset["sha256"])

    def load_data(self):
        self.data = records

    CustomDataset.load_data = load_data
    original = serve.get_samples

    def get_samples(args, tokenizer):
        if (args.dataset_name != "custom" or args.dataset_path != "/dataset.jsonl"
                or not args.skip_chat_template or not args.disable_shuffle or not args.no_oversample
                or args.custom_output_len != dataset["output_tokens"]):
            raise ValueError("JSONL requires fixed order, raw prompts and fixed output budget")
        count = args.num_prompts
        if count <= 0 or count > len(records) or count % replicas or not 0 <= index < replicas:
            raise ValueError("JSONL insufficient rows or invalid replica partition")
        requests = original(args, tokenizer)
        if len(requests) != count:
            raise ValueError("JSONL official sampler returned the wrong request count")
        manifest = []
        for position, (row, request) in enumerate(zip(records, requests)):
            if request.prompt != row["prompt"] or request.expected_output_len != dataset["output_tokens"]:
                raise ValueError("JSONL official sampler changed prompt/order/output budget")
            # Count independently of optional metadata using the actual client tokenizer.
            length = len(tokenizer(row["prompt"]).input_ids)
            if request.prompt_len != length or not 0 < length <= dataset["max_input_tokens"]:
                raise ValueError(f"JSONL {row['id']}: actual input exceeds bound or sampler count differs")
            if "input_tokens" in row and row["input_tokens"] != length:
                raise ValueError(f"JSONL {row['id']}: input_tokens metadata differs from actual tokenizer ({length})")
            manifest.append({"index": position, "id": row["id"], "input_tokens": length,
                             "output_tokens": request.expected_output_len})
        selected = manifest[index::replicas]
        evidence = {"sha256": digest, "selection": "first_n_interleaved", "global_requests": count,
                    "replica": index, "replicas": replicas, "requests": selected,
                    "total_input_tokens": sum(r["input_tokens"] for r in selected),
                    "total_output_tokens": sum(r["output_tokens"] for r in selected),
                    "tokenizer": str(getattr(tokenizer, "name_or_path", "unknown")),
                    "tokenization": "tokenizer(prompt) defaults; no chat template"}
        (output / "dataset-manifest.json").write_text(json.dumps(evidence, indent=2))
        return requests  # The existing synchronized hook performs the actual split.

    serve.get_samples = get_samples


def validated_manifest(path, dataset, expected, partition=None):
    """Verify recorded workload identity/counts before accepting client metrics."""
    value = json.loads(Path(path).read_text())
    if value.get("sha256") != dataset["sha256"] or value.get("selection") != "first_n_interleaved":
        raise ValueError("JSONL manifest identity mismatch")
    index, replicas, count = (value.get(k) for k in ("replica", "replicas", "global_requests"))
    if (type(index) is not int or type(replicas) is not int or type(count) is not int
            or replicas not in (1, 2) or not 0 <= index < replicas or count != expected * replicas):
        raise ValueError("JSONL manifest partition mismatch")
    if partition is not None and (index, replicas, count) != partition:
        raise ValueError("JSONL manifest differs from scheduled partition")
    rows = value.get("requests")
    if not isinstance(rows, list) or len(rows) != expected:
        raise ValueError("JSONL manifest request count mismatch")
    ids = set()
    for position, row in zip(range(index, count, replicas), rows):
        if not isinstance(row, dict) or row.get("index") != position:
            raise ValueError("JSONL manifest order mismatch")
        name, length = row.get("id"), row.get("input_tokens")
        if not isinstance(name, str) or not name or name in ids:
            raise ValueError("JSONL manifest duplicate or missing id")
        ids.add(name)
        if type(length) is not int or not 0 < length <= dataset["max_input_tokens"]:
            raise ValueError("JSONL manifest invalid input length")
        if row.get("output_tokens") != dataset["output_tokens"]:
            raise ValueError("JSONL manifest output budget mismatch")
    if (value.get("total_input_tokens") != sum(r["input_tokens"] for r in rows)
            or value.get("total_output_tokens") != expected * dataset["output_tokens"]):
        raise ValueError("JSONL manifest token totals mismatch")
    return value
