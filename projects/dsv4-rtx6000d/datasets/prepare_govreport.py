#!/usr/bin/env python3
"""Offline, input-only GovReport datasets for the pinned DSV4 tokenizer."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import tempfile

REPO = 'ccdv/govreport-summarization'
REVISION = '4e21184e01ae8017e2c036e180fe5e541fef60a0'
SOURCE_HASHES = {
    'train-00000-of-00002.parquet': '8b02abb5691b3fa1ba95f76331cadc8bb3e80195c626b4656b480d36e0c92242',
    'train-00001-of-00002.parquet': 'ce7786d8626c94c58163b195b6d4c6ddaa01ff0a9c3574ac21399ae56c3dd3a3',
}
MODEL_HASHES = {
    'tokenizer.json': '8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf',
    'tokenizer_config.json': '6ac8c8dc065ed118161d02dd532749ae3f52c243deac27872134fae2f50d8547',
    'encoding/encoding_dsv4.py': 'abc0d26120250dda0ae077dc64aa28836026e61e970854aaeb792445e6a0dde6',
}
INSTRUCTION = (
    'Read the government report below and write a detailed summary in English. '
    'Cover its background, central questions, major findings, supporting evidence, '
    'and conclusions or recommendations where present. Preserve important facts '
    'and figures, organize the summary into coherent paragraphs, and do not add '
    'information that is absent from the report.'
)
MANIFEST_NAME = 'govreport-manifest.json'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def text_sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def emit(**values):
    print(json.dumps(values, ensure_ascii=False), flush=True)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, obj):
    with Path(path).open('x') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write('\n')


def filename(isl, count):
    return f'govreport-isl{isl}-exact-n{count}.jsonl'


class Context:
    def __init__(self, args):
        import pyarrow.parquet as pq
        from transformers import AutoTokenizer

        self.model = args.model_dir.resolve()
        for name, expected in MODEL_HASHES.items():
            require(sha(self.model / name) == expected, f'Model file hash mismatch: {name}')
        spec = importlib.util.spec_from_file_location('govreport_dsv4_encoding', self.model / 'encoding/encoding_dsv4.py')
        self.encoder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.encoder)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model, trust_remote_code=True, local_files_only=True)
        require(self.tokenizer.is_fast, 'A fast tokenizer with character offsets is required')
        self.docs = {}
        for name, expected in SOURCE_HASHES.items():
            path = args.source_dir / name
            require(sha(path) == expected, f'Source file hash mismatch: {name}')
            self.docs[name] = pq.read_table(path, columns=['report'], use_threads=False)['report'].to_pylist()
        self.overhead = self.length(self.prompt(''))
        emit(phase='loaded', source_rows=sum(map(len, self.docs.values())), prompt_overhead=self.overhead)

    def prompt(self, report):
        return self.encoder.encode_messages([{'role': 'user', 'content': INSTRUCTION + '\n\nReport:\n' + report + '\n\nDetailed summary:\n'}], thinking_mode='chat')

    def length(self, prompt):
        return len(self.tokenizer(prompt).input_ids)

    def document(self, record):
        name, row = record['source_file'], record['source_row']
        require(name in self.docs, f'Unsupported source file: {name}')
        require(type(row) is int and 0 <= row < len(self.docs[name]), 'Invalid source row')
        doc = self.docs[name][row]
        require(isinstance(doc, str) and text_sha(doc) == record['report_sha256'], 'Source report hash mismatch')
        return doc

    def exact_prefix(self, doc, isl):
        offsets = self.tokenizer(doc, add_special_tokens=False, return_offsets_mapping=True)['offset_mapping']
        if not offsets:
            return None
        index, seen, cut = min(isl - self.overhead, len(offsets) - 1), set(), 0
        for _ in range(8):
            if index in seen or not 0 <= index < len(offsets):
                break
            seen.add(index)
            cut = offsets[index][0]
            p = self.prompt(doc[:cut])
            n = self.length(p)
            if cut > 0 and n == isl:
                return cut, p
            index += isl - n
        for distance in range(1, 17):
            for candidate_cut in (cut - distance, cut + distance):
                if 0 < candidate_cut <= len(doc):
                    p = self.prompt(doc[:candidate_cut])
                    if self.length(p) == isl:
                        return candidate_cut, p
        return None


def identity():
    return dict(schema_version=1, source=dict(repo_id=REPO, revision=REVISION, config='document', split='train',
                                            license='CC-BY-4.0', files=SOURCE_HASHES),
                model=dict(repo_id='nvidia/DeepSeek-V4-Flash-0731-NVFP4', files=MODEL_HASHES),
                prompt=dict(instruction=INSTRUCTION, thinking_mode='chat', already_chat_encoded=True),
                format=['id', 'prompt', 'input_tokens'], datasets={})


def load_manifest(path):
    value = read_json(path) if path.exists() else identity()
    expected = identity()
    for key in ('schema_version', 'source', 'model', 'prompt', 'format'):
        require(value.get(key) == expected[key], f'Manifest identity mismatch: {key}')
    return value


def validate(ctx, path, entry):
    seen_ids, seen_prompts, seen_reports = set(), set(), set()
    require(sha(path) == entry['sha256'], f'Dataset hash mismatch: {path}')
    records = entry['records']
    require(len(records) == entry['count'], 'Manifest record count mismatch')
    with path.open() as f:
        count = 0
        for i, line in enumerate(f):
            require(i < len(records), 'Too many JSONL rows')
            row, rec = json.loads(line), records[i]
            require(set(row) == {'id', 'prompt', 'input_tokens'}, 'JSONL must contain exactly id/prompt/input_tokens')
            require(isinstance(row['id'], str) and row['id'], 'Invalid ID')
            require(isinstance(row['prompt'], str) and row['prompt'], 'Invalid prompt')
            require(row['id'] == rec['id'] and rec['index'] == i, 'ID/order mismatch')
            require(type(row['input_tokens']) is int and row['input_tokens'] == entry['isl'], 'Input length metadata mismatch')
            doc = ctx.document(rec)
            cut = rec['retained_report_chars']
            require(type(cut) is int and 0 < cut <= len(doc), 'Invalid prefix boundary')
            require(row['prompt'] == ctx.prompt(doc[:cut]), 'Prompt is not the recorded literal source prefix')
            digest = text_sha(row['prompt'])
            require(digest == rec['prompt_sha256'], 'Prompt hash mismatch')
            require(ctx.length(row['prompt']) == entry['isl'], 'Actual tokenizer length mismatch')
            require(row['id'] not in seen_ids and digest not in seen_prompts and rec['report_sha256'] not in seen_reports,
                    'Duplicate ID, final prompt, or source report')
            seen_ids.add(row['id']); seen_prompts.add(digest); seen_reports.add(rec['report_sha256'])
            count += 1
    require(count == entry['count'], f'Expected {entry["count"]} rows, found {count}')
    return dict(status='PASS_OFFLINE', rows=count, unique_reports=len(seen_reports), unique_prompts=len(seen_prompts),
                actual_input_tokens_min=entry['isl'], actual_input_tokens_max=entry['isl'], inference_requests=0)


def publish(ctx, args, rows, records, provenance):
    directory = args.output_dir
    directory.mkdir(parents=True, exist_ok=True)
    name = filename(args.isl, args.count)
    dest, manifest_path = directory / name, directory / MANIFEST_NAME
    require(not dest.exists(), f'Refusing to overwrite {dest}')
    manifest = load_manifest(manifest_path)
    require(name not in manifest['datasets'], f'Manifest already contains {name}')
    require(len(rows) == len(records) == args.count, 'Insufficient final rows; nothing published')
    with tempfile.TemporaryDirectory(prefix='.prepare-', dir=directory) as tmp:
        temporary = Path(tmp) / name
        with temporary.open('x') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        entry = dict(isl=args.isl, count=args.count, bytes=temporary.stat().st_size, sha256=sha(temporary),
                     provenance=provenance, records=records,
                     preparation_script_sha256=sha(__file__),
                     libraries={p: importlib.metadata.version(p) for p in ('transformers', 'tokenizers', 'pyarrow')})
        entry['validation'] = validate(ctx, temporary, entry)
        manifest['datasets'][name] = entry
        temp_manifest = Path(tmp) / MANIFEST_NAME
        write_json(temp_manifest, manifest)
        # Exclusive publication: never silently replace an existing JSONL.
        os.link(temporary, dest)
        try:
            os.replace(temp_manifest, manifest_path)
        except BaseException:
            dest.unlink()
            raise
    emit(phase='published', file=name, sha256=entry['sha256'], validation=entry['validation'])


def record_for(i, rid, source, source_row, doc, cut, prompt):
    return dict(index=i, id=rid, source_file=source, source_row=source_row, report_sha256=text_sha(doc),
                retained_report_chars=cut, prompt_sha256=text_sha(prompt))


def generate(ctx, args):
    require(args.isl > ctx.overhead, f'ISL must exceed prompt overhead {ctx.overhead}')
    candidates, seen, empty, duplicate = [], set(), 0, 0
    for name, docs in ctx.docs.items():
        for i, doc in enumerate(docs):
            if not isinstance(doc, str) or not doc.strip():
                empty += 1
                continue
            h = text_sha(doc)
            if h in seen:
                duplicate += 1
                continue
            seen.add(h)
            candidates.append((name, i))
    require(len(candidates) >= args.count,
            f'Insufficient unique source reports: requested={args.count}, available={len(candidates)}; no output published')
    random.Random(args.seed).shuffle(candidates)
    rows, records, skipped, prompt_hashes = [], [], [], set()
    counts = dict(unique_source_reports=len(candidates), empty=empty, duplicate_source_reports=duplicate,
                  examined=0, too_short=0, no_exact_prefix=0, duplicate_prompt=0)
    for source, source_row in candidates:
        counts['examined'] += 1
        doc = ctx.docs[source][source_row]
        full_length = ctx.length(ctx.prompt(doc))
        if full_length < args.isl:
            counts['too_short'] += 1
            continue
        found = ctx.exact_prefix(doc, args.isl)
        if found is None:
            counts['no_exact_prefix'] += 1
            skipped.append(dict(source_file=source, source_row=source_row, reason='no_exact_prefix_within_bounded_search'))
            continue
        cut, p = found
        digest = text_sha(p)
        if digest in prompt_hashes:
            counts['duplicate_prompt'] += 1
            skipped.append(dict(source_file=source, source_row=source_row, reason='duplicate_final_prompt'))
            continue
        prompt_hashes.add(digest)
        rid = f'govreport-isl{args.isl}-{source.split("-of-")[0]}-row{source_row}'
        records.append(record_for(len(rows), rid, source, source_row, doc, cut, p))
        rows.append(dict(id=rid, prompt=p, input_tokens=args.isl))
        if len(rows) % 128 == 0:
            emit(phase='selected', selected=len(rows), counts=counts)
        if len(rows) == args.count:
            break
    if len(rows) < args.count:
        emit(status='INSUFFICIENT', requested=args.count, achieved=len(rows), counts=counts, skipped=skipped)
        raise ValueError(f'Insufficient exact unique prefixes: requested={args.count}, achieved={len(rows)}; no output published')
    publish(ctx, args, rows, records, dict(operation='generate', seed=args.seed,
            selection='source order then global report deduplication then seeded shuffle; first N exact unique prefixes',
            boundary_search=dict(token_adjustments=8, nearby_characters_each_side=16), counts=counts, skipped=skipped))


def import_legacy(ctx, args):
    old_hash, old_meta_hash = sha(args.input_jsonl), sha(args.source_manifest)
    meta = read_json(args.source_manifest)
    require(isinstance(meta, list) and len(meta) == args.count, 'Legacy manifest count mismatch')
    rows, records, output_budgets = [], [], set()
    with args.input_jsonl.open() as f:
        for i, line in enumerate(f):
            require(i < len(meta), 'Legacy JSONL has more rows than requested')
            original, m = json.loads(line), meta[i]
            require(set(original) <= {'id', 'prompt', 'input_tokens', 'output_tokens'} and
                    {'id', 'prompt', 'input_tokens'} <= set(original), 'Unexpected legacy JSONL fields')
            require(original['id'] == m['id'] and m['selected_index'] == i, 'Legacy ID/order mismatch')
            require(original['input_tokens'] == args.isl, 'Legacy ISL mismatch')
            if 'output_tokens' in original:
                output_budgets.add(original['output_tokens'])
            row = {k: original[k] for k in ('id', 'prompt', 'input_tokens')}
            rec = dict(index=i, id=m['id'], source_file=m['source_file'], source_row=m['source_row'],
                       report_sha256=m.get('report_sha256', m.get('source_sha256')),
                       retained_report_chars=m['retained_report_chars'], prompt_sha256=text_sha(row['prompt']))
            ctx.document(rec)
            rows.append(row); records.append(rec)
    require(len(rows) == args.count, 'Legacy JSONL has fewer rows than requested')
    require(sha(args.input_jsonl) == old_hash and sha(args.source_manifest) == old_meta_hash, 'Legacy inputs changed during import')
    publish(ctx, args, rows, records, dict(operation='import', legacy_file=args.input_jsonl.name,
            legacy_sha256=old_hash, legacy_manifest_sha256=old_meta_hash,
            transformation='Remove output_tokens only; preserve ID, prompt, input_tokens and row order',
            removed_output_budgets=sorted(output_budgets), legacy_inputs_unchanged=True))


def verify_or_reproduce(ctx, args):
    manifest = load_manifest(args.manifest)
    require(args.manifest.is_file() and manifest['datasets'], 'A populated manifest is required')
    names = args.dataset or list(manifest['datasets'])
    if args.command == 'reproduce':
        for name in names:
            require(name in manifest['datasets'] and Path(name).name == name, 'Invalid dataset name')
            require(not (args.output_dir / name).exists(), f'Refusing to overwrite {name}')
    for name in names:
        require(name in manifest['datasets'] and Path(name).name == name, 'Invalid dataset name')
        entry = manifest['datasets'][name]
        if args.command == 'verify':
            result = validate(ctx, args.manifest.parent / name, entry)
        else:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix='.reproduce-', dir=args.output_dir) as tmp:
                path = Path(tmp) / name
                with path.open('x') as f:
                    for record in entry['records']:
                        doc = ctx.document(record)
                        p = ctx.prompt(doc[:record['retained_report_chars']])
                        f.write(json.dumps(dict(id=record['id'], prompt=p, input_tokens=entry['isl']), ensure_ascii=False) + '\n')
                result = validate(ctx, path, entry)
                os.link(path, args.output_dir / name)
        emit(phase=args.command, file=name, sha256=entry['sha256'], validation=result)


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return number


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    for name in ('generate', 'import', 'verify', 'reproduce'):
        s = sub.add_parser(name)
        s.add_argument('--source-dir', type=Path, required=True, help='Local document/ directory containing both pinned train Parquets')
        s.add_argument('--model-dir', type=Path, required=True, help='Local pinned DSV4 tokenizer/encoding directory; weights are not loaded')
        if name in ('generate', 'import', 'reproduce'):
            s.add_argument('--output-dir', type=Path, required=True)
        if name in ('generate', 'import'):
            s.add_argument('--isl', type=positive, required=True)
            s.add_argument('--count', type=positive, required=True)
        if name == 'generate':
            s.add_argument('--seed', type=int, default=0)
        if name == 'import':
            s.add_argument('--input-jsonl', type=Path, required=True)
            s.add_argument('--source-manifest', type=Path, required=True)
        if name in ('verify', 'reproduce'):
            s.add_argument('--manifest', type=Path, required=True)
            s.add_argument('--dataset', action='append', help='Filename from manifest; defaults to all datasets')
    args = p.parse_args()
    try:
        if args.command in ('generate', 'import'):
            dest = args.output_dir / filename(args.isl, args.count)
            require(not dest.exists(), f'Refusing to overwrite {dest}')
            manifest = load_manifest(args.output_dir / MANIFEST_NAME)
            require(dest.name not in manifest['datasets'], f'Manifest already contains {dest.name}')
        ctx = Context(args)
        if args.command == 'generate':
            generate(ctx, args)
        elif args.command == 'import':
            import_legacy(ctx, args)
        else:
            verify_or_reproduce(ctx, args)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        emit(status='ERROR', error=str(exc), inference_requests=0)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
