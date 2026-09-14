#!/usr/bin/env python3
"""Create a source archive without historical results, caches or environments."""
from __future__ import annotations

import argparse
import hashlib
import tarfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = (args.output or root / "dist/llm-serving-benchmarks-source.tar.gz").expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    excluded = {".git", ".venv", "__pycache__", "dist", "build", ".DS_Store"}
    with tarfile.open(output, "w:gz") as archive:
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root)
            if path.is_symlink() or not path.is_file() or path == output:
                continue
            if any(p in excluded or p.endswith(".egg-info") for p in relative.parts):
                continue
            if relative.parts[0] == "results" and relative.name != ".gitkeep":
                continue
            archive.add(path, arcname=str(Path(root.name) / relative), recursive=False)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_name(output.name + ".sha256").write_text(f"{digest}  {output.name}\n")
    print(output)
    print(digest)


if __name__ == "__main__":
    main()
