"""Atomic JSON and immutable, checksummed trial bundles.

Each trial is committed by atomically writing its checksum file last. A crashed
attempt retains its directory and journal; it never becomes a new sample.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
from .domain import canonical


def read_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def read_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def write_jsonl(path: str | Path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(canonical(row) + "\n")
    os.replace(temp, path)


def file_hash(path: str | Path) -> str:
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def checksum_tree(path: Path) -> dict:
    return {str(p.relative_to(path)).replace("\\", "/"): file_hash(p)
            for p in sorted(path.rglob("*")) if p.is_file() and p.name != "checksums.json"}


def verify_tree(path: Path) -> None:
    if read_json(path / "checksums.json") != checksum_tree(path):
        raise ValueError(f"Checksum mismatch: {path}")
