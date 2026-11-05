"""Small filesystem primitives. Corpus reads and experiment writes never overlap."""
from __future__ import annotations
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any
DATA_NAME = 'dclde_2027_killer_whales'

def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def project_root() -> Path:
    root = Path(os.environ.get('ORCATCA_ROOT', Path(__file__).resolve().parents[2])).resolve()
    if not (root / 'pyproject.toml').is_file():
        raise RuntimeError('Project root is missing pyproject.toml; set ORCATCA_ROOT.')
    return root

def output_path(root: Path, relative: str | Path) -> Path:
    root = root.resolve()
    p = (root / relative).resolve()
    raw = (root / DATA_NAME).resolve()
    if not p.is_relative_to(root) or p == root or p.is_relative_to(raw) or raw.is_relative_to(p):
        raise ValueError(f'Unsafe output path: {p}')
    return p

def raw_path(root: Path, relative: str) -> Path:
    rel = Path(relative)
    base = (root / DATA_NAME).resolve()
    p = (base / rel).resolve()
    if rel.is_absolute() or '..' in rel.parts or (not p.is_relative_to(base)):
        raise ValueError('Audio path escapes the corpus')
    return p

def atomic_json(path: Path, obj: Any):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(obj, f, indent=2, sort_keys=True, allow_nan=False)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write_rows(path: Path, rows: list[dict]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with gzip.GzipFile(filename=str(tmp), mode='wb', mtime=0) as f:
        for row in rows:
            f.write((json.dumps(row, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode())
    os.replace(tmp, path)

def read_rows(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys((k for r in rows for k in r)))
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

def code_digest(root: Path) -> str:
    return digest({str(p.relative_to(root)): sha256(p) for p in sorted((root / 'src').rglob('*.py'))})

def process_identity(pid: int):
    try:
        text = Path(f'/proc/{pid}/stat').read_text()
        fields = text[text.rfind(')') + 2:].split()
        if fields[0] == 'Z':
            return None
        return {'pid': pid, 'start_ticks': fields[19], 'pgid': int(fields[2])}
    except (OSError, IndexError, ValueError):
        return None
