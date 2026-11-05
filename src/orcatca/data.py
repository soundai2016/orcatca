"""All-file indexing and deterministic, source-isolated annotated-window protocols.

Use the archive labels and retain the reason for every excluded window.
Unreadable recordings stay on disk. No absent-annotation negatives are created.
"""
from __future__ import annotations
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import sqlite3
from urllib.parse import unquote
import numpy as np
import soundfile as sf
from .audio import load_wave
from .container_check import inspect_container, compare_decoder_geometry
from .logging import event
from .utils import DATA_NAME, atomic_json, digest, output_path, raw_path, sha256, write_rows, read_rows, write_csv, read_json
EXTENSIONS = {'.wav', '.flac', '.aif', '.aiff', '.w64'}

def _probe(item):
    root, rel, old = item
    candidate = root / DATA_NAME / rel
    if candidate.is_symlink():
        st = candidate.lstat()
        return (dict(audio=rel, size=st.st_size, mtime_ns=st.st_mtime_ns, error='unverified_symlink_recording'), False)
    try:
        p = raw_path(root, rel)
        st = p.stat()
    except OSError as exc:
        return (dict(audio=rel, size=None, mtime_ns=None, error=f'unreadable_file_identity: {exc}'), False)
    if old and (old['size'], old['mtime_ns']) == (st.st_size, st.st_mtime_ns):
        return (old, True)
    r = dict(audio=rel, size=st.st_size, mtime_ns=st.st_mtime_ns, error=None)
    try:
        container = inspect_container(p)
        with sf.SoundFile(str(p), 'r') as f:
            info = dict(frames=len(f), samplerate=f.samplerate, channels=f.channels)
            compare_decoder_geometry(container, info)
            if len(f) < 1 or f.samplerate < 1:
                raise ValueError('Empty recording')
            for start in sorted({0, max(0, len(f) // 2 - 256), max(0, len(f) - 512)}):
                f.seek(start)
                n = min(512, len(f) - start)
                x = f.read(n, dtype='float32', always_2d=True)
                if len(x) != n or not np.isfinite(x).all():
                    raise ValueError('Short/nonfinite probe')
            r.update(rate=f.samplerate, channels=f.channels, frames=len(f), format=f.format, subtype=f.subtype)
        r['sha256'] = sha256(p)
        after = p.stat()
        if (after.st_size, after.st_mtime_ns) != (st.st_size, st.st_mtime_ns):
            raise ValueError('File changed while hashing')
    except Exception as exc:
        r['error'] = f'{type(exc).__name__}: {exc}'
        r.pop('sha256', None)
    return (r, False)

def inventory(root, cfg):
    raw = (root / DATA_NAME).resolve()
    if not raw.is_dir():
        raise FileNotFoundError(f'Dataset is missing: {raw}')
    out = output_path(root, 'state')
    out.mkdir(parents=True, exist_ok=True)
    db = out / 'index.sqlite'
    con = sqlite3.connect(db)
    con.execute('CREATE TABLE IF NOT EXISTS files (audio TEXT PRIMARY KEY, payload TEXT NOT NULL)')
    con.execute('CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
    oldmeta = dict(con.execute('SELECT key,value FROM meta'))
    if oldmeta and (oldmeta.get('format') != 'orcatca2' or oldmeta.get('root') != str(raw)):
        con.close()
        raise RuntimeError('Foreign index: only this fresh project index is accepted')
    con.executemany('INSERT OR REPLACE INTO meta VALUES (?,?)', [('format', 'orcatca2'), ('root', str(raw)), ('complete', 'false')])
    con.commit()
    old = {k: json.loads(v) for k, v in con.execute('SELECT audio,payload FROM files')}
    paths, walk_errors = ([], [])
    for directory, subdirs, names in os.walk(raw, followlinks=False, onerror=lambda e: walk_errors.append(str(e))):
        subdirs[:] = sorted((d for d in subdirs if not Path(directory, d).is_symlink()))
        for name in sorted(names):
            p = Path(directory, name)
            if p.suffix.lower() in EXTENSIONS:
                paths.append(p.relative_to(raw).as_posix())
    if walk_errors or not paths:
        con.close()
        raise RuntimeError(f'Incomplete/empty corpus traversal: {walk_errors[:10]}')
    if set(old) - set(paths):
        con.close()
        raise RuntimeError('Previously indexed recordings disappeared; inspect raw data before rebuilding')
    event('inventory_started', total=len(paths), existing=len(old), full_stream_hash_for_new_files=True)
    results, reused = ({}, 0)
    try:
        with ThreadPoolExecutor(max_workers=cfg['index_workers']) as pool:
            for i, (row, cached) in enumerate(pool.map(_probe, ((root, rel, old.get(rel)) for rel in sorted(paths))), 1):
                results[row['audio']] = row
                reused += int(cached)
                con.execute('INSERT OR REPLACE INTO files VALUES (?,?)', (row['audio'], json.dumps(row, sort_keys=True)))
                if i % 100 == 0 or i == len(paths):
                    con.commit()
                    event('inventory_progress', processed=i, total=len(paths), reused=reused)
        con.execute("INSERT OR REPLACE INTO meta VALUES ('complete','true')")
        con.commit()
    finally:
        con.close()
    errors = [r for r in results.values() if r['error']]
    summary = dict(total=len(results), accepted=len(results) - len(errors), excluded=len(errors), reused=reused, raw_modified=False, verification='container/header + finite probes + content hash; not remote completeness or full-stream CRC')
    atomic_json(out / 'inventory.json', summary)
    write_csv(out / 'audio_exclusions.csv', errors)
    if not summary['accepted']:
        raise ValueError('No readable recordings')
    event('inventory_complete', **summary)
    return results

def _text(v):
    value = '' if v is None else str(v).strip()
    return '' if value.lower() in {'na', 'nan', 'none', 'null'} else value

def _number(v):
    if str(v).strip().lower() in {'true', 'false'}:
        return float(str(v).strip().lower() == 'true')
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None

def resolve(raw, name, records, basenames):
    p = unquote(_text(raw)).replace('\\', '/')
    for marker in ('noaa-passive-bioacoustic/', 'dclde/2027/', 'dclde/2026/', DATA_NAME + '/'):
        if marker in p:
            p = p.split(marker, 1)[1]
    p = p.lstrip('/')
    name = _text(name)
    if '..' in PurePosixPath(p).parts:
        raise ValueError('path_traversal')
    if not p:
        p = name
    elif name and (not PurePosixPath(p).suffix):
        p = p.rstrip('/') + '/' + name
    if p in records:
        return p
    options = basenames.get(PurePosixPath(p).name, [])
    suffix = [v for v in options if v.endswith('/' + p)]
    if len(suffix) == 1:
        return suffix[0]
    if len(options) == 1:
        return options[0]
    raise ValueError('audio_mapping_ambiguous' if options else 'audio_mapping_missing')

class Intervals:

    def __init__(self, rows):
        self.rows = sorted(rows)
        self.starts = [x[0] for x in self.rows]
        self.ends = []
        end = -float('inf')
        for x in self.rows:
            end = max(end, x[1])
            self.ends.append(end)

    def query(self, a, b):
        right = bisect_left(self.starts, b)
        left = bisect_right(self.ends, a, 0, right)
        return [r for r in self.rows[left:right] if r[1] > a]

def annotation_candidates(root, cfg, records):
    table = (root / DATA_NAME / 'Annotations.csv').resolve()
    if not table.is_file():
        raise FileNotFoundError('Expected corpus-root Annotations.csv; no provider-table merging is performed')
    basenames = defaultdict(list)
    for rel in records:
        basenames[PurePosixPath(rel).name].append(rel)
    rows, errors, invalid_timing, intervals = ([], [], set(), defaultdict(list))
    required = {'filepath', 'soundfile', 'classspecies', 'filebeginsec', 'fileendsec', 'provider', 'dataset', 'kw', 'kw_certain', 'annotationlevel'}
    with table.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if not required <= {x.strip().lower() for x in reader.fieldnames or []}:
            raise ValueError('Aggregate CSV fields are incomplete')
        for line, original in enumerate(reader, 2):
            r = {k.strip().lower(): v for k, v in original.items() if k is not None}
            rel = None
            try:
                rel = resolve(r['filepath'], r['soundfile'], records, basenames)
                info = records[rel]
                if info['error']:
                    raise ValueError('unverified_audio')
                a, b = (_number(r['filebeginsec']), _number(r['fileendsec']))
                if a is None or b is None or a < 0 or (b <= a) or (b > info['frames'] / info['rate'] + 1 / info['rate']):
                    invalid_timing.add(rel)
                    raise ValueError('invalid_timing')
                label = {'abiotic': 'AB', 'killer whale': 'KW', 'humpback whale': 'HW'}.get(_text(r['classspecies']).lower(), _text(r['classspecies']).upper())
                kw = _number(r['kw'])
                intervals[rel].append((a, b, label, kw if kw is not None else -1.0))
                rows.append(dict(audio=rel, begin=a, end=b, raw_class=label, kw=kw, certain=_number(r['kw_certain']), utc=_text(r.get('utc')), level=_text(r['annotationlevel']).lower(), provider=_text(r['provider']), dataset=_text(r['dataset']), annotation_line=line))
            except (ValueError, KeyError) as exc:
                errors.append(dict(line=line, audio=rel, reason=str(exc)))
    indexes = {k: Intervals(v) for k, v in intervals.items()}
    selected = {}
    conflicts = set()
    label_errors = []
    for r in rows:
        info = records[r['audio']]
        why = None
        frames = round(cfg['seconds'] * info['rate'])
        start = min(max(0, round(((r['begin'] + r['end']) / 2 - cfg['seconds'] / 2) * info['rate'])), info['frames'] - frames)
        if not r['provider'] or not r['dataset']:
            why = 'missing_source'
        elif 'orcasound' in r['provider'].lower():
            why = 'non_srkw_is_not_no_kw'
        elif r['level'] not in {'call', 'detection'}:
            why = 'nonlocalized_annotation'
        elif info['channels'] != 1:
            why = 'unreviewed_multichannel'
        elif info['rate'] < cfg['sample_rate']:
            why = 'native_rate_too_low'
        elif start < 0:
            why = 'recording_too_short'
        elif r['raw_class'] == 'KW':
            if r['kw'] != 1 or r['certain'] != 1:
                why = 'uncertain_kw'
            elif r['end'] - r['begin'] > cfg['seconds']:
                why = 'positive_interval_longer_than_window'
        elif r['raw_class'] in {'AB', 'HW'}:
            if r['kw'] != 0:
                why = 'negative_absence_flag_missing'
            elif r['audio'] in invalid_timing:
                why = 'invalid_timing_prevents_negative_assessment'
            elif any((c in {'KW', 'UNDBIO'} or flag == 1 for _, _, c, flag in indexes[r['audio']].query(start / info['rate'], (start + frames) / info['rate']))):
                why = 'negative_overlaps_possible_kw'
        else:
            why = 'unknown_class'
        if why:
            label_errors.append(dict(line=r['annotation_line'], audio=r['audio'], reason=why))
            continue
        sid = digest([r['audio'], start, frames])[:24]
        s = dict(id=sid, audio=r['audio'], start_frame=start, label=int(r['raw_class'] == 'KW'), raw_class=r['raw_class'], provider=r['provider'], dataset=r['dataset'], site=r['provider'] + '::' + r['dataset'], recording=info['sha256'], annotation_line=r['annotation_line'], recording_utc=utc_start(r.get('utc'), r['begin']))
        if s['label'] == 0 and any((c == 'HW' for _, _, c, _ in indexes[r['audio']].query(start / info['rate'], (start + frames) / info['rate']))):
            s['raw_class'] = 'HW'
        if sid in selected and any((selected[sid][k] != s[k] for k in ('label', 'provider', 'dataset'))):
            conflicts.add(sid)
        selected[sid] = s
    for sid in conflicts:
        label_errors.append(dict(line=selected[sid]['annotation_line'], audio=selected[sid]['audio'], reason='conflicting_duplicate_window'))
        del selected[sid]
    event('annotations_filtered', normalized=len(rows), candidates=len(selected), structural_exclusions=len(errors), label_exclusions=len(label_errors))
    unique = {}
    bad_content = set()
    for row in sorted(selected.values(), key=lambda x: x['id']):
        key = (row['recording'], row['start_frame'])
        if key in unique:
            other = unique[key]
            if (other['label'], other['raw_class']) != (row['label'], row['raw_class']):
                bad_content.add(key)
            other['provider_links'] = sorted(set(other.get('provider_links', [other['provider']]) + [row['provider']]))
            label_errors.append(dict(line=row['annotation_line'], audio=row['audio'], reason='duplicate_content_window'))
        else:
            unique[key] = row
    for key in bad_content:
        row = unique.pop(key)
        label_errors.append(dict(line=row['annotation_line'], audio=row['audio'], reason='conflicting_content_window'))
    return (sorted(unique.values(), key=lambda x: x['id']), errors + label_errors, sha256(table))

def family(provider):
    p = re.sub('[^A-Z0-9]+', '_', provider.upper()).strip('_')
    tokens = set(p.split('_'))
    if tokens & {'JASCO', 'VFPA', 'VPFA', 'ONC'}:
        return 'JASCO_VFPA_ONC'
    if 'DFO' in tokens:
        return 'DFO'
    return p

def utc_start(value, offset):
    if not value:
        return None
    from datetime import datetime, timezone
    try:
        stamp = datetime.fromisoformat(value.strip().replace('Z', '+00:00'))
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        if not 1990 <= stamp.year <= 2035:
            return None
        return stamp.timestamp() - offset
    except (ValueError, TypeError):
        return None
from .splitting import make_splits, validate_splits

def prepare(root, cfg):
    out = output_path(root, 'state/protocol')
    out.mkdir(parents=True, exist_ok=True)
    records = inventory(root, cfg)
    identity = digest([cfg, {k: v for k, v in sorted(records.items())}, sha256(root / DATA_NAME / 'Annotations.csv')])
    state = out / 'complete.json'
    if state.exists():
        info = read_json(state)
        if info['identity'] != identity:
            raise ValueError('Protocol inputs changed. Preserve the existing protocol and use a separate project for changed data settings.')
        for name, h in info['files'].items():
            if sha256(out / name) != h:
                raise ValueError('Protocol output changed: ' + name)
        event('protocol_reused', folds=len(info['folds']))
        return info
    candidates, errors, ann_hash = annotation_candidates(root, cfg, records)
    event('decode_check_started', clips=len(candidates), note='Includes holdout eligibility, before splitting; no holdout fitting/statistics')
    failed = {}

    def decode(row):
        try:
            load_wave(root, row, records[row['audio']], cfg)
            return None
        except Exception as exc:
            return (row['audio'], f'{type(exc).__name__}: {exc}')
    with ThreadPoolExecutor(max_workers=cfg['decode_workers']) as pool:
        for i, result in enumerate(pool.map(decode, candidates), 1):
            if result:
                failed[result[0]] = result[1]
            if i % 2000 == 0 or i == len(candidates):
                event('decode_check_progress', checked=i, total=len(candidates), failed_recordings=len(failed))
    for r in candidates:
        if r['audio'] in failed:
            errors.append(dict(line=r['annotation_line'], audio=r['audio'], reason='selected_clip_decode_failure'))
    rows = [r for r in candidates if r['audio'] not in failed]
    folds = make_splits(rows, records, cfg)
    used = {r['audio'] for r in rows}
    selected_records = {rel: records[rel] for rel in sorted(used)}
    atomic_json(out / 'recordings.json', selected_records)
    write_csv(out / 'annotation_exclusions.csv', errors)
    atomic_json(out / 'decode_failures.json', failed)
    paths = ['recordings.json', 'annotation_exclusions.csv', 'decode_failures.json']
    reports = []
    for i, (manifest, report) in enumerate(folds):
        name = f'fold{i:02d}.jsonl.gz'
        write_rows(out / name, sorted(manifest, key=lambda r: r['id']))
        paths.append(name)
        report['manifest'] = name
        report['manifest_sha256'] = sha256(out / name)
        reports.append(report)
    info = dict(format='orcatca230', identity=identity, annotations_sha256=ann_hash, folds=reports, files={p: sha256(out / p) for p in paths}, excluded_reasons=dict(Counter((r['reason'] for r in errors))), confirmatory=False, labels='archive annotations with deterministic overlap and conflict filters', groups='observed contributor families joined by shared recordings/content; not verified encounters or physical deployments', prior_use='This corpus has been used for development; regrouping does not create new external confirmation data.')
    atomic_json(state, info)
    event('protocol_complete', folds=reports)
    return info
