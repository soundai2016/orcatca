"""Contributor-held-out folds with blocked source-only model selection."""
from __future__ import annotations
from collections import Counter, defaultdict
import math
import numpy as np
from .utils import digest

class UnionFind:

    def __init__(self, keys):
        self.parent = {k: k for k in keys}

    def find(self, key):
        while self.parent[key] != key:
            self.parent[key] = self.parent[self.parent[key]]
            key = self.parent[key]
        return key

    def join(self, a, b):
        a, b = (self.find(a), self.find(b))
        self.parent[max(a, b)] = min(a, b)

def support(rows):
    classes = Counter((r['raw_class'] for r in rows))
    return dict(windows={str(y): sum((r['label'] == y for r in rows)) for y in (0, 1)}, recordings={str(y): len({r['recording'] for r in rows if r['label'] == y}) for y in (0, 1)}, classes=dict(classes), subtype_recordings={c: len({r['recording'] for r in rows if r['raw_class'] == c}) for c in ('KW', 'HW', 'AB')}, temporal_blocks=len({r.get('block', r['recording']) for r in rows}), sites=len({r['site'] for r in rows}))

def enough(rows, cfg):
    s = support(rows)
    return min(s['windows'].values()) >= cfg['min_per_class'] and min(s['recordings'].values()) >= cfg['min_recordings_per_class']

def add_blocks(rows, records, cfg):
    """Join shared content and UTC day blocks; short boundary gaps are purged later."""
    uf = UnionFind({r['recording'] for r in rows})
    entries = defaultdict(dict)
    timestamps = defaultdict(list)
    for r in rows:
        if r.get('recording_utc') is not None:
            timestamps[r['recording']].append(r['recording_utc'])
    starts = {rec: min(values) if values and max(values) - min(values) <= 2.0 else None for rec, values in timestamps.items()}
    for r in rows:
        rec = r['recording']
        start = starts.get(rec)
        duration = records[r['audio']]['frames'] / records[r['audio']]['rate']
        entries[r['site']][rec] = (start, duration)
    dated = set()
    for site, items in entries.items():
        days = {}
        ordered = []
        for rec, (start, duration) in items.items():
            if start is None:
                continue
            dated.add(rec)
            day = math.floor(start / (3600 * cfg['block_hours']))
            if day in days:
                uf.join(rec, days[day])
            else:
                days[day] = rec
            ordered.append((start, start + duration, rec))
    output = [dict(r, block=digest(['temporal-block', uf.find(r['recording'])])[:24], time_metadata_available=r['recording'] in dated, block_utc=starts.get(r['recording']), recording_duration=records[r['audio']]['frames'] / records[r['audio']]['rate']) for r in rows]
    return output

def inner_partition(rows, cfg, fold):
    """Reserve grouped source recordings without withholding a whole confuser-rich source."""
    blocks = defaultdict(list)
    for r in rows:
        blocks[r['block']].append(r)
    site_uf = UnionFind({r['site'] for r in rows})
    for members in blocks.values():
        sites = sorted({r['site'] for r in members})
        for site in sites[1:]:
            site_uf.join(sites[0], site)
    strata = defaultdict(list)
    for block, members in blocks.items():
        strata[site_uf.find(members[0]['site'])].append(block)
    roles = {}
    warnings = []
    for site, ids in sorted(strata.items()):
        whole = [r for b in ids for r in blocks[b]]
        if len(ids) < 5 or not enough(whole, cfg):
            roles.update({b: 'train' for b in ids})
            warnings.append(dict(site=site, reason='insufficient_blocked_dual_class_support', blocks=len(ids)))
            continue
        minimum_blocks = max(cfg['min_recordings_per_class'], int(math.ceil(len(ids) * cfg['min_per_class'] / min(support(whole)['windows'].values()))))
        ns = max(minimum_blocks, int(round(len(ids) * cfg['select_fraction'])))
        nc = max(minimum_blocks, int(round(len(ids) * cfg['calibrate_fraction'])))
        if ns + nc >= len(ids) - 1:
            roles.update({b: 'train' for b in ids})
            warnings.append(dict(site=site, reason='too_few_blocks_for_minimum_support', blocks=len(ids)))
            continue
        required = {c for c in ('KW', 'HW', 'AB') if len({r['block'] for r in whole if r['raw_class'] == c}) >= 3}
        best = None
        for trial in range(128):
            order = sorted(ids, key=lambda block: digest([cfg['split_seed'], fold, site, trial, block]))
            proposal = {b: 'select' if j < ns else 'calibrate' if j < ns + nc else 'train' for j, b in enumerate(order)}
            parts = {role: [r for b in ids if proposal[b] == role for r in blocks[b]] for role in ('train', 'select', 'calibrate')}
            if not all((enough(v, cfg) for v in parts.values())):
                continue
            if any((not any((r['raw_class'] == c for r in part)) for c in required for part in parts.values())):
                continue
            total = Counter((r['raw_class'] for r in whole))
            fractions = {'train': 1 - cfg['select_fraction'] - cfg['calibrate_fraction'], 'select': cfg['select_fraction'], 'calibrate': cfg['calibrate_fraction']}
            cost = sum((abs(sum((r['raw_class'] == c for r in part)) / total[c] - fractions[role]) for role, part in parts.items() for c in total))
            candidate = (cost, trial, proposal)
            if best is None or candidate[:2] < best[:2]:
                best = candidate
        if best is None:
            roles.update({b: 'train' for b in ids})
            warnings.append(dict(site=site, reason='no_feasible_blocked_inner_split', blocks=len(ids)))
        else:
            roles.update(best[2])
    from .support import repair_source_support
    roles, coverage_report = repair_source_support(blocks, roles, cfg, fold)
    warnings.extend(coverage_report)
    candidates = defaultdict(dict)
    for row in rows:
        if row.get('time_metadata_available') and row.get('block_utc') is not None:
            candidates[row['site']][row['recording']] = (row['block_utc'], row['recording_duration'], roles[row['block']])
    removed = set()
    priority = {'train': 0, 'select': 1, 'calibrate': 2}
    for site, items in candidates.items():
        active = []
        for begin, duration, role, rec in sorted(((a, b, c, r) for r, (a, b, c) in items.items())):
            active = [v for v in active if v[0] + cfg['session_gap_seconds'] >= begin]
            for end, other_role, other_rec in active:
                if role != other_role:
                    removed.add(rec if priority[role] < priority[other_role] else other_rec)
            active.append((begin + duration, role, rec))
    result = {role: [r for r in rows if roles[r['block']] == role and r['recording'] not in removed] for role in ('train', 'select', 'calibrate')}
    if removed:
        warnings.append(dict(reason='temporal_boundary_purge', recordings=sorted(removed), windows=sum((r['recording'] in removed for r in rows))))
    if not all((enough(v, cfg) for v in result.values())):
        raise ValueError(f'Insufficient blocked source support: { {k: support(v) for k, v in result.items()}}')
    for subtype in ('KW', 'HW', 'AB'):
        n = len({r['recording'] for r in result['train'] if r['raw_class'] == subtype})
        if n < cfg['min_train_subtype_recordings']:
            raise ValueError(f'Training subtype {subtype} has only {n} recordings; do not invent negative labels')
    warnings.append(dict(reason='final_source_subtype_support', support={role: support(rr) for role, rr in result.items()}))
    return (result, warnings)

def make_splits(rows, records, cfg):
    from .data import family
    families = {family(p) for r in rows for p in r.get('provider_links', [r['provider']])}
    uf = UnionFind(families)
    owner = {}
    for r in rows:
        links = [family(p) for p in r.get('provider_links', [r['provider']])]
        for g in links[1:]:
            uf.join(links[0], g)
        g = links[0]
        for key in ('audio', 'recording'):
            value = (key, r[key])
            if value in owner:
                uf.join(g, owner[value])
            owner[value] = g
    grouped = defaultdict(list)
    rows = add_blocks(rows, records, cfg)
    for r in rows:
        g = uf.find(family(r['provider']))
        grouped[g].append(dict(r, group=g))
    eligible = sorted([g for g, rr in grouped.items() if enough(rr, cfg)], key=lambda g: digest([cfg['split_seed'], g]))
    if len(eligible) < cfg['folds'] or len(grouped) < 3:
        raise ValueError(f'Insufficient source components: { {g: support(rr) for g, rr in grouped.items()}}')
    folds = []
    for index, held in enumerate(eligible[:cfg['folds']]):
        source = [r for g, rr in grouped.items() if g != held for r in rr]
        parts, warnings = inner_partition(source, cfg, index)
        parts['holdout'] = grouped[held]
        manifest = [dict(r, split=role) for role, rr in parts.items() for r in rr]
        validate_splits(manifest)
        folds.append((manifest, dict(index=index, holdout_source=held, inner_policy='blocked_source_with_subtype_coverage', support={role: support(rr) for role, rr in parts.items()}, train_only_sites=warnings, missing_time_recordings=len({r['recording'] for r in manifest if not r['time_metadata_available']}))))
    return folds

def validate_splits(rows):
    owner = {}
    groups = defaultdict(set)
    for r in rows:
        if r['split'] not in {'train', 'select', 'calibrate', 'holdout'}:
            raise ValueError('Unknown split')
        for key in ('recording', 'audio', 'block'):
            value = r.get(key, r['recording'])
            if owner.setdefault((key, value), r['split']) != r['split']:
                raise ValueError(f'{key} leakage between roles')
        groups[r['group']].add(r['split'])
    if any(('holdout' in roles and len(roles) > 1 for roles in groups.values())):
        raise ValueError('Outer source leakage')
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate sample ids')
