"""Repair source subtype coverage without inspecting scores or outer labels."""
from __future__ import annotations
from collections import defaultdict
from .utils import digest


def coverage(blocks, roles):
    result = {}
    for role in ('train', 'select', 'calibrate'):
        result[role] = {}
        for subtype in ('KW', 'HW', 'AB'):
            members = [(b, r) for b, rr in blocks.items() if roles[b] == role
                       for r in rr if r['raw_class'] == subtype]
            result[role][subtype] = dict(windows=len(members),
                recordings=len({r['recording'] for _, r in members}),
                blocks=len({b for b, _ in members}))
    return result


def repair_source_support(blocks, roles, cfg, fold):
    """Move whole training blocks only; the outer source is not an input."""
    if not cfg.get('repair_subtype_coverage', True):
        return roles, []
    if any(r.get('split') == 'holdout' for rr in blocks.values() for r in rr):
        raise ValueError('Outer rows cannot enter source support repair')
    roles = dict(roles)
    details = {b: {c: {r['recording'] for r in rr if r['raw_class'] == c}
                  for c in ('KW', 'HW', 'AB')} for b, rr in blocks.items()}
    sizes = {b: len(rr) for b, rr in blocks.items()}
    min_b = int(cfg.get('min_eval_subtype_blocks', 2))
    min_r = int(cfg.get('min_eval_subtype_recordings', 2))
    train_min = int(cfg.get('min_train_subtype_recordings', 2))
    moved = []

    def resources(role, cls):
        selected = [b for b in blocks if roles[b] == role and details[b][cls]]
        return set(selected), set().union(*(details[b][cls] for b in selected)) if selected else set()

    # Alternation prevents one role consuming all rare subtype blocks.
    for _ in range(24):
        changed = False
        for cls in ('HW', 'AB', 'KW'):
            for role in ('calibrate', 'select'):
                bb, rr = resources(role, cls)
                if len(bb) >= min_b and len(rr) >= min_r:
                    continue
                candidates = []
                for b in blocks:
                    if roles[b] != 'train' or not details[b][cls]:
                        continue
                    safe = True
                    for other in ('KW', 'HW', 'AB'):
                        others = [k for k in blocks if roles[k] == 'train' and k != b]
                        retained = set().union(*(details[k][other] for k in others)) if others else set()
                        if len(retained) < train_min:
                            safe = False
                            break
                    if safe:
                        gain = min(len(details[b][cls] - rr), max(1, min_r-len(rr)))
                        candidates.append((-gain, sizes[b], digest([cfg['split_seed'], fold, role, cls, b]), b))
                if candidates:
                    b = min(candidates)[-1]
                    roles[b] = role
                    moved.append(dict(block=b, to=role, reason_subtype=cls, windows=sizes[b]))
                    changed = True
        if not changed:
            break
    observed = coverage(blocks, roles)
    deficits = []
    for role in ('select', 'calibrate'):
        for cls in ('KW', 'HW', 'AB'):
            s = observed[role][cls]
            if s['blocks'] < min_b or s['recordings'] < min_r:
                deficits.append(dict(role=role, subtype=cls, **s))
    return roles, [dict(reason='source_subtype_coverage', moved=moved, support=observed,
                        remaining_deficits=deficits, uses_prediction_scores=False,
                        uses_outer_source=False)]
