"""Read-only audits of frozen window manifests and prediction metadata.

Counts describe the retained protocol, not exclusions from the original archive.
Linked units are identity components, not verified independent encounters.
No audio, weights, scores, or training configuration is changed or loaded here.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
import hashlib
import json
import math
from numbers import Real

ROLES = ('train', 'select', 'calibrate', 'holdout')
CLASSES = ('KW', 'HW', 'AB')
_REQUIRED = ('id', 'recording', 'audio', 'start_frame', 'block', 'site', 'group',
             'split', 'raw_class', 'label')
_PREDICTION_FIELDS = ('recording', 'site', 'group', 'block', 'label', 'raw_class')
_MANIFEST_FIELDS = ('id', 'audio', 'start_frame', 'recording', 'block', 'site',
                    'group', 'provider', 'provider_links', 'dataset', 'split',
                    'label', 'raw_class', 'annotation_line', 'recording_utc',
                    'block_utc', 'time_metadata_available', 'recording_duration')


def _present(value):
    return value is not None and value != ''


def _finite(value):
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def _text(value, missing='__missing__'):
    return str(value) if _present(value) else missing


def _fold_name(fold):
    return f'fold{fold:02d}' if isinstance(fold, int) else str(fold)


def _linked_units(rows):
    """Resolve transitive recording/audio/block bridges before stratifying rows."""
    parent = {}

    def find(key):
        parent.setdefault(key, key)
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    identities = []
    for i, row in enumerate(rows):
        keys = [(field, str(row[field])) for field in ('recording', 'audio', 'block')
                if _present(row.get(field))]
        keys = keys or [('missing_identity', str(i))]
        for key in keys[1:]:
            a, b = find(keys[0]), find(key)
            if a != b:
                parent[max(a, b)] = min(a, b)
        find(keys[0])
        identities.append(keys)
    members = defaultdict(set)
    for keys in identities:
        members[find(keys[0])].update(keys)
    names = {key: hashlib.sha256(json.dumps(sorted(tokens)).encode()).hexdigest()[:24]
             for key, tokens in members.items()}
    return [names[find(keys[0])] for keys in identities]


def _recording_times(rows):
    grouped = defaultdict(list)
    for row in rows:
        if _present(row.get('recording')):
            grouped[str(row['recording'])].append(row)
    answer = {}
    for recording, members in grouped.items():
        declared = [r.get('time_metadata_available') for r in members]
        starts = [r.get('block_utc') for r in members]
        durations = [r.get('recording_duration') for r in members]
        usable = (all(flag is True for flag in declared)
                  and all(_finite(t) for t in starts)
                  and max(starts) - min(starts) <= 2.0)
        duration_ok = (all(_finite(d) and d > 0 for d in durations)
                       and max(durations) - min(durations) <= 1e-6)
        answer[recording] = dict(
            usable=usable,
            start=min(starts) if usable else None,
            duration=max(durations) if duration_ok else None,
            declared_available=any(flag is True for flag in declared),
            inconsistent_flags=len({str(flag) for flag in declared}) > 1)
    return answer


def _prediction_audit(rows, prediction_metadata):
    if prediction_metadata is None:
        return dict(status='not_provided', roles={}, errors=[])
    if not isinstance(prediction_metadata, Mapping):
        raise TypeError('prediction_metadata must map protocol roles to metadata rows')
    if not prediction_metadata:
        return dict(status='not_provided', roles={}, errors=[])
    errors, reports, visited = [], {}, set()
    for supplied_role, observed in prediction_metadata.items():
        role = 'calibrate' if supplied_role == 'calibration' else supplied_role
        if role not in ROLES or role in visited:
            errors.append(dict(code='invalid_prediction_role', role=str(supplied_role)))
            continue
        visited.add(role)
        expected = [r for r in rows if r.get('split') == role]
        observed = list(observed)
        counts = Counter(_text(r.get('id')) for r in observed)
        duplicates = sorted(key for key, n in counts.items() if n > 1)
        lookup = {_text(r.get('id')): r for r in expected}
        seen = set(counts)
        missing, extra = sorted(set(lookup) - seen), sorted(seen - set(lookup))
        mismatches = []
        for row in observed:
            sid = _text(row.get('id'))
            if sid in lookup:
                fields = [key for key in _PREDICTION_FIELDS
                          if row.get(key) != lookup[sid].get(key)]
                if fields:
                    mismatches.append(dict(id=sid, fields=fields))
        report = dict(expected_windows=len(lookup), prediction_rows=len(observed),
                      duplicate_ids=len(duplicates), missing_ids=len(missing),
                      extra_ids=len(extra), metadata_mismatches=len(mismatches))
        reports[role] = report
        for code, values in (('duplicate_prediction_ids', duplicates),
                             ('missing_prediction_ids', missing),
                             ('extra_prediction_ids', extra),
                             ('prediction_metadata_mismatch', mismatches)):
            if values:
                errors.append(dict(code=code, role=role, count=len(values), examples=values[:20]))
    return dict(status='aligned' if not errors else 'invalid', roles=reports, errors=errors,
                scores_examined=False)


def audit_protocol(rows, fold, prediction_metadata=None, *,
                   temporal_gap_seconds=1800, block_hours=24):
    """Return JSON/CSV-ready flow rows and an explicit, non-certifying audit.

    ``rows`` are the full ``CombinedStore.protocol(fold)`` manifest. Optional
    prediction metadata is a role-to-rows mapping, e.g. ``{'calibrate': cal,
    'holdout': held}``. Use the frozen configuration's gap and block hours.
    Duplicate IDs, including identical duplicates, invalidate the audit.
    Multiple labels in one recording are allowed; conflicting labels for the
    same window ID or (file-content hash, start frame) are not.
    """
    if not _finite(temporal_gap_seconds) or temporal_gap_seconds < 0:
        raise ValueError('temporal_gap_seconds must be finite and nonnegative')
    if not _finite(block_hours) or block_hours <= 0:
        raise ValueError('block_hours must be finite and positive')
    rows = list(rows)
    if any(not isinstance(row, Mapping) for row in rows):
        raise TypeError('Protocol rows must be mappings')
    fold = _fold_name(fold)
    errors = []

    def issue(code, examples):
        if examples:
            errors.append(dict(code=code, count=len(examples), examples=list(examples)[:20]))

    missing = {key: sum(not _present(row.get(key)) for row in rows) for key in _REQUIRED}
    issue('missing_required_metadata', [dict(field=k, rows=n) for k, n in missing.items() if n])
    issue('unknown_role', [dict(id=r.get('id'), role=r.get('split')) for r in rows
                           if r.get('split') not in ROLES])
    issue('inconsistent_label', [dict(id=r.get('id'), raw_class=r.get('raw_class'),
                                     label=r.get('label')) for r in rows
                                 if r.get('raw_class') not in CLASSES
                                 or r.get('label') != int(r.get('raw_class') == 'KW')])
    issue('invalid_window_start_frame', [dict(id=r.get('id')) for r in rows
          if not (_finite(r.get('start_frame')) and r['start_frame'] >= 0
                  and int(r['start_frame']) == r['start_frame'])])
    ids, content = defaultdict(list), defaultdict(list)
    identities = {key: defaultdict(list) for key in ('recording', 'audio', 'block', 'group')}
    for i, row in enumerate(rows):
        if _present(row.get('id')):
            ids[str(row['id'])].append(i)
        frame = row.get('start_frame')
        if _present(row.get('recording')) and _finite(frame) and frame >= 0 and int(frame) == frame:
            content[str(row['recording']), int(frame)].append(i)
        for key, mapping in identities.items():
            if _present(row.get(key)):
                mapping[str(row[key])].append(i)
    duplicates = sorted(key for key, indices in ids.items() if len(indices) > 1)
    issue('duplicate_window_id', duplicates)
    conflicting = [key for key, indices in ids.items()
                   if len({(str(rows[i].get('label')), str(rows[i].get('raw_class')))
                           for i in indices}) > 1]
    issue('conflicting_window_labels', conflicting)
    duplicate_content = [dict(recording=key[0], start_frame=key[1],
                              ids=sorted({_text(rows[i].get('id')) for i in indices}),
                              labels=sorted({_text(rows[i].get('raw_class')) for i in indices}))
                         for key, indices in content.items() if len(indices) > 1]
    issue('duplicate_content_window', duplicate_content)

    for key, mapping in identities.items():
        crossings = []
        for value, indices in mapping.items():
            roles = sorted({_text(rows[i].get('split')) for i in indices})
            if len(roles) > 1 and (key != 'group' or 'holdout' in roles):
                crossings.append(dict(value=value, roles=roles))
        issue(f'{key}_crosses_roles', crossings)
    issue('audio_has_conflicting_content_hashes', [audio for audio, indices in identities['audio'].items()
          if len({_text(rows[i].get('recording')) for i in indices}) > 1])

    linked = _linked_units(rows)
    units = defaultdict(list)
    for i, unit in enumerate(linked):
        units[unit].append(i)
    issue('linked_unit_crosses_roles', [dict(unit=unit,
          roles=sorted({_text(rows[i].get('split')) for i in indices}))
          for unit, indices in units.items()
          if len({_text(rows[i].get('split')) for i in indices}) > 1])

    times = _recording_times(rows)
    missing_time = {key for key, info in times.items() if not info['usable']}
    issue('contradictory_time_metadata', [key for key, info in times.items()
          if info['inconsistent_flags'] or (info['declared_available'] and not info['usable'])])
    # Only source-role boundaries were purged by splitting.inner_partition.
    site_records, day_roles = defaultdict(dict), defaultdict(set)
    source_recordings, temporal_covered = set(), set()
    for row in rows:
        rec, role, site = _text(row.get('recording')), row.get('split'), _text(row.get('site'))
        if role not in ROLES[:3] or rec not in times:
            continue
        source_recordings.add(rec)
        info = times[rec]
        if info['usable']:
            day_roles[site, math.floor(info['start'] / (3600 * block_hours))].add(role)
            if info['duration'] is not None:
                site_records[site][rec, role] = (info['start'], info['duration'])
                temporal_covered.add(rec)
    issue('same_site_time_bin_crosses_source_roles', [dict(site=site, time_bin=day,
          roles=sorted(roles)) for (site, day), roles in day_roles.items() if len(roles) > 1])
    close_pairs = []
    pair_count = 0
    for site, members in site_records.items():
        active = []
        for start, duration, recording, role in sorted(
                (value[0], value[1], key[0], key[1]) for key, value in members.items()):
            active = [v for v in active if v[0] + temporal_gap_seconds >= start]
            for end, other_recording, other_role in active:
                if recording != other_recording and role != other_role:
                    pair_count += 1
                    if len(close_pairs) < 20:
                        close_pairs.append(dict(site=site, recordings=[other_recording, recording],
                                                roles=[other_role, role], gap_seconds=start-end))
            active.append((start + duration, recording, role))
    if pair_count:
        errors.append(dict(code='temporal_buffer_crosses_source_roles', count=pair_count,
                           examples=close_pairs))

    prediction_check = _prediction_audit(rows, prediction_metadata)
    errors.extend(prediction_check['errors'])
    collections = sorted({_text(row.get('group')) for row in rows})
    held_collections = sorted({_text(row.get('group')) for row in rows if row.get('split') == 'holdout'})
    cells = defaultdict(list)
    for i, row in enumerate(rows):
        cells[_text(row.get('group')), _text(row.get('split')), _text(row.get('raw_class'))].append(i)
    flow = []
    for collection in collections:
        for role in ROLES:
            for label in CLASSES:
                indices = cells[collection, role, label]
                recs = {_text(rows[i].get('recording')) for i in indices if _present(rows[i].get('recording'))}
                window_ids = {_text(rows[i].get('id')) for i in indices if _present(rows[i].get('id'))}
                flow.append(dict(fold=fold, collection=collection, role=role, raw_class=label,
                                 status='observed' if indices else 'no_retained_windows',
                                 rows=len(indices), unique_windows=len(window_ids),
                                 recordings=len(recs), linked_units=len({linked[i] for i in indices}),
                                 blocks=len({_text(rows[i].get('block')) for i in indices if _present(rows[i].get('block'))}),
                                 sites=len({_text(rows[i].get('site')) for i in indices if _present(rows[i].get('site'))}),
                                 missing_timestamp_recordings=len(recs & missing_time),
                                 missing_timestamp_windows=len({_text(rows[i].get('id')) for i in indices
                                     if _text(rows[i].get('recording')) in missing_time}),
                                 duplicate_id_windows=len(window_ids & set(duplicates)),
                                 conflicting_label_windows=len(window_ids & set(conflicting))))
    limitation = bool(missing_time or source_recordings - temporal_covered)
    status = ('not_assessable' if not rows else 'invalid' if errors
              else 'audited_with_limitations' if limitation else 'audited')
    providers = defaultdict(set)
    for row in rows:
        values = row.get('provider_links') or [row.get('provider')]
        for value in values:
            if _present(value):
                providers[_text(row.get('group'))].add(str(value))
    summary = dict(
        schema='orcatca.provenance.v1', fold=fold, status=status,
        protocol_valid=None if not rows else not errors, rows=len(rows), unique_windows=len(ids),
        recordings=len(identities['recording']), linked_units=len(units),
        duplicate_window_ids=len(duplicates), duplicate_excess_rows=sum(len(ids[k])-1 for k in duplicates),
        conflicting_window_label_ids=len(conflicting), duplicate_content_windows=len(duplicate_content),
        missing_timestamp_recordings=len(missing_time),
        missing_timestamp_recording_examples=sorted(missing_time)[:20],
        required_fields_missing={k: n for k, n in missing.items() if n},
        collections=collections, heldout_collections=held_collections,
        providers_by_collection={k: sorted(v) for k, v in sorted(providers.items())},
        errors=errors, prediction_metadata=prediction_check,
        temporal_check=dict(scope='source train/select/calibrate roles only',
                            gap_seconds=temporal_gap_seconds, block_hours=block_hours,
                            source_recordings=len(source_recordings),
                            recordings_with_usable_start_and_duration=len(temporal_covered),
                            uncovered_recordings=len(source_recordings-temporal_covered),
                            observed_cross_role_pairs=pair_count),
        original_archive_exclusion_flow='not_reconstructed_from_retained_manifest',
        original_data_modified=False, scores_examined=False, independent_encounters_verified=False,
        interpretation=('Observed contributor components and retained-window counts only. '
                        'Recording, block, and site counts are distinct within each cell and '
                        'are not additive across labels. Missing times leave temporal proximity '
                        'unassessed; no observed identifier crossing does not establish encounter independence.'))
    return flow, summary


def minimal_manifest(rows, fold):
    """Project retained metadata without probabilities, audio samples, or weights.

    Audit first; this projection does not repair or silently deduplicate rows.
    Audio values, when present, are archive-relative path identifiers only.
    """
    rows = list(rows)
    linked = _linked_units(rows)
    return [dict({key: row[key] for key in _MANIFEST_FIELDS if key in row},
                 fold=_fold_name(fold), linked_unit=linked[i])
            for i, row in enumerate(rows)]
