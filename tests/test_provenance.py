"""Audits must expose support and identity failures without changing inputs."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from orcatca.provenance import audit_protocol, minimal_manifest
from orcatca.results import pack


def row(sid, recording, block, label='AB', role='train', collection='DFO',
        start=0, day=100, **extra):
    result = dict(id=sid, recording=recording, audio=f'{recording}.wav', block=block,
                  start_frame=start, raw_class=label, label=int(label == 'KW'), split=role,
                  group=collection, provider=collection, site=collection+'::site',
                  block_utc=day*86400, recording_utc=day*86400,
                  time_metadata_available=True, recording_duration=10.0)
    result.update(extra)
    return result


class ProvenanceTests(unittest.TestCase):
    def test_report_completion_record_matches_selected_protocol(self):
        for frozen in (False, True):
            with self.subTest(frozen=frozen), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                suite = root / 'runs/historical'
                suite.mkdir(parents=True)
                records = {}
                for relative, identity in (('state/protocol', 'current'),
                                           ('state/frozen_protocol', 'historical')):
                    directory = root / relative
                    directory.mkdir(parents=True)
                    manifest = (identity + ' manifest').encode()
                    (directory / 'fold00.jsonl.gz').write_bytes(manifest)
                    complete = json.dumps({'identity': identity, 'files': {
                        'fold00.jsonl.gz': hashlib.sha256(manifest).hexdigest()
                    }}).encode()
                    (directory / 'complete.json').write_bytes(complete)
                    records[identity] = (manifest, complete)
                cfg = {}
                if frozen:
                    cfg['input_reference'] = {
                        'protocol_complete': 'state/frozen_protocol/complete.json',
                        'protocol_sha256': hashlib.sha256(records['historical'][1]).hexdigest()
                    }
                (suite / 'config.yaml').write_text(json.dumps(cfg), encoding='utf-8')
                (suite / 'best.pt').write_bytes(b'model excluded from reports')
                (suite / 'recording.wav').write_bytes(b'audio excluded from reports')
                before = {p: p.read_bytes() for p in (root / 'state').rglob('*') if p.is_file()}

                path = pack(root, 'historical')
                expected_manifest, expected_complete = records['historical' if frozen else 'current']
                with zipfile.ZipFile(path) as archive:
                    names = archive.namelist()
                    self.assertEqual(len(names), len(set(names)))
                    self.assertEqual(archive.read('state/protocol/complete.json'), expected_complete)
                    self.assertEqual(archive.read('state/protocol/fold00.jsonl.gz'), expected_manifest)
                    checksums = json.loads(archive.read('REPORT_SHA256.json'))
                    for name in ('state/protocol/complete.json', 'state/protocol/fold00.jsonl.gz'):
                        self.assertEqual(checksums[name], hashlib.sha256(archive.read(name)).hexdigest())
                    self.assertFalse(any(name.endswith(('.pt', '.wav')) for name in names))
                for path, contents in before.items():
                    self.assertEqual(path.read_bytes(), contents)

    def test_changed_frozen_protocol_preserves_existing_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suite = root / 'runs/historical'
            protocol = root / 'state/frozen_protocol'
            suite.mkdir(parents=True)
            protocol.mkdir(parents=True)
            original = b'{"identity": "original"}'
            (protocol / 'complete.json').write_bytes(b'{"identity": "changed"}')
            (suite / 'config.yaml').write_text(json.dumps({'input_reference': {
                'protocol_complete': 'state/frozen_protocol/complete.json',
                'protocol_sha256': hashlib.sha256(original).hexdigest()
            }}), encoding='utf-8')
            report = root / 'state/historical_report.zip'
            report.write_bytes(b'existing report must survive failed validation')
            before = report.read_bytes()
            with self.assertRaisesRegex(ValueError, 'Referenced protocol changed'):
                pack(root, 'historical')
            self.assertEqual(report.read_bytes(), before)

    def test_transitive_bridge_is_linked_before_label_counts(self):
        rows = [row('a', 'r1', 'b1', 'HW'),
                row('b', 'r2', 'b1', 'KW'),
                row('c', 'r2', 'b2', 'KW', start=16000),
                row('d', 'r3', 'b2', 'AB')]
        before = copy.deepcopy(rows)
        flow, summary = audit_protocol(rows, 0)
        self.assertTrue(summary['protocol_valid'])
        self.assertEqual(summary['linked_units'], 1)
        self.assertEqual(sum(r['unique_windows'] for r in flow), 4)
        self.assertEqual({r['linked_units'] for r in flow if r['rows']}, {1})
        self.assertEqual(rows, before)
        exported = minimal_manifest(rows, 0)
        self.assertEqual(len({r['linked_unit'] for r in exported}), 1)
        reverse = minimal_manifest(list(reversed(rows)), 0)
        self.assertEqual({r['linked_unit'] for r in exported}, {r['linked_unit'] for r in reverse})

    def test_bridge_crossing_roles_is_not_independent(self):
        rows = [row('a', 'r1', 'b1'), row('b', 'r2', 'b1'),
                row('c', 'r2', 'b2', start=16000),
                row('d', 'r3', 'b2', role='calibrate')]
        _, summary = audit_protocol(rows, 'fold00')
        self.assertEqual(summary['status'], 'invalid')
        codes = {e['code'] for e in summary['errors']}
        self.assertIn('linked_unit_crosses_roles', codes)
        self.assertIn('block_crosses_roles', codes)

    def test_missing_time_is_not_reported_as_temporal_safety(self):
        rows = [row('a', 'r1', 'b1', time_metadata_available=False,
                    recording_utc=None, block_utc=None)]
        flow, summary = audit_protocol(rows, 0)
        self.assertEqual(summary['status'], 'audited_with_limitations')
        self.assertEqual(summary['missing_timestamp_recordings'], 1)
        self.assertEqual(summary['temporal_check']['uncovered_recordings'], 1)
        self.assertFalse(summary['independent_encounters_verified'])
        self.assertEqual(sum(r['missing_timestamp_windows'] for r in flow), 1)

    def test_same_recording_different_labels_is_allowed_but_same_id_is_not(self):
        rows = [row('a', 'r1', 'b1', 'HW'), row('b', 'r1', 'b1', 'AB', start=80000)]
        flow, summary = audit_protocol(rows, 0)
        self.assertTrue(summary['protocol_valid'])
        self.assertEqual(summary['recordings'], 1)
        self.assertEqual(sum(r['recordings'] for r in flow), 2)
        rows[1]['id'] = 'a'
        _, summary = audit_protocol(rows, 0)
        self.assertFalse(summary['protocol_valid'])
        self.assertEqual(summary['duplicate_window_ids'], 1)
        self.assertEqual(summary['conflicting_window_label_ids'], 1)

    def test_identical_duplicate_and_content_alias_are_invalid(self):
        a = row('a', 'r1', 'b1')
        _, duplicate = audit_protocol([a, dict(a)], 0)
        self.assertEqual(duplicate['duplicate_excess_rows'], 1)
        self.assertEqual(duplicate['status'], 'invalid')
        alias = dict(a, id='alias', audio='alias.wav', raw_class='HW')
        _, content = audit_protocol([a, alias], 0)
        self.assertEqual(content['duplicate_content_windows'], 1)
        self.assertEqual(content['status'], 'invalid')

    def test_extra_training_provider_is_reported_without_invalidating(self):
        rows = [row('a', 'r1', 'b1', collection='OTHER_PROVIDER'),
                row('b', 'r2', 'b2', collection='SIO', role='holdout', label='KW')]
        flow, summary = audit_protocol(rows, 2)
        self.assertTrue(summary['protocol_valid'])
        self.assertEqual(summary['heldout_collections'], ['SIO'])
        self.assertIn('OTHER_PROVIDER', summary['collections'])
        absent = [r for r in flow if r['collection'] == 'SIO' and r['raw_class'] == 'HW']
        self.assertTrue(all(r['unique_windows'] == 0 for r in absent))

    def test_prediction_ids_and_labels_are_compared_to_exact_roles(self):
        a = row('a', 'r1', 'b1', role='calibrate')
        b = row('b', 'r2', 'b2', role='holdout', collection='SIO', label='KW')
        _, good = audit_protocol([a, b], 0, {'calibration': [dict(a, p=.9)], 'holdout': [b]})
        self.assertEqual(good['prediction_metadata']['status'], 'aligned')
        bad = dict(a, raw_class='HW', label=1)
        _, bad_report = audit_protocol([a, b], 0, {'calibrate': [bad, bad], 'holdout': []})
        codes = {e['code'] for e in bad_report['errors']}
        self.assertIn('duplicate_prediction_ids', codes)
        self.assertIn('prediction_metadata_mismatch', codes)
        self.assertIn('missing_prediction_ids', codes)

    def test_temporal_buffer_checks_across_day_boundaries(self):
        a = row('a', 'r1', 'b1', block_utc=86400-10, recording_utc=86400-10,
                recording_duration=5)
        b = row('b', 'r2', 'b2', role='calibrate', block_utc=86400+100,
                recording_utc=86400+100)
        _, summary = audit_protocol([a, b], 0, temporal_gap_seconds=1800)
        self.assertIn('temporal_buffer_crosses_source_roles', {e['code'] for e in summary['errors']})
        self.assertEqual(summary['temporal_check']['observed_cross_role_pairs'], 1)

    def test_empty_input_is_not_a_successful_audit(self):
        flow, summary = audit_protocol([], 0)
        self.assertEqual(flow, [])
        self.assertEqual(summary['status'], 'not_assessable')
        self.assertIsNone(summary['protocol_valid'])

    def test_missing_window_location_prevents_content_identity_audit(self):
        a = row('a', 'r1', 'b1')
        del a['start_frame']
        _, summary = audit_protocol([a], 0, {})
        self.assertFalse(summary['protocol_valid'])
        self.assertEqual(summary['prediction_metadata']['status'], 'not_provided')
        self.assertIn('invalid_window_start_frame', {e['code'] for e in summary['errors']})

    def test_invalid_negative_class_and_unknown_label_are_flagged(self):
        a = row('a', 'r1', 'b1', label='HW')
        a['label'] = 1
        b = row('b', 'r2', 'b2', label='UNDBIO')
        _, summary = audit_protocol([a, b], 0)
        issue = next(e for e in summary['errors'] if e['code'] == 'inconsistent_label')
        self.assertEqual(issue['count'], 2)

    def test_minimal_manifest_excludes_scores_without_synthesizing_times(self):
        a = row('a', 'r1', 'b1', p=.99)
        del a['recording_utc']
        exported = minimal_manifest([a], 0)[0]
        self.assertNotIn('p', exported)
        self.assertNotIn('recording_utc', exported)
        self.assertEqual(exported['fold'], 'fold00')
        self.assertEqual(exported['audio'], 'r1.wav')


if __name__ == '__main__':
    unittest.main()
