"""An old output directory or changed input must not validate as the current pair."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('check_figures', ROOT/'scripts/check_figures.py')
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FigureOutputTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for name in (*check.CODE_FILES, *(check.ANALYSIS+'/'+n for n in check.INPUT_FILES)):
            destination = self.root/name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/name, destination)
        shutil.copytree(ROOT/'assets/acoustic_examples', self.root/'assets/acoustic_examples')
        for name in check.FIGURES:
            (self.root/'result'/name).write_bytes(('rendered '+name+'\n').encode())
        assets = self.root/'assets/acoustic_examples'
        metadata = json.loads((assets/'examples.json').read_text())
        self.evidence = dict(mode='completed_analysis', input_directory=check.ANALYSIS, alpha=.01,
            input_sha256={name: sha(self.root/check.ANALYSIS/name) for name in check.INPUT_FILES},
            renderer=dict(layout=check.LAYOUT, output_directory='result',
                code_sha256={name: sha(self.root/name) for name in check.CODE_FILES},
                environment={'matplotlib':'an-independent-runtime'}),
            figure_sha256={name: sha(self.root/'result'/name) for name in check.FIGURES},
            acoustic_context=dict(mode='measured', asset_directory='assets/acoustic_examples',
                asset_sha256={name: sha(assets/name) for name in check.ASSETS},
                recording=metadata['recording'], site=metadata['site'], labels=list(check.LABELS),
                window_ids={label: metadata['labels'][label]['id'] for label in check.LABELS}))
        self.save_evidence()

    def save_evidence(self):
        path = self.root/check.EVIDENCE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.evidence), encoding='utf-8')

    def inspect(self):
        return check.inspect_outputs(self.root, code_root=self.root)

    def assert_failed(self, text):
        report = self.inspect()
        self.assertEqual(report['status'], 'failed')
        self.assertIn(text, '\n'.join(report['errors']))
        return report

    def test_current_pair_passes_without_changing_files(self):
        before = {str(p): (sha(p), p.stat().st_mtime_ns)
                  for p in self.root.rglob('*') if p.is_file()}
        report = self.inspect()
        self.assertEqual(report['status'], 'passed', report['errors'])
        self.assertEqual(report['checked_files'], 22)
        self.assertEqual(report['layout'], check.LAYOUT)
        self.assertEqual(report['source_recording'], self.evidence['acoustic_context']['recording'])
        self.assertTrue(all(Path(p).is_absolute() for p in report['outputs'].values()))
        self.assertEqual(before, {str(p): (sha(p), p.stat().st_mtime_ns)
                                 for p in self.root.rglob('*') if p.is_file()})

    def test_nested_figures_never_replace_missing_canonical_files(self):
        legacy = self.root/'result/figures'
        legacy.mkdir()
        for name in check.FIGURES:
            (self.root/'result'/name).rename(legacy/name)
        before = {p.name: (sha(p), p.stat().st_mtime_ns) for p in legacy.iterdir()}
        report = self.assert_failed('Missing file: ')
        self.assertEqual(report['noncanonical_outputs'], [str(legacy/n) for n in check.FIGURES])
        self.assertEqual(before, {p.name: (sha(p), p.stat().st_mtime_ns) for p in legacy.iterdir()})
        self.assertFalse(any((self.root/'result'/n).exists() for n in check.FIGURES))

    def test_old_nested_copy_is_reported_but_does_not_fail_current_outputs(self):
        old = self.root/'result/figures/Fig1.pdf'
        old.parent.mkdir()
        old.write_bytes(b'older output')
        report = self.inspect()
        self.assertEqual(report['status'], 'passed', report['errors'])
        self.assertEqual(report['noncanonical_outputs'], [str(old)])
        self.assertEqual(old.read_bytes(), b'older output')

    def test_replaced_figure_fails(self):
        (self.root/'result/Fig1.pdf').write_bytes(b'different output')
        self.assert_failed('SHA-256 mismatch: '+str(self.root/'result/Fig1.pdf'))

    def test_changed_rendering_code_fails(self):
        path = self.root/'scripts/acoustic_panels.py'
        path.write_bytes(path.read_bytes()+b'\n# changed\n')
        self.assert_failed('SHA-256 mismatch: '+str(path))

    def test_changed_analysis_input_fails(self):
        path = self.root/check.ANALYSIS/'transfer_gap_summary.csv'
        path.write_bytes(path.read_bytes()+b'\n')
        self.assert_failed('SHA-256 mismatch: '+str(path))

    def test_changed_acoustic_features_fail_even_if_the_asset_hash_is_refreshed(self):
        path = self.root/'assets/acoustic_examples/features.npz'
        path.write_bytes(path.read_bytes()+b'changed')
        self.assert_failed('SHA-256 mismatch: '+str(path))
        self.evidence['acoustic_context']['asset_sha256']['features.npz'] = sha(path)
        self.save_evidence()
        self.assert_failed('examples.json features_sha256 does not match features.npz')

    def test_context_identifies_the_same_recording_site_and_windows(self):
        for key, value, text in [('recording', 'different.flac', 'Acoustic recording'),
                                 ('site', 'different-site', 'Acoustic site'),
                                 ('window_ids', {'KW':'old','HW':'old','AB':'old'}, 'Acoustic window ID')]:
            with self.subTest(key=key):
                previous = self.evidence['acoustic_context'][key]
                self.evidence['acoustic_context'][key] = value
                self.save_evidence()
                self.assert_failed(text)
                self.evidence['acoustic_context'][key] = previous

    def test_absolute_asset_directory_is_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            external = Path(tmp)/'selected'
            shutil.copytree(self.root/'assets/acoustic_examples', external)
            self.evidence['acoustic_context']['asset_directory'] = str(external)
            self.evidence['renderer']['output_directory'] = str(self.root/'result')
            self.save_evidence()
            report = self.inspect()
            self.assertEqual(report['status'], 'passed', report['errors'])

    def test_paths_in_metadata_cannot_traverse_to_unlisted_locations(self):
        self.evidence['input_sha256']['../outside.csv'] = '0'*64
        self.evidence['acoustic_context']['asset_directory'] = '../outside'
        self.evidence['renderer']['output_directory'] = '../result'
        self.save_evidence()
        report = self.assert_failed('Unsafe input filename')
        errors = '\n'.join(report['errors'])
        self.assertIn('Unsafe or missing acoustic_context.asset_directory', errors)
        self.assertIn('Unsafe or missing renderer.output_directory', errors)

    def test_required_code_and_input_hashes_cannot_be_omitted(self):
        del self.evidence['renderer']['code_sha256']['scripts/plot.py']
        del self.evidence['input_sha256']['crossfit_operating.csv']
        self.save_evidence()
        report = self.assert_failed('exactly the five current rendering scripts')
        self.assertIn('Missing required input hashes: crossfit_operating.csv', report['errors'])

    def test_invalid_json_and_missing_objects_fail_without_a_traceback(self):
        path = self.root/check.EVIDENCE
        for contents, text in [('{', 'Invalid JSON'), ('[]', 'Expected a JSON object'),
                               ('{}', 'Missing or invalid object: renderer')]:
            with self.subTest(contents=contents):
                path.write_text(contents)
                self.assert_failed(text)

    def test_other_layout_or_output_directory_does_not_validate_as_standard(self):
        self.evidence['renderer'].update(layout='statistics-only', output_directory='result/figures')
        self.save_evidence()
        report = self.assert_failed('Expected renderer layout=')
        self.assertIn('The recorded output directory is not the canonical result directory', report['errors'])

    def test_main_prints_json_and_returns_failure_for_missing_outputs(self):
        (self.root/'result/Fig1.pdf').unlink()
        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            code = check.main(['--root', str(self.root)])
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(stdout.getvalue())['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
