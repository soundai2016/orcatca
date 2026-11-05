#!/usr/bin/env python3
"""Check that the standard figures match their recorded inputs and current code."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re


LAYOUT = 'kw-hw-ab-spectrum'
CODE_FILES = ('scripts/plot.py', 'scripts/acoustic_panels.py',
              'scripts/acoustic_examples.py', 'scripts/check_label_mixtures.py',
              'scripts/check_figures.py')
FIGURES = ('Fig1.pdf', 'Fig1.png', 'Fig2.pdf', 'Fig2.png')
ANALYSIS = 'result/analysis/transfer_review'
INPUT_FILES = ('analysis.json', 'transfer_gap_summary.csv',
               'transfer_original_labels.csv', 'transfer_original_summary.csv',
               'main_results.csv', 'support_sensitivity.csv', 'joint_differences.csv',
               'site_metrics.csv', 'transfer_label_summary.csv',
               'crossfit_operating.csv', 'crossfit_transfer.csv')
ASSETS = ('examples.json', 'features.npz')
LABELS = ('KW', 'HW', 'AB')
EVIDENCE = 'result/figure_data/derived/available_evidence.json'
ROOT = Path(__file__).resolve().parents[1]
SCOPE = ('Checks local render provenance and file consistency. Window predictions, '
         'bootstrap draws and model training are not replayed; raw audio is not reread.')


def inspect_outputs(root, code_root=None):
    """Inspect named files without changing them; return a JSON-compatible report."""
    errors = []
    checked = set()
    report = dict(status='failed', errors=errors, checked_files=0, layout=None,
                  source_recording=None, noncanonical_outputs=[], outputs={}, scope=SCOPE)

    def error(message):
        if message not in errors:
            errors.append(message)

    try:
        root = Path(root).expanduser().resolve()
        code_root = ROOT if code_root is None else Path(code_root).expanduser().resolve()
    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        error('Cannot resolve the selected root: '+str(exc))
        return report

    result = root/'result'
    report['outputs'] = {name: str(result/name) for name in FIGURES}
    report['noncanonical_outputs'] = [str(result/'figures'/name) for name in FIGURES
        if (result/'figures'/name).is_file() or (result/'figures'/name).is_symlink()]

    def readable(path, boundary=None):
        try:
            if boundary is not None and not path.resolve().is_relative_to(boundary):
                error('File resolves outside the selected root: '+str(path))
                return False
            if not path.is_file():
                error('Missing file: '+str(path))
                return False
            return True
        except (OSError, RuntimeError, ValueError) as exc:
            error('Cannot access '+str(path)+': '+str(exc))
            return False

    def invalid_constant(value):
        raise ValueError('Non-finite JSON constant: '+value)

    def read_json(path, boundary=None):
        if not readable(path, boundary):
            return None
        try:
            value = json.loads(path.read_text(encoding='utf-8'), parse_constant=invalid_constant)
        except (OSError, UnicodeError, ValueError) as exc:
            error('Invalid JSON in '+str(path)+': '+str(exc))
            return None
        if not isinstance(value, dict):
            error('Expected a JSON object in '+str(path))
            return None
        return value

    def mapping(parent, key):
        value = parent.get(key)
        if not isinstance(value, dict):
            error('Missing or invalid object: '+key)
            return {}
        return value

    def verify(path, expected, boundary=None):
        if not readable(path, boundary):
            return None
        if not isinstance(expected, str) or re.fullmatch(r'[0-9a-f]{64}', expected) is None:
            error('Missing or invalid SHA-256 for '+str(path))
            return None
        try:
            with path.open('rb') as stream:
                actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        except OSError as exc:
            error('Cannot read '+str(path)+': '+str(exc))
            return None
        if actual != expected:
            error('SHA-256 mismatch: '+str(path))
        else:
            checked.add(str(path.resolve()))
        return actual

    def directory(value, label, allow_absolute=False):
        if (not isinstance(value, str) or not value or '\x00' in value or '\\' in value or
                '..' in PurePosixPath(value).parts):
            error('Unsafe or missing '+label)
            return None
        path = Path(value)
        if path.is_absolute() and not allow_absolute:
            error('Absolute path is not allowed for '+label)
            return None
        try:
            selected = path.resolve() if path.is_absolute() else (root/path).resolve()
        except (OSError, RuntimeError, ValueError) as exc:
            error('Cannot resolve '+label+': '+str(exc))
            return None
        if not path.is_absolute() and not selected.is_relative_to(root):
            error(label+' resolves outside the selected root')
            return None
        return selected

    for name in FIGURES:
        readable(result/name, root)
    evidence = read_json(root/EVIDENCE, root)
    if evidence is None:
        return report
    if evidence.get('mode') != 'completed_analysis':
        error('Expected mode=completed_analysis')
    if evidence.get('input_directory') != ANALYSIS:
        error('Expected input_directory='+ANALYSIS)
    if evidence.get('alpha') != .01:
        error('Expected the 1% source rule (alpha=0.01)')

    renderer = mapping(evidence, 'renderer')
    report['layout'] = renderer.get('layout')
    if report['layout'] != LAYOUT:
        error('Expected renderer layout='+LAYOUT+'; regenerate the standard pair')
    output = directory(renderer.get('output_directory'), 'renderer.output_directory', True)
    if output is not None and output != result.resolve():
        error('The recorded output directory is not the canonical result directory')
    code_hashes = mapping(renderer, 'code_sha256')
    if set(code_hashes) != set(CODE_FILES):
        error('renderer.code_sha256 must name exactly the five current rendering scripts')
    for name in CODE_FILES:
        verify(code_root/name, code_hashes.get(name), code_root)

    hashes = mapping(evidence, 'figure_sha256')
    if set(hashes) != set(FIGURES):
        error('figure_sha256 must name exactly Fig1.pdf, Fig1.png, Fig2.pdf and Fig2.png')
    for name in FIGURES:
        verify(result/name, hashes.get(name), root)

    hashes = mapping(evidence, 'input_sha256')
    missing = sorted(set(INPUT_FILES)-set(hashes))
    if missing:
        error('Missing required input hashes: '+', '.join(missing))
    for name, digest in hashes.items():
        path = PurePosixPath(name)
        if (not name or '\x00' in name or '\\' in name or len(path.parts) != 1 or
                path.is_absolute() or path.name in ('.', '..') or path.as_posix() != name):
            error('Unsafe input filename in input_sha256: '+repr(name))
            continue
        verify(root/ANALYSIS/name, digest, root)

    context = mapping(evidence, 'acoustic_context')
    if context.get('mode') != 'measured':
        error('The standard pair requires measured acoustic context')
    if context.get('labels') != list(LABELS):
        error('Expected acoustic labels in order KW, HW, AB')
    asset_dir = directory(context.get('asset_directory'), 'acoustic_context.asset_directory', True)
    asset_hashes = mapping(context, 'asset_sha256')
    if set(asset_hashes) != set(ASSETS):
        error('asset_sha256 must name exactly examples.json and features.npz')
    if asset_dir is not None:
        actual = {name: verify(asset_dir/name, asset_hashes.get(name), asset_dir) for name in ASSETS}
        metadata = read_json(asset_dir/'examples.json', asset_dir)
        if metadata is not None:
            for key in ('recording', 'site'):
                value = metadata.get(key)
                if not isinstance(value, str) or not value or context.get(key) != value:
                    error('Acoustic '+key+' does not match examples.json')
            report['source_recording'] = metadata.get('recording')
            labels = mapping(metadata, 'labels')
            window_ids = mapping(context, 'window_ids')
            if set(labels) != set(LABELS) or set(window_ids) != set(LABELS):
                error('Acoustic window metadata must contain exactly KW, HW and AB')
            for label in LABELS:
                item = labels.get(label)
                ident = item.get('id') if isinstance(item, dict) else None
                if not isinstance(ident, str) or not ident or window_ids.get(label) != ident:
                    error('Acoustic window ID does not match examples.json: '+label)
            feature_hash = metadata.get('features_sha256')
            if (not isinstance(feature_hash, str) or actual['features.npz'] is None or
                    feature_hash != actual['features.npz']):
                error('examples.json features_sha256 does not match features.npz')
    report['checked_files'] = len(checked)
    report['status'] = 'failed' if errors else 'passed'
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT, help='Existing project directory')
    args = parser.parse_args(argv)
    result = inspect_outputs(args.root)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
