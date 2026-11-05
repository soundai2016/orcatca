"""Validated experiment configuration."""
from __future__ import annotations
import math
import re
from pathlib import Path
import yaml
METHODS = {'natural_mixstyle': dict(balance=False, transform='mixstyle', subtype=False, consistency=False), 'balanced_erm': dict(balance=True, transform='none', subtype=False, consistency=False), 'balanced_filter': dict(balance=True, transform='filter', subtype=False, consistency=False), 'balanced_mixstyle': dict(balance=True, transform='mixstyle', subtype=False, consistency=False), 'subtype_mixstyle': dict(balance=True, transform='mixstyle', subtype=True, consistency=False), 'subtype_shift': dict(balance=True, transform='shift', subtype=True, consistency=False), 'subtype_shift_js': dict(balance=True, transform='shift', subtype=True, consistency=True), 'subtype_random_js': dict(balance=True, transform='random_matched', subtype=True, consistency=True)}

METHODS.update({
    'natural_subtype_mixstyle': dict(balance=False, transform='mixstyle', subtype=True, consistency=False),
    'natural_allneg_mixstyle': dict(balance=False, transform='mixstyle', subtype=True, consistency=False, tail='all'),
    'natural_randomtail_mixstyle': dict(balance=False, transform='mixstyle', subtype=True, consistency=False, tail='random'),
    'natural_tail_mixstyle': dict(balance=False, transform='mixstyle', subtype=False, consistency=False, tail='hard'),
    'natural_subtype_tail_mixstyle': dict(balance=False, transform='mixstyle', subtype=True, consistency=False, tail='hard'),
})

METHODS.update({
    'natural_coarse_mixstyle': dict(balance=False, transform='mixstyle', subtype=True,
                                    consistency=False, objective='coarse_aux'),
    'natural_threeclass_mixstyle': dict(balance=False, transform='mixstyle', subtype=False,
                                        consistency=False, objective='direct3'),
})

def load(path):
    config = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    return validate(config)

def validate(c):
    required = {'name', 'runtime', 'data', 'cache', 'experiment', 'augmentation', 'evaluation'}
    if not isinstance(c, dict) or not required <= set(c) or set(c) - required - {'input_reference'}:
        raise ValueError(f'Expected configuration sections: {sorted(required)}')
    if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,63}', str(c['name'])):
        raise ValueError('Unsafe experiment name')
    for section in c.values():
        if isinstance(section, dict):
            for key, value in section.items():
                if isinstance(value, float) and (not math.isfinite(value)):
                    raise ValueError(f'Nonfinite configuration value: {key}')
    r, d, e, a, k, v = [c[s] for s in ('runtime', 'data', 'experiment', 'augmentation', 'cache', 'evaluation')]
    if d.get('decode_all_candidates') is not True:
        raise ValueError('Full candidate decoding is required')
    if r['device'] not in ('npu', 'cpu') or r['precision'] not in ('fp32', 'bf16'):
        raise ValueError('Unsupported device or precision')
    if r['device'] == 'npu':
        from .ascend_policy import communication_policy
        communication_policy(r, {})
    for name in ('allow_internal_format', 'jit_compile'):
        if name in r and type(r[name]) is not bool:
            raise ValueError(f'{name} must be boolean')
    if r['device'] == 'cpu' and r['precision'] != 'fp32':
        raise ValueError('CPU validation requires FP32')
    for section, names in ((r, ['nproc', 'threads', 'timeout_s']), (d, ['sample_rate', 'n_fft', 'hop', 'index_workers', 'decode_workers', 'folds', 'min_per_class', 'min_recordings_per_class', 'min_train_subtype_recordings']), (e, ['folds', 'epochs', 'patience', 'batch_size', 'global_batch', 'width', 'log_every']), (k, ['shard_size'])):
        for key in names:
            if type(section[key]) is not int or section[key] < 1:
                raise ValueError(f'Expected positive integer: {key}')
    if type(r['workers']) is not int or r['workers'] < 0:
        raise ValueError('workers must be nonnegative')
    if not 0 <= d['min_hz'] < d['max_hz'] <= d['sample_rate'] / 2:
        raise ValueError('Invalid frequency range')
    if not 1 <= d['hop'] <= d['n_fft'] <= round(d['seconds'] * d['sample_rate']):
        raise ValueError('Invalid STFT geometry')
    if d['session_gap_seconds'] < 0 or d['block_hours'] < 1:
        raise ValueError('Invalid temporal block settings')
    if not 0 < d['select_fraction'] < 0.3 or not 0 < d['calibrate_fraction'] < 0.3 or d['select_fraction'] + d['calibrate_fraction'] >= 0.5:
        raise ValueError('Selection/calibration fractions must preserve a majority training pool')
    if e['folds'] > d['folds'] or not e['methods'] or len(set(e['methods'])) != len(e['methods']) or set(e['methods']) - METHODS.keys():
        raise ValueError('Invalid experiment folds or methods')
    if not e['seeds'] or len(set(e['seeds'])) != len(e['seeds']) or any((type(s) is not int or s < 0 for s in e['seeds'])):
        raise ValueError('Invalid seeds')
    if e['global_batch'] % (r['nproc'] * e['batch_size']):
        raise ValueError('global_batch must be divisible by nproc * batch_size')
    if e['width'] < 4 or not 0 <= e['dropout'] < 1 or (not 0 <= e['label_smoothing'] < 1):
        raise ValueError('Invalid model regularization')
    if e['learning_rate'] <= 0 or e['weight_decay'] < 0 or (not 0 < e['ema_decay'] < 1):
        raise ValueError('Invalid optimizer settings')
    if not 0 <= e['warmup_epochs'] < e['epochs'] or e['consistency_warmup'] < 0:
        raise ValueError('Invalid warmup')
    if not 0 <= e['balance_strength'] <= 1 or e['max_weight'] < 1 or min(e['subtype_weight'], e['consistency_weight']) < 0:
        raise ValueError('Invalid loss weights')
    if e['selection_metric'] not in ('macro_auc', 'macro_pauc') or not 0 < e['selection_max_fpr'] <= 1:
        raise ValueError('Invalid primary endpoint')
    if not 0 <= a['probability'] <= 1 or a['filter_db'] < 0 or a['knots'] < 2 or (a['mixstyle_alpha'] <= 0):
        raise ValueError('Invalid augmentation settings')
    if k['dtype'] not in ('float16', 'float32') or k['budget_gib'] <= 0 or k['reserve_gib'] < 0:
        raise ValueError('Invalid cache policy')
    if not v['fpr_targets'] or any((not 0 < x < 1 for x in v['fpr_targets'])):
        raise ValueError('Invalid operating points')
    if type(v['bootstrap_repeats']) is not int or v['bootstrap_repeats'] < 10:
        raise ValueError('At least ten bootstrap replicates are required')
    if e.get('tail_weight', 0) < 0 or not 0 < e.get('tail_fraction', .25) <= 1:
        raise ValueError('Invalid negative-tail settings')
    if e.get('tail_warmup_epochs', 0) < 0 or e.get('tail_min_recordings', 2) < 2:
        raise ValueError('Tail mining needs at least two recordings and nonnegative warmup')
    for key in ('min_eval_subtype_blocks', 'min_eval_subtype_recordings'):
        if type(d.get(key, 2)) is not int or d.get(key, 2) < 1:
            raise ValueError('Invalid source subtype support')
    if d.get('repair_subtype_coverage', True) not in (True, False):
        raise ValueError('repair_subtype_coverage must be boolean')
    if set(v.get('required_negative_subtypes', ['AB', 'HW'])) != {'AB', 'HW'}:
        raise ValueError('Operating support must include AB and HW')
    for key in ('min_negative_recordings', 'min_negative_blocks'):
        if type(v.get(key, 2)) is not int or v.get(key, 2) < 1:
            raise ValueError('Invalid negative calibration support')
    if 'input_reference' in c:
        ref = c['input_reference']
        fields = {'source_suite', 'source_config', 'source_config_sha256', 'protocol_complete', 'protocol_sha256', 'reuse_weights'}
        if not isinstance(ref, dict) or not fields <= ref.keys() or ref.get('reuse_weights') is not False:
            raise ValueError('Invalid read-only input binding')
        for key in ('source_config_sha256', 'protocol_sha256'):
            if not re.fullmatch('[0-9a-f]{64}', ref[key]):
                raise ValueError('Invalid input digest')
    return c
