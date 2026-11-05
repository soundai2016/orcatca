"""Model-sized gradient and collective checks on the requested device."""
from __future__ import annotations
import argparse
import copy
import os
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from .config import load, METHODS
from .environment import capture
from .logging import configure, event, close, logger
from .model import AcousticNet
from .objective import loss_per_sample, score_probability, output_contract
from .tail import negative_tail
from .runtime import Runtime
from .transforms import transform
from .utils import project_root, output_path, atomic_json, digest

def run(config):
    cfg = load(config)
    root = project_root()
    rt = None
    rank = int(os.getenv('RANK', '0'))
    configure(output_path(root, 'state/logs'), 'smoke', rank)
    report_path = output_path(root, f'state/smoke.rank{rank}.json')
    phase = 'runtime_initialization'
    atomic_json(report_path, dict(status='running', phase=phase, rank=rank,
                                 pid=os.getpid(), config_sha256=digest(cfg)))
    try:
        rt = Runtime(cfg['runtime'])
        rt.seed(177)
        d, e = (cfg['data'], cfg['experiment'])
        hz = np.fft.rfftfreq(d['n_fft'], 1 / d['sample_rate'])
        frequencies = int(((hz >= d['min_hz']) & (hz <= d['max_hz'])).sum())
        frames = 1 + (round(d['seconds'] * d['sample_rate']) - d['n_fft']) // d['hop']
        shape = (e['batch_size'], 1, frequencies, frames)
        from .numerics import model_parity
        phase = 'model_numerical_parity'
        event('model_numerical_parity_started', shape=list(shape), width=e['width'])
        parity = model_parity(rt, shape, e['width'])
        atomic_json(output_path(root, f'state/parity.rank{rt.rank}.json'), parity)
        event('model_numerical_parity', **parity)
        if parity['status'] != 'passed':
            raise RuntimeError('FP32 CPU/backend parity failed; inspect the report before training')
        phase = 'model_optimizer_collectives'
        rt.seed(177)
        model = AcousticNet(e['width'], e['dropout']).to(rt.device)
        ema = copy.deepcopy(model).eval()
        net = DDP(model, device_ids=[rt.local] if rt.kind == 'npu' else None, broadcast_buffers=False) if rt.world > 1 else model
        opt = torch.optim.AdamW(model.parameters(), lr=e['learning_rate'])
        before = next(model.parameters()).detach().clone()
        from .train import update_ema
        checked = []
        tail_checks = []
        cases = [('natural_subtype_mixstyle', 'hard'), ('natural_subtype_mixstyle', 'random'), ('natural_subtype_mixstyle', 'all'), ('natural_coarse_mixstyle', None), ('natural_threeclass_mixstyle', None)]
        for step, (method_name, policy) in enumerate(cases):
            kind = 'mixstyle'
            opt.zero_grad(set_to_none=True)
            x = torch.randn(shape).to(rt.device)
            subtype_host = np.arange(shape[0]) % 3
            y_host = (subtype_host == 2).astype(np.int64)
            y = torch.as_tensor(y_host, device=rt.device)
            subtype = torch.as_tensor(subtype_host, device=rt.device)
            a, diag = transform(x, kind, dict(cfg['augmentation'], probability=1.0), np.arange(len(x)) % 2, np.arange(len(x)), 17, 0, step)
            with rt.autocast():
                output = net(torch.cat((x, a)))
            loss, *_ = loss_per_sample(output[:len(x)], output[len(x):], y, subtype, e, METHODS[method_name], 0)
            tail, info = negative_tail(output[:len(x)], output[len(x):], y_host, subtype_host, np.arange(len(x)), e, policy, 17, 0, step)
            loss = loss.mean() + tail
            tail_checks.append(dict(policy=policy, method=method_name, contract=output_contract(METHODS[method_name]), **info))
            probability = score_probability(output, METHODS[method_name])
            if not torch.isfinite(probability).all():
                raise FloatingPointError('Nonfinite semantic probability')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            opt.step()
            update_ema(ema, model, e['ema_decay'])
            rt.sync()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite smoke loss')
            checked.append(kind)
        if torch.equal(before, next(model.parameters()).detach()):
            raise AssertionError('No optimizer update')
        total = rt.sum([rt.rank + 1])[0]
        if total != rt.world * (rt.world + 1) / 2:
            raise AssertionError('Collective failed')
        result = dict(status='passed', numerical_parity=parity, original_shape=list(shape), forward_shape=[shape[0] * 2, *shape[1:]], transforms=checked, negative_tail_checks=tail_checks, loss=float(loss.detach()), collective_sum=total, **rt.metadata())
        atomic_json(report_path, result)
        capture(root, output_path(root, f'state/environment.rank{rt.rank}.json'), rt, digest(cfg))
        event('model_smoke_passed', **result)
    except BaseException as exc:
        failure = dict(status='failed', phase=phase, rank=rank, pid=os.getpid(),
                       exception=type(exc).__name__, message=str(exc), config_sha256=digest(cfg))
        if exc.__cause__ is not None:
            failure['cause'] = dict(exception=type(exc.__cause__).__name__, message=str(exc.__cause__))
        if rt is not None:
            failure['runtime'] = rt.metadata()
        atomic_json(report_path, failure)
        logger.exception('Device smoke failed')
        raise
    finally:
        if rt:
            rt.close()
        close()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    run(parser.parse_args().config)
