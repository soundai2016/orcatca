"""Detached single-node supervisor. PID identity, shared lock and graceful stops."""
from __future__ import annotations
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from .config import load
from .utils import project_root, output_path, process_identity, read_json, atomic_json, digest, sha256, code_digest

def fresh_name(base):
    from datetime import datetime, timezone
    from uuid import uuid4
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    return f'{base[:32]}_{stamp}_{uuid4().hex[:8]}'

def _job(root):
    p = root / 'state/job.json'
    return read_json(p) if p.exists() else None

def is_live(job):
    return bool(job and process_identity(job['pid']) == job.get('process'))

def start(config, resume=False, evaluate_holdouts=False, fresh=False):
    root = project_root()
    if fresh and resume:
        raise ValueError('--fresh and --resume are mutually exclusive')
    cfg = load(config)
    state = output_path(root, 'state')
    state.mkdir(parents=True, exist_ok=True)
    lock = (state / 'supervisor.lock').open('a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError('An OrcaTCA supervisor/worker group is already active; use status or stop')
    if fresh:
        cfg['name'] = fresh_name(cfg['name'])
        import yaml
        config = output_path(root, f"state/requests/{cfg['name']}.yaml")
        config.parent.mkdir(parents=True, exist_ok=True)
        with config.open('x') as request:
            request.write(yaml.safe_dump(cfg, sort_keys=False))
    elif not resume and (root / f"runs/{cfg['name']}/config.json").exists():
        lock.close()
        raise ValueError('Suite already exists. Use --fresh for new random initialization')
    cmd = [sys.executable, '-m', 'orcatca.control', '_worker', '--config', str(Path(config).resolve()), '--lock-fd', str(lock.fileno())]
    if resume:
        cmd.append('--resume')
    if evaluate_holdouts:
        cmd.append('--evaluate-holdouts')
    log = state / 'bootstrap.log'
    if log.exists() and log.stat().st_size > 20 * 1024 ** 2:
        import gzip
        with log.open('rb') as source, gzip.open(state / f'bootstrap_{time.time_ns()}.log.gz', 'wb') as target:
            import shutil
            shutil.copyfileobj(source, target)
        log.unlink()
    env = os.environ.copy()
    env['ORCATCA_ROOT'] = str(root)
    try:
        with log.open('ab', buffering=0) as sink, open(os.devnull, 'rb') as inp:
            child = subprocess.Popen(cmd, cwd=root, stdin=inp, stdout=sink, stderr=subprocess.STDOUT, env=env, start_new_session=True, pass_fds=(lock.fileno(),))
        ident = process_identity(child.pid)
        if ident is None:
            raise RuntimeError('Supervisor exited during startup; see state/bootstrap.log')
        atomic_json(state / 'job.json', dict(pid=child.pid, process=ident, config=str(Path(config).resolve()), suite_name=cfg['name'], fresh=fresh, command=cmd[:6], bootstrap=str(log)))
    finally:
        lock.close()
    print(f'Background supervisor started: PID={child.pid}\nSuite: {cfg["name"]}\nLog: {log}\nThis is process startup, not evidence of successful training.', flush=True)
    time.sleep(0.4)
    if child.poll() is not None and child.returncode:
        raise RuntimeError(f'Supervisor exited with {child.returncode}; inspect {log}')

def status():
    root = project_root()
    job = _job(root)
    report = dict(active=is_live(job), job=job, status=read_json(root / 'state/status.json') if (root / 'state/status.json').exists() else None)
    print(__import__('json').dumps(report, indent=2))

def stop(now=False):
    root = project_root()
    job = _job(root)
    if not is_live(job):
        print('No matching active supervisor. No signal sent.')
        return
    if job['process']['pgid'] != job['pid']:
        raise RuntimeError('Unexpected process group; refusing to signal')
    if now:
        os.killpg(job['pid'], signal.SIGTERM)
        print('SIGTERM sent only to the verified OrcaTCA process group. Resume uses the last completed epoch.')
    else:
        atomic_json(output_path(root, 'state/STOP'), dict(requested_at=time.time()))
        print('Stop requested at the next completed epoch/stage. Use stop --now for immediate termination.')

def _child(root, cmd, lock_fd):
    process = subprocess.Popen(cmd, cwd=root, pass_fds=(lock_fd,))
    return process.wait()

def worker(config, resume, lock_fd, evaluate_holdouts=False):
    from .logging import configure, event, logger, close
    root = project_root()
    cfg = load(config)
    state = output_path(root, 'state')
    configure(state / 'logs', 'supervisor')
    if lock_fd < 0:
        raise ValueError('Supervisor lock handoff missing')
    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    phase_name = 'initializing'
    try:
        stopfile = state / 'STOP'
        if stopfile.exists():
            stopfile.unlink()
        suite = output_path(root, f"runs/{cfg['name']}")
        suite.mkdir(parents=True, exist_ok=True)
        frozen = suite / 'config.json'
        identity = digest([cfg, code_digest(root)])
        if frozen.exists():
            if not resume:
                raise ValueError('Suite exists; use --resume or a new name. Historical data are not reused implicitly')
            if read_json(frozen)['identity'] != identity:
                raise ValueError('Frozen suite/config/code changed')
        else:
            atomic_json(frozen, dict(identity=identity, config=cfg, evaluate_holdouts_requested=evaluate_holdouts))
        frozen_yaml = suite / 'config.yaml'
        import yaml
        text = yaml.safe_dump(cfg, sort_keys=True)
        if frozen_yaml.exists() and frozen_yaml.read_text() != text:
            raise ValueError('Frozen YAML altered')
        if not frozen_yaml.exists():
            frozen_yaml.write_text(text)
        config = str(frozen_yaml)

        def phase(name, **extra):
            nonlocal phase_name
            phase_name = name
            atomic_json(state / 'status.json', dict(status='running', stage=name, pid=os.getpid(), updated=time.time(), **extra))
            event(name, **extra)
        from .environment import capture
        capture(root, suite / f'environment_supervisor_{time.time_ns()}.json', config_hash=digest(cfg))
        phase('runtime_preflight', python=sys.executable, nproc=cfg['runtime']['nproc'])
        launch = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1', f"--nproc-per-node={cfg['runtime']['nproc']}"]
        rc = _child(root, launch + ['-m', 'orcatca.smoke', '--config', config], lock_fd)
        if rc:
            raise RuntimeError(f'Model/collective smoke failed: {rc}')
        for stage in ('prepare', 'cache', 'audit'):
            phase(stage)
            rc = _child(root, [sys.executable, '-m', 'orcatca.stages', stage,
                               '--config', config], lock_fd)
            if rc:
                raise RuntimeError(f'{stage} failed: {rc}')
            if stopfile.exists():
                raise InterruptedError(f'Stop requested after {stage}')
        for fold in range(cfg['experiment']['folds']):
            for method in cfg['experiment']['methods']:
                for seed in cfg['experiment']['seeds']:
                    if stopfile.exists():
                        raise InterruptedError('Stop requested')
                    phase('training', fold=fold, method=method, seed=seed)
                    command = launch + ['-m', 'orcatca.train', '--config', config, '--fold', str(fold), '--method', method, '--seed', str(seed)]
                    if resume:
                        command += ['--resume']
                    rc = _child(root, command, lock_fd)
                    if stopfile.exists():
                        raise InterruptedError('Stopped after completed epoch')
                    if rc:
                        raise RuntimeError(f'Training failed: {method}/seed{seed}/fold{fold}, code {rc}')
        stage = 'evaluate' if evaluate_holdouts else 'summary'
        phase(stage, all_fits_frozen=True)
        rc = _child(root, [sys.executable, '-m', 'orcatca.stages', stage,
                           '--config', config], lock_fd)
        if rc:
            raise RuntimeError(f'{stage} failed: {rc}')
        atomic_json(state / 'status.json', dict(status='completed', stage='suite_complete', holdout_evaluated=evaluate_holdouts, pid=os.getpid()))
        event('suite_complete', holdout_evaluated=evaluate_holdouts)
    except InterruptedError:
        event('suite_stopped')
        atomic_json(state / 'status.json', dict(status='stopped', pid=os.getpid()))
        return 0
    except BaseException as exc:
        logger.exception('Supervisor failed')
        atomic_json(state / 'status.json', dict(status='failed', stage=phase_name, pid=os.getpid(),
                    exception=type(exc).__name__, message=str(exc), updated=time.time(),
                    details='See state/logs, smoke.rank*.json and bootstrap.log'))
        return 1
    finally:
        close()
        os.close(lock_fd)
    return 0
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['_worker'])
    p.add_argument('--config', required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--evaluate-holdouts', action='store_true')
    p.add_argument('--lock-fd', type=int, required=True)
    a = p.parse_args()
    raise SystemExit(worker(a.config, a.resume, a.lock_fd, a.evaluate_holdouts))
