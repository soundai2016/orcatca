"""Project commands."""
from __future__ import annotations
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
from . import __version__
from .utils import project_root, output_path

def main():
    parser = argparse.ArgumentParser(description='Acoustic source-generalization experiments')
    parser.add_argument('--version', action='version', version=__version__)
    sub = parser.add_subparsers(dest='command', required=True)
    item = sub.add_parser('start')
    item.add_argument('--config')
    group = item.add_mutually_exclusive_group()
    group.add_argument('--resume', action='store_true')
    group.add_argument('--fresh', action='store_true', help='Create an empty suite and randomly initialize every model')
    item.add_argument('--pilot', action='store_true')
    item.add_argument('--evaluate-holdouts', action='store_true')
    item = sub.add_parser('supplement')
    item.add_argument('--source', required=True, help='Completed local source suite; no weights are reused')
    item.add_argument('--pilot', action='store_true')
    item.add_argument('--with-baselines', action='store_true')
    item.add_argument('--evaluate-holdouts', action='store_true')
    item.add_argument('--prepare-only', action='store_true')
    item = sub.add_parser('analyze')
    item.add_argument('--source', required=True, help='Result suite directory or report ZIP')
    item.add_argument('--output')
    item.add_argument('--reference')
    item.add_argument('--expect-trace', help='Verify settings and every consumed input against a completed analysis.json')
    item.add_argument('--methods', nargs='+', default=[
        'natural_mixstyle', 'natural_coarse_mixstyle',
        'natural_threeclass_mixstyle', 'natural_subtype_mixstyle'])
    item.add_argument('--repeats', type=int, default=1000)
    item.add_argument('--crossfit-repeats', type=int, default=5)
    item.add_argument('--seed', type=int, default=20261029)
    item = sub.add_parser('review')
    item.add_argument('--source', required=True)
    item.add_argument('--output')
    item.add_argument('--reference')
    item.add_argument('--count', type=int, default=240)
    item.add_argument('--export-audio', action='store_true')
    sub.add_parser('status')
    item = sub.add_parser('stop')
    item.add_argument('--now', action='store_true')
    sub.add_parser('logs')
    item = sub.add_parser('report')
    item.add_argument('--name')
    item = sub.add_parser('evaluate')
    item.add_argument('--name')
    item.add_argument('--confirm-holdout', action='store_true')
    item = sub.add_parser('compare')
    item.add_argument('--a', required=True)
    item.add_argument('--b', required=True)
    item.add_argument('--role', default='holdout', choices=['holdout', 'calibration', 'select_best'])
    item.add_argument('--unit', default='block', choices=['block', 'recording'])
    item.add_argument('--repeats', type=int, default=2000)
    item.add_argument('--metric', default='macro_pauc', choices=['auc', 'macro_auc', 'macro_pauc'])
    item.add_argument('--max-fpr', type=float, default=0.1)
    item = sub.add_parser('prepare')
    item.add_argument('--config', default='configs/full.yaml')
    item = sub.add_parser('smoke')
    item.add_argument('--config', default='configs/full.yaml')
    args = parser.parse_args()
    root = project_root()
    try:
        if args.command == 'supplement':
            from .supplement import create
            config = create(root, args.source, args.pilot, args.with_baselines)
            print('Frozen request:', config, flush=True)
            if not args.prepare_only:
                from .control import start
                start(config, evaluate_holdouts=args.evaluate_holdouts)
        elif args.command == 'analyze':
            from .analysis import run
            print(run(root, args.source, args.output, args.methods, args.repeats, args.crossfit_repeats, args.seed, args.reference, args.expect_trace))
        elif args.command == 'review':
            from .review import create
            print(create(root, args.source, args.output, args.count, args.export_audio, args.reference))
        elif args.command == 'start':
            from .control import start
            from .control import _job
            config = root / (args.config or ('configs/pilot.yaml' if args.pilot else 'configs/full.yaml'))
            if args.resume and args.config is None and not args.pilot:
                job = _job(root)
                if not job:
                    raise ValueError('No current-release request to resume; use --fresh')
                config = Path(job['config'])
            start(config, args.resume, args.evaluate_holdouts, args.fresh)
        elif args.command == 'status':
            from .control import status
            status()
        elif args.command == 'stop':
            from .control import stop
            stop(args.now)
        elif args.command == 'logs':
            path = root / 'state/bootstrap.log'
            if not path.exists():
                raise FileNotFoundError('No background log')
            subprocess.run(['tail', '-n', '80', '-F', str(path)], check=False)
        elif args.command == 'report':
            from .results import pack
            from .control import _job
            name = args.name or (_job(root) or {}).get('suite_name', 'semantic')
            print(pack(root, name))
        elif args.command == 'compare':
            from .results import compare
            print(json.dumps(compare(root, args.a, args.b, args.repeats, args.role, args.unit, args.metric, args.max_fpr), indent=2))
        elif args.command in ('prepare', 'evaluate'):
            lock = output_path(root, 'state/supervisor.lock')
            lock.parent.mkdir(parents=True, exist_ok=True)
            with lock.open('a+') as stream:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                if args.command == 'prepare':
                    from .config import load
                    from .data import prepare
                    from .logging import configure, close
                    configure(lock.parent / 'logs', 'prepare')
                    try:
                        prepare(root, load(root / args.config)['data'])
                    finally:
                        close()
                else:
                    from .results import evaluate_all
                    from .control import _job
                    name = args.name or (_job(root) or {}).get('suite_name', 'semantic')
                    evaluate_all(root, name, args.confirm_holdout)
        elif args.command == 'smoke':
            from .config import load
            cfg = load(root / args.config)
            subprocess.run([sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1', f"--nproc-per-node={cfg['runtime']['nproc']}", '-m', 'orcatca.smoke', '--config', str(root / args.config)], check=True)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1
    return 0
