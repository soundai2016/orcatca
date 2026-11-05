"""Isolated preparation and evaluation stages release memory between jobs."""
from __future__ import annotations
import argparse
from pathlib import Path
from .config import load
from .logging import configure, close, logger
from .utils import project_root, output_path, read_json, atomic_json, sha256


def run(stage: str, config: str) -> None:
    root = project_root()
    cfg = load(config)
    configure(output_path(root, 'state/logs'), stage)
    try:
        if stage == 'prepare':
            if cfg.get('input_reference'):
                from .references import verify_reference
                from .logging import event
                info = verify_reference(root, cfg)
                atomic_json(output_path(root, f"runs/{cfg['name']}/input_reference.json"), cfg['input_reference'])
                event('frozen_inputs_verified', protocol_identity=info['identity'], weights_reused=False)
            else:
                from .data import prepare
                prepare(root, cfg['data'])
        elif stage == 'cache':
            from .cache import build
            if cfg.get('input_reference'):
                from .references import reference_cache
                cache = reference_cache(root, cfg, cfg['cache']['verify_sha256'])
            else:
                info = read_json(root / 'state/protocol/complete.json')
                cache = build(root, cfg, info)
            if cache:
                atomic_json(output_path(root, f"runs/{cfg['name']}/cache.json"),
                            dict(path=cache['path'], summary_sha256=sha256(Path(cache['path'])),
                                 windows=cache['windows'], dtype=cache['dtype']))
        elif stage == 'audit':
            from .audit import run as audit
            audit(root, cfg)
        elif stage == 'evaluate':
            from .results import evaluate_all
            evaluate_all(root, cfg['name'], True)
        else:
            from .results import summarize
            summarize(root, cfg['name'])
    except BaseException:
        logger.exception('Stage failed')
        raise
    finally:
        close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'cache', 'audit', 'summary', 'evaluate'])
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    run(args.stage, args.config)
