"""Paired-view training with source-only selection and threshold calibration."""
from __future__ import annotations
import argparse
from contextlib import nullcontext
import copy
import gzip
import json
import math
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset
from . import __version__
from .cache import verify as verify_cache
from .config import METHODS, load
from .dataset import AudioDataset, init_worker
from .environment import capture
from .logging import configure, event, close, logger
from .metrics import evaluate, selection_score, calibrate, fpr_threshold, report
from .model import AcousticNet
from .objective import training_weights, loss_per_sample, score_probability, output_contract
from .runtime import Runtime
from .transforms import transform
from .tail import negative_tail
from .calibration import operating_rules
from .utils import project_root, output_path, read_rows, read_json, atomic_json, sha256, digest, code_digest

def primary(rt, fn):
    result = error = None
    if rt.rank == 0:
        try:
            result = fn()
        except Exception as exc:
            error = exc
    flag = torch.tensor([int(error is not None)], device=rt.device, dtype=torch.int32)
    if rt.world > 1:
        dist.broadcast(flag, 0)
    if flag.item():
        if error:
            raise error
        raise RuntimeError('Rank zero failed; inspect its log')
    return result

def atomic_torch(path, value):
    tmp = path.with_name('.' + path.name + '.tmp')
    torch.save(value, tmp)
    os.replace(tmp, path)

def loader(dataset, cfg, rt, training=False, seed=0, persistent=True):
    if training:
        sampler = DistributedSampler(dataset, num_replicas=rt.world, rank=rt.rank, shuffle=True, seed=seed, drop_last=False)
        data = dataset
    else:
        data = Subset(dataset, list(range(rt.rank, len(dataset), rt.world)))
        sampler = None
    opts = dict(batch_size=cfg['experiment']['batch_size'], sampler=sampler, shuffle=False, num_workers=cfg['runtime']['workers'], pin_memory=False, drop_last=False, generator=torch.Generator().manual_seed(seed + 9001), worker_init_fn=init_worker)
    if opts['num_workers']:
        opts.update(multiprocessing_context='spawn', prefetch_factor=2, persistent_workers=persistent)
    return DataLoader(data, **opts)

def close_loader(data_loader):
    if data_loader is None:
        return
    iterator = getattr(data_loader, '_iterator', None)
    if iterator is not None:
        iterator._shutdown_workers()
        data_loader._iterator = None

@torch.no_grad()
def predict(model, ds, cfg, rt, out, tag, data_loader=None, method=None):
    model.eval()
    indices = []
    probabilities = []
    owned = data_loader is None
    if owned:
        data_loader = loader(ds, cfg, rt, persistent=False)
    try:
        for batch in data_loader:
            x, index = batch[0], batch[-1]
            with rt.autocast():
                logits = model(x.to(rt.device))
            p = score_probability(logits, method or METHODS['natural_mixstyle']).cpu().numpy()
            indices.append(index.numpy())
            probabilities.append(p)
    finally:
        if owned:
            close_loader(data_loader)
    shard = out / f'.{tag}.rank{rt.rank}.npz'
    with shard.open('wb') as f:
        np.savez(f, index=np.concatenate(indices) if indices else np.empty(0, np.int64), p=np.concatenate(probabilities) if probabilities else np.empty(0, np.float32))
    rt.barrier()

    def merge():
        parts = []
        for rank in range(rt.world):
            with np.load(out / f'.{tag}.rank{rank}.npz', allow_pickle=False) as item:
                parts.append((item['index'].copy(), item['p'].copy()))
        index = np.concatenate([v[0] for v in parts])
        p = np.concatenate([v[1] for v in parts])
        if len(index) != len(ds) or not np.array_equal(np.sort(index), np.arange(len(ds))):
            raise ValueError('Incomplete or duplicate distributed predictions')
        if not np.isfinite(p).all():
            raise FloatingPointError('Nonfinite predictions')
        return p[np.argsort(index)]
    result = primary(rt, merge)
    rt.barrier()
    if rt.rank == 0:
        for rank in range(rt.world):
            (out / f'.{tag}.rank{rank}.npz').unlink()
    return result

def rng_state(rt):
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(), npu=torch.npu.get_rng_state(rt.device) if rt.kind == 'npu' else None)

def restore_rng(rt, state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if rt.kind == 'npu':
        torch.npu.set_rng_state(state['npu'], rt.device)

@torch.no_grad()
def update_ema(ema, model, decay):
    for target, value in zip(ema.parameters(), model.parameters()):
        target.mul_(decay).add_(value.detach(), alpha=1 - decay)
    for target, value in zip(ema.buffers(), model.buffers()):
        target.copy_(value)

def write_predictions(path, rows, probabilities):
    tmp = path.with_name('.' + path.name + '.tmp')
    with gzip.open(tmp, 'wt', encoding='utf-8') as f:
        for row, p in zip(rows, probabilities):
            item = {k: row[k] for k in ('id', 'recording', 'site', 'group', 'block', 'label', 'raw_class') if k in row}
            f.write(json.dumps(item | {'p': float(p)}, sort_keys=True) + '\n')
    os.replace(tmp, path)

def protocol_inputs(root, cfg, fold):
    from .references import protocol_directory
    directory = protocol_directory(root, cfg)
    protocol = read_json(directory / 'complete.json')
    for name, expected in protocol['files'].items():
        if sha256(directory / name) != expected:
            raise ValueError('Protocol file changed: ' + name)
    manifest = directory / protocol['folds'][fold]['manifest']
    rows = read_rows(manifest)
    from .splitting import validate_splits
    validate_splits(rows)
    records = read_json(directory / 'recordings.json')
    for rel in {r['audio'] for r in rows}:
        stat = (root / 'dclde_2027_killer_whales' / rel).stat()
        record = records[rel]
        if (stat.st_size, stat.st_mtime_ns) != (record['size'], record['mtime_ns']):
            raise ValueError('Raw file identity changed: ' + rel)
    cache_ref = root / f"runs/{cfg['name']}/cache.json"
    cache_path = None
    cache_hash = None
    if cfg['cache']['enabled']:
        cache = read_json(cache_ref)
        cache_path = Path(cache['path'])
        if sha256(cache_path) != cache['summary_sha256']:
            raise ValueError('Cache summary changed')
        metadata = verify_cache(cache_path, False)
        if metadata['protocol_identity'] != protocol['identity']:
            raise ValueError('Cache/protocol mismatch')
        cache_hash = cache['summary_sha256']
    return (protocol, manifest, rows, records, cache_path, cache_hash)

def run(config_path, fold, method, seed, resume=False, stop_after=None):
    cfg = load(config_path)
    root = project_root()
    settings = cfg['experiment']
    if method not in settings['methods'] or seed not in settings['seeds'] or (not 0 <= fold < settings['folds']):
        raise ValueError('Run is outside frozen suite')
    recipe = METHODS[method]
    out = output_path(root, f"runs/{cfg['name']}/fold{fold:02d}/{method}_s{seed}")
    out.mkdir(parents=True, exist_ok=True)
    rank = int(os.getenv('RANK', '0'))
    configure(out / 'logs', 'train', rank, console=rank == 0)
    rt = None
    train_loader = val_loader = None
    try:
        rt = Runtime(cfg['runtime'])
        event('runtime_ready', **rt.metadata())
        if rt.world != cfg['runtime']['nproc']:
            raise ValueError('World size differs from frozen configuration')
        proto, manifest, rows, records, cache_path, cache_hash = protocol_inputs(root, cfg, fold)
        train_rows = [r for r in rows if r['split'] == 'train']
        select_rows = [r for r in rows if r['split'] == 'select']
        cal_rows = [r for r in rows if r['split'] == 'calibrate']
        identity = digest([__version__, code_digest(root), cfg, fold, method, seed, proto['identity'], sha256(manifest), cache_hash, {k: v for k, v in rt.metadata().items() if k != 'device'}])
        statefile = out / 'status.json'
        last = out / 'last.pt'
        bestfile = out / 'best.pt'
        prior = read_json(statefile) if statefile.exists() else None
        if not resume and (prior or last.exists() or bestfile.exists() or (out / 'initialization.json').exists()):
            raise ValueError('Existing run; use --resume')
        if prior and prior.get('identity') != identity:
            raise ValueError('Run inputs, code or environment changed')
        if prior and prior.get('status') == 'completed':
            cal = read_json(out / 'calibration.json')
            if cal['identity'] != identity or sha256(bestfile) != cal['best_sha256']:
                raise ValueError('Completed artifacts changed')
            event('completed_run_skipped')
            return 0

        def record_environment():
            stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
            capture(root, out / f'environment_{stamp}.json', rt, digest(cfg))
        primary(rt, record_environment)
        strength = settings['balance_strength'] if recipe['balance'] else 0.0
        weights, weight_report = training_weights(train_rows, strength, settings['max_weight'])
        primary(rt, lambda: atomic_json(out / 'training_support.json', weight_report))
        rt.seed(seed)
        model = AcousticNet(settings['width'], settings['dropout']).to(rt.device)
        ema = copy.deepcopy(model).eval()
        for p in ema.parameters():
            p.requires_grad_(False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=settings['weight_decay'])
        start = 0
        best = -float('inf')
        bad = 0
        updates = 0
        saved = None
        if resume and last.exists():
            saved = torch.load(last, map_location='cpu', weights_only=False)
            if saved.get('format') != 'orcatca230' or saved['identity'] != identity or saved.get('output_contract') != output_contract(recipe):
                raise ValueError('Checkpoint identity mismatch')
            if not bestfile.exists() or sha256(bestfile) != saved['best_sha256']:
                raise ValueError('Best checkpoint changed')
            model.load_state_dict(saved['model'])
            ema.load_state_dict(saved['ema'])
            optimizer.load_state_dict(saved['optimizer'])
            start = saved['epoch'] + 1
            best = saved['best']
            bad = saved['bad']
            updates = saved['updates']
        if saved is None:
            import hashlib
            h = hashlib.sha256()
            for key, value in model.state_dict().items():
                h.update(key.encode())
                h.update(value.detach().cpu().contiguous().numpy().tobytes())
            init = dict(source='random', pretrained=False, loaded_checkpoint=None,
                        initialization_seed=seed, parameters_sha256=h.hexdigest(), output_contract=output_contract(recipe))
            from .references import initialization_bridge
            bridge = initialization_bridge(root, cfg, fold, seed, h.hexdigest())
            primary(rt, lambda: atomic_json(out / 'bridge.json', bridge))
            primary(rt, lambda: atomic_json(out / 'initialization.json', init))
            event('model_initialized', **init)
        else:
            event('same_run_resumed', checkpoint=str(last), epoch=start)
        network = DDP(model, device_ids=[rt.local] if rt.kind == 'npu' else None, broadcast_buffers=False) if rt.world > 1 else model
        rt.seed(seed + rt.rank * 10007)
        if saved:
            restore_rng(rt, saved['rng'][rt.rank])
        primary(rt, lambda: atomic_json(statefile, dict(status='running', identity=identity, method=method, seed=seed, fold=fold)))
        train_ds = AudioDataset(root, train_rows, records, cfg, weights, cache_path)
        val_ds = AudioDataset(root, select_rows, records, cfg, cache_path=cache_path)
        train_loader = loader(train_ds, cfg, rt, True, seed)
        val_loader = loader(val_ds, cfg, rt)
        accumulate = settings['global_batch'] // (rt.world * settings['batch_size'])
        stopped = False
        for epoch in range(start, settings['epochs']):
            if bad >= settings['patience']:
                break
            if (root / 'state/STOP').exists():
                stopped = True
                break
            train_loader.sampler.set_epoch(epoch)
            batches = len(train_loader)
            if not batches:
                raise ValueError('Empty training loader')
            if epoch < settings['warmup_epochs']:
                scale = (epoch + 1) / max(1, settings['warmup_epochs'])
            else:
                phase = (epoch - settings['warmup_epochs']) / max(1, settings['epochs'] - settings['warmup_epochs'] - 1)
                scale = 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * phase))
            for group in optimizer.param_groups:
                group['lr'] = settings['learning_rate'] * scale
            network.train()
            optimizer.zero_grad(set_to_none=True)
            totals = torch.zeros(5, device=rt.device, dtype=torch.float32)
            tail_totals = torch.zeros(7, device=rt.device, dtype=torch.float32)
            samples = 0
            active = 0
            cross = 0
            waiting = 0.0
            host_compute = 0.0
            rt.sync()
            t0 = last_time = time.monotonic()
            event('epoch_started', epoch=epoch, batches=batches, lr=optimizer.param_groups[0]['lr'], views_per_sample=2)
            for step, batch in enumerate(train_loader):
                now = time.monotonic()
                waiting += now - last_time
                x, y, subtype, w, site, rec, _ = batch
                count = len(y)
                if step % accumulate == 0:
                    end = min(batches, step + accumulate)
                    group_samples = sum((min(settings['batch_size'], len(train_loader.sampler) - j * settings['batch_size']) for j in range(step, end)))
                y_host, subtype_host = y.numpy(), subtype.numpy()
                x = x.to(rt.device)
                y = y.to(rt.device)
                subtype = subtype.to(rt.device)
                w = w.to(rt.device)
                augmented, diagnostic = transform(x, recipe['transform'], cfg['augmentation'], site.numpy(), rec.numpy(), seed, epoch, step + rt.rank * 1000000)
                active += diagnostic['active']
                cross += diagnostic['cross_site']
                update = (step + 1) % accumulate == 0 or step + 1 == batches
                context = network.no_sync() if isinstance(network, DDP) and (not update) else nullcontext()
                with context:
                    with rt.autocast():
                        logits = network(torch.cat((x, augmented), 0))
                    per_sample, supervised, aux, js = loss_per_sample(logits[:count], logits[count:], y, subtype, settings, recipe, epoch)
                    tail_loss, tail_info = negative_tail(
                        logits[:count], logits[count:], y_host, subtype_host, rec.numpy(),
                        settings, recipe.get('tail'), seed, epoch, step + rt.rank * 1000000)
                    loss = ((per_sample * w).sum() + count * tail_loss) / group_samples
                    loss.backward()
                with torch.no_grad():
                    score = score_probability(logits[:count], recipe).clamp(1e-7, 1-1e-7)
                    plain = torch.nn.functional.binary_cross_entropy(score, y.float(), reduction='sum')
                    tail_totals += torch.stack((tail_loss.detach() * count,
                        *[torch.as_tensor(tail_info[k], device=rt.device, dtype=torch.float32)
                          for k in ('eligible_groups', 'negative_recordings', 'selected_recordings',
                                    'sparse_groups', 'hw_groups', 'ab_groups')]))
                    totals += torch.stack(((per_sample.detach() * w).sum() + tail_loss.detach() * count, supervised.detach().sum(), aux.detach().sum(), js.detach().sum(), plain))
                samples += count
                if update:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    updates += 1
                    update_ema(ema, model, settings['ema_decay'])
                if cfg['runtime']['profile_steps']:
                    rt.sync()
                last_time = time.monotonic()
                host_compute += last_time - now
                if rt.rank == 0 and (step == 0 or (step + 1) % settings['log_every'] == 0 or step + 1 == batches):
                    event('train_progress', epoch=epoch, batch=step + 1, batches=batches, loss=float(totals[0] / samples), updates=updates)
            rt.sync()
            elapsed = time.monotonic() - t0
            total = rt.sum(totals.cpu().tolist() + [samples, active, cross])
            tail_total = rt.sum(tail_totals.cpu().tolist())
            event('selection_started', epoch=epoch, metric=settings['selection_metric'])
            p = predict(ema, val_ds, cfg, rt, out, 'select', val_loader, recipe)

            def select_and_save():
                nonlocal best, bad
                value = selection_score(select_rows, p, settings['selection_metric'], settings['selection_max_fpr'])
                if value > best + 1e-10:
                    best = value
                    bad = 0
                    atomic_torch(bestfile, dict(format='orcatca230', identity=identity, model={k: v.detach().cpu() for k, v in ema.state_dict().items()}, width=settings['width'], dropout=settings['dropout'], epoch=epoch, selection_score=best, selection_metric=settings['selection_metric'], manifest_sha256=sha256(manifest), output_contract=output_contract(recipe), method=method))
                    write_predictions(out / 'select_best.jsonl.gz', select_rows, p)
                else:
                    bad += 1
                metrics = dict(epoch=epoch, selection_score=value, selection_metric=settings['selection_metric'], best_selection_score=best, bad_epochs=bad, weighted_train_loss=total[0] / total[5], detection_ce=total[1] / total[5], subtype_ce=total[2] / total[5], js=total[3] / total[5], plain_train_ce=total[4] / total[5], seconds=elapsed, rank0_data_wait_seconds=waiting, rank0_host_compute_seconds=host_compute, timing_scope='Host enqueue/wait unless profile_steps=true; not accelerator utilization', examples_including_sampler_padding=int(total[5]), unique_source_examples=len(train_rows), augmentation_active=int(total[6]), cross_site_donors=int(total[7]), lr=optimizer.param_groups[0]['lr'], metrics_at_fixed_05=report(select_rows, p, 0.5, settings['selection_max_fpr']))
                metrics['output_contract'] = output_contract(recipe)
                metrics['tail'] = dict(weighted_loss=tail_total[0] / total[5],
                    eligible_groups=int(tail_total[1]), negative_recordings=int(tail_total[2]),
                    selected_recordings=int(tail_total[3]), sparse_groups=int(tail_total[4]),
                    hw_groups=int(tail_total[5]), ab_groups=int(tail_total[6]),
                    scope='Rank-local minibatch recording means; not full-data CVaR or pAUC')
                atomic_json(out / f'epoch_{epoch:03d}.json', metrics)
                event('epoch_completed', epoch=epoch, selection_score=value, best=best, bad_epochs=bad, seconds=elapsed)
            primary(rt, select_and_save)
            shared = [best, bad]
            if rt.world > 1:
                dist.broadcast_object_list(shared, src=0)
                if rt.rank:
                    best, bad = (float(shared[0]), int(shared[1]))
            states = [None] * rt.world
            if rt.world > 1:
                dist.all_gather_object(states, rng_state(rt))
            else:
                states = [rng_state(rt)]

            def checkpoint():
                atomic_torch(last, dict(format='orcatca230', identity=identity, epoch=epoch, best=best, bad=bad, updates=updates, model={k: v.detach().cpu() for k, v in model.state_dict().items()}, ema={k: v.detach().cpu() for k, v in ema.state_dict().items()}, optimizer=optimizer.state_dict(), rng=states, best_sha256=sha256(bestfile), output_contract=output_contract(recipe), method=method))
                atomic_json(statefile, dict(status='running', identity=identity, epoch=epoch, best=best, bad=bad))
            primary(rt, checkpoint)
            if stop_after is not None and epoch + 1 >= stop_after:
                stopped = True
                break
            if (root / 'state/STOP').exists():
                stopped = True
                break
        if stopped:
            primary(rt, lambda: atomic_json(statefile, dict(status='stopped', identity=identity, last_checkpoint=last.name if last.exists() else None)))
            event('run_stopped')
            return 20
        close_loader(train_loader)
        close_loader(val_loader)
        train_loader = val_loader = None
        beststate = torch.load(bestfile, map_location='cpu', weights_only=True)
        ema.load_state_dict(beststate['model'])
        event('threshold_calibration_started', epoch=beststate['epoch'], threshold_search='all_distinct_scores')
        cds = AudioDataset(root, cal_rows, records, cfg, cache_path=cache_path)
        cp = predict(ema, cds, cfg, rt, out, 'calibrate', method=recipe)

        def finish():
            threshold = calibrate(cal_rows, cp)
            rules, support_report = operating_rules(cal_rows, cp, cfg['evaluation'])
            event('calibration_support', **support_report)
            if support_report['rules_withheld']:
                logger.warning('Low-FPR rules withheld: missing or sparse source negative subtypes')
            calibration = dict(format='orcatca230', identity=identity, best_sha256=sha256(bestfile), threshold=threshold, operating_thresholds=rules, checkpoint_epoch=beststate['epoch'], selection_score=beststate['selection_score'], selection_metric=settings['selection_metric'], calibration_metrics=report(cal_rows, cp, threshold, settings['selection_max_fpr']), operating_metrics={k: report(cal_rows, cp, t, settings['selection_max_fpr']) for k, t in rules.items()}, calibration_is_not_test=True, probability_calibrated=False, scope='Empirical source-calibration thresholds; no future FPR guarantee')
            calibration['operating_support'] = support_report
            calibration['output_contract'] = output_contract(recipe)
            calibration['method'] = method
            atomic_json(out / 'calibration.json', calibration)
            write_predictions(out / 'calibration.jsonl.gz', cal_rows, cp)
            atomic_json(statefile, dict(status='completed', identity=identity, checkpoint_epoch=beststate['epoch'], threshold=threshold))
            event('run_completed', method=method, seed=seed, fold=fold, threshold=threshold, holdout_evaluated=False)
        primary(rt, finish)
        return 0
    except BaseException:
        logger.exception('Training failed')
        if rank == 0:
            previous = read_json(out / 'status.json') if (out / 'status.json').exists() else {}
            atomic_json(out / 'status.json', dict(previous, status='failed', resume_from='last completed epoch'))
        raise
    finally:
        close_loader(train_loader)
        close_loader(val_loader)
        if rt:
            rt.close()
        close()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--method', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after-epochs', type=int)
    args = parser.parse_args()
    result = run(args.config, args.fold, args.method, args.seed, args.resume, args.stop_after_epochs)
    raise SystemExit(0 if result == 20 else result)
