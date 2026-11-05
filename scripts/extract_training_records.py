#!/usr/bin/env python3
"""Read-only extraction of training records for 36 fits; uses the Python standard library.

python extract_training_records.py --reports /path/to/verified_reports --output /new/output
python scripts/extract_training_records.py --verify-runs runs --records result/training_records
Optional: --checkpoint-root /original/project/runs (expects each suite name below).
All inputs are read-only. Output directory must be new or lack the two output files.
"""
import argparse
import collections
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import re
import statistics as st
from zipfile import ZipFile

SUITES = {'jel220_20261026T021531Z_ac09b0f3': ['natural_mixstyle','natural_subtype_mixstyle'],
          'semantic230_20261029T160625Z_9e1b11a8': ['natural_coarse_mixstyle','natural_threeclass_mixstyle']}
SEEDS = [17,29,43]

def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def stats(x):return {'minimum':min(x),'median':st.median(x),'maximum':max(x),'sum':math.fsum(x),'count':len(x)}

def check(test,what):
    if not test:raise ValueError(what)

def extract(reports, output, checkpoint_root=None):
    fits=[]; suites={}; envs=[]; all_times=[]; timings=[]; all_suite_epochs=[]
    for suite,methods in SUITES.items():
        path=reports/(suite+'_report.zip')
        with ZipFile(path) as z:
            names=z.namelist();check(len(names)==len(set(names)),'Duplicate ZIP member')
            sums=json.loads(z.read('REPORT_SHA256.json'))
            for key,expected in sums.items():
                with z.open(key) as f:check(hashlib.file_digest(f,'sha256').hexdigest()==expected,key)
            def read(key):
                check(key in sums,'Used JSON lacks source checksum: '+key);return json.loads(z.read(key))
            prefix,=[p for p in ('runs/','result/runs/') if p+'config.json' in sums]
            cfg=read(prefix+'config.json')['config']; ex=cfg['experiment'];rcfg=cfg['runtime']
            check(cfg['name']==suite and ex['seeds']==SEEDS and ex['folds']==3,'Config/analysis scope differs')
            ptn=re.compile(re.escape(prefix)+r'fold\d\d/[^/]+_s\d+/status\.json$')
            completed=sum(read(key)['status']=='completed' for key in sums if ptn.match(key))
            suites[suite]={'report':path.name,'report_bytes':path.stat().st_size,'report_sha256':digest(path),
                'suite_code_hashes':{},'all_completed_fit_count':completed,'study_fit_count':18,
                'included_methods':methods,'other_configured_methods':sorted(set(ex['methods'])-set(methods)),
                'verified_members':len(sums),'sha256_manifest_digest':hashlib.sha256(z.read('REPORT_SHA256.json')).hexdigest(),
                'unmanifested_members':sorted(set(names)-set(sums)),
                'checkpoint_members':[s for s in names if s.endswith(('.pt','.pth','.safetensors'))]}
            # Descriptive context only: all completed fits, including 36 other-objective fits.
            suite_epochs=[]
            for key in sorted(k for k in sums if ptn.match(k)):
                if read(key)['status']!='completed':continue
                base=key[:-len('status.json')]
                keys=sorted(k for k in sums if k.startswith(base+'epoch_') and k.endswith('.json'))
                logs=[read(k) for k in keys];check(len(logs)>0 and [v['epoch'] for v in logs]==list(range(len(logs))),base+'incomplete epoch logs')
                times=[v['seconds'] for v in logs];check(all(math.isfinite(v) and v>0 for v in times),'Invalid epoch duration')
                suite_epochs.extend(times)
            all_suite_epochs.extend(suite_epochs)
            suites[suite].update(all_objectives_observed_epochs=len(suite_epochs),all_objectives_training_loop_hours=math.fsum(suite_epochs)/3600)
            check(ex['width']==32,'Parameter count assumes a width-32 model')
            for i in range(3):
                for method in methods:
                    for seed in SEEDS:
                        fold=f'fold{i:02d}'; base=prefix+fold+'/'+method+f'_s{seed}/'
                        envname,=[key for key in sums if key.startswith(base+'environment_') and key.endswith('.json')]
                        env=read(envname); rt=env['runtime']; s=read(base+'status.json'); init=read(base+'initialization.json'); cal=read(base+'calibration.json'); h=read(base+'holdout.json')
                        check(s['status']=='completed' and s['identity']==cal['identity'] and s['checkpoint_epoch']==cal['checkpoint_epoch'] and cal['best_sha256']==h['best_sha256'],base+'identity or status mismatch')
                        check(init['source']=='random' and not init['pretrained'] and init['loaded_checkpoint'] is None and init['initialization_seed']==seed,base+'unexpected initialization')
                        en=dt.datetime.fromisoformat(env['timestamp']); zs=dt.datetime(*z.getinfo(envname).date_time); ze=dt.datetime(*z.getinfo(base+'status.json').date_time)
                        # ZIP mtimes are local DOS timestamps (2s ticks). Derive offset via environment UTC; tolerance accounts for rounding / file-write delay.
                        offset=(zs-en.replace(tzinfo=None)).total_seconds(); offset_hr=round(offset/3600);check(abs(offset-3600*offset_hr)<5 and ze>=zs,base+'timestamp mismatch')
                        end=(ze-dt.timedelta(hours=offset_hr)).replace(tzinfo=dt.timezone.utc); wall=(ze-zs).total_seconds();all_times.append((en,end))
                        epkeys=sorted(k for k in sums if k.startswith(base+'epoch_') and k.endswith('.json'))
                        ep=[read(k) for k in epkeys];check([e['epoch'] for e in ep]==list(range(len(ep))) and 0<=s['checkpoint_epoch']<len(ep),base+'non-contiguous epochs')
                        secs=[e['seconds'] for e in ep];timings.extend(secs); check(all(math.isfinite(v) and v>0 for v in secs) and math.fsum(secs)<wall+5,base+'invalid epoch durations')
                        smi=env['npu_smi']; chipnames=set(re.findall(r'(?m)^\|\s*\d+\s+(Ascend\w+)\s*\|',smi['stdout'])); check(smi['returncode']==0 and len(chipnames)==1,'Unrecognized hardware log')
                        cann_file,=[v for k,v in env['cann_files'].items() if k.endswith('ascend_toolkit_install.info')]; driver_file,=[v for k,v in env['cann_files'].items() if k.endswith('/driver/version.info')];
                        cann=re.search(r'(?m)^version=(.+)$',cann_file['text'])[1];driver=re.search(r'(?m)^Version=(.+)$',driver_file['text'])[1]
                        arch=dict(device_model_as_reported=next(iter(chipnames)),hardware_submodel=None,model_source='npu-smi info in each fit environment JSON',cpu_arch=env['architecture'],os=env['platform'],logical_cpus=env['cpu_count'],
                                  logical_npus_visible=rt['visible_npu'],world_size=rt['world'],runtime_device_rank0=rt['device'],distributed_backend=rt['backend'],precision=rt['precision'],
                                  torch=rt['torch'],torch_npu=rt['torch_npu'],cann=cann,ascend_driver=driver,python=env['python'].split()[0],
                                  numpy=env['packages']['numpy'],scikit_learn=env['packages']['scikit-learn'],soundfile=env['packages']['soundfile'],
                                  deterministic_algorithms=env['deterministic_algorithms'],jit_compile=rt['formats']['jit_compile'],
                                  step_profiling=rcfg['profile_steps'],microbatch_original_windows_per_rank=ex['batch_size'],nominal_global_batch_original_windows=ex['global_batch'],
                                  gradient_accumulation_steps=ex['global_batch']//(ex['batch_size']*rt['world']),max_epochs=ex['epochs'],patience_epochs=ex['patience'],
                                  source_selection_metric=ex['selection_metric'],selection_max_fpr=ex['selection_max_fpr'],two_views_per_window=True)
                        check(arch['world_size']==rcfg['nproc'] and arch['precision']==rcfg['precision'],base+'runtime mismatch');envs.append(arch)
                        suites[suite]['suite_code_hashes'][env['packages']['orcatca']]=env['code_sha256']
                        best_bytes=last_bytes=None
                        if checkpoint_root is not None:
                            loc=checkpoint_root/suite/fold/(method+f'_s{seed}');best=loc/'best.pt';last=loc/'last.pt'
                            if best.is_file():
                                check(digest(best)==cal['best_sha256'],str(best)+' identity differs');best_bytes=best.stat().st_size
                            if last.is_file():last_bytes=last.stat().st_size
                        fits.append(dict(suite=suite,fold=fold,method=method,seed=seed,observed_epochs=len(ep),selected_epoch_zero_based=s['checkpoint_epoch'],
                            epoch_training_loop_seconds_sum=math.fsum(secs),rank0_data_wait_seconds_sum=math.fsum(e['rank0_data_wait_seconds'] for e in ep),
                            rank0_host_enqueue_compute_seconds_sum=math.fsum(e['rank0_host_compute_seconds'] for e in ep),
                            environment_capture_utc=en.isoformat(),completion_utc_approx_from_zip_mtime=end.isoformat(),
                            environment_to_completed_status_seconds_approx=wall,fit_identity=s['identity'],best_pt_sha256=cal['best_sha256'],
                            initialization_parameters_sha256=init['parameters_sha256'],measured_best_pt_bytes=best_bytes,local_unverified_last_pt_bytes=last_bytes))
    check(len(fits)==36 and all(e==envs[0] for e in envs),'Expected 36 fits with shared environment/config fields')
    width=32; count=width*49+2*width; cin=width
    for i in range(4):
        cout=width*2**i
        for _ in range(2):
            count+=9*cin*cout+2*cout+9*cout*cout+2*cout+(cin*cout+2*cout if cin!=cout else 0);cin=cout
    count+=2*cin*5+5
    start=min(v[0] for v in all_times);end=max(v[1] for v in all_times); loop=[r['epoch_training_loop_seconds_sum'] for r in fits];wall=[r['environment_to_completed_status_seconds_approx'] for r in fits]
    summary=dict(schema='training_records.v1',fits=36,reports=suites,environment=envs[0],all_reported_objectives_context=dict(completed_fits=sum(v['all_completed_fit_count'] for v in suites.values()),observed_epochs=len(all_suite_epochs),training_loop_hours=math.fsum(all_suite_epochs)/3600,includes_study_fits=True,nonstudy_fit_count=sum(v['all_completed_fit_count'] for v in suites.values())-len(fits),nonstudy_epochs=len(all_suite_epochs)-len(timings),nonstudy_training_loop_hours=(math.fsum(all_suite_epochs)-math.fsum(timings))/3600),observed_epochs_per_fit=stats([r['observed_epochs'] for r in fits]),
            training_seconds_per_epoch=stats(timings),training_seconds_per_fit=stats(loop),total_training_loop_hours=math.fsum(loop)/3600,
            training_loop_nominal_allocated_npu_hours=envs[0]['world_size']*math.fsum(loop)/3600,
            approximate_fit_wall_seconds=stats(wall),sum_approximate_fit_wall_hours=math.fsum(wall)/3600,
            overall_calendar_window=dict(first_fit_environment_utc=start.isoformat(),last_completed_status_utc_approx=end.isoformat(),hours=(end-start).total_seconds()/3600),
            storage=dict(serialized_checkpoints_present_in_reports=False,measured_best_file_count=sum(r['measured_best_pt_bytes'] is not None for r in fits),
                         width32_model_trainable_parameters=count,fp32_parameter_payload_bytes=count*4,
                         scope='Theoretical FP32 parameter payload from AcousticNet in model.py: excludes PyTorch file metadata, optimizer, gradients and activations. best.pt is EMA selection; last.pt is a different, larger resume artifact. No serialized size is inferred.'),
            interpretation=['Rank-0 epoch JSON seconds: synchronized elapsed time across the training-loader loop (source minibatches); do not sum across ranks. Selection, calibration, holdout, checkpoint writing, setup and data preparation are outside the loop.',
            'rank0_data_wait_seconds and rank0_host_compute_seconds measure host-side wait/enqueue, not separately profiled accelerator utilization; profile_steps was false.',
            'Approximate fit wall time uses local ZIP member mtimes for the unique per-fit environment capture and completed status.json: 2-second DOS timestamp resolution. Checksums cover member contents, not source filesystem timestamps; the whole-report hash anchors the ZIP metadata as received. This envelope includes selection/checkpoints/calibration but omits launch before capture, holdout afterwards and preparation. It is not a scheduler allocation record.',
            'Overall calendar window includes intervening experiments and idle time. Sum of 36 fit times and calendar duration answer different questions. Device-hours use two allocated devices per fit; no utilization, energy consumption or FLOP measurement is available.',
            'ZIP/report byte lengths include other experiments and metadata. Weight SHA-256 authenticates a file but does not encode file size. Optional --checkpoint-root reports local best.pt size only after digest matches; last.pt uses local stat only.'])
    fits.sort(key=lambda r:(r['fold'],r['method'],r['seed']));output.mkdir(parents=True,exist_ok=True)
    csvfile=output/'training_fits.csv';js=output/'training_compute_summary.json'
    check(not csvfile.exists() and not js.exists(),'Refusing to overwrite existing outputs')
    with csvfile.open('x',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(fits[0]));w.writeheader();w.writerows(fits)
    with js.open('x',encoding='utf-8') as f:json.dump(summary,f,indent=2,allow_nan=False);f.write('\n')

def verify_runs(runs, records, checkpoint_root=None):
    """Check retained logs in place; an absent suite is reported, not imputed."""
    with (records/'training_fits.csv').open(newline='', encoding='utf-8') as f:
        fits = list(csv.DictReader(f))
    summary = json.loads((records/'training_compute_summary.json').read_text())
    keys = {(suite, f'fold{i:02d}', method, seed) for suite, methods in SUITES.items()
            for i in range(3) for method in methods for seed in SEEDS}
    actual = [(r['suite'],r['fold'],r['method'],int(r['seed'])) for r in fits]
    check(len(fits) == len(keys) and set(actual) == keys, 'Retained study fit keys differ')
    durations = [float(r['epoch_training_loop_seconds_sum']) for r in fits]
    epochs = sum(int(r['observed_epochs']) for r in fits)
    hours = math.fsum(durations) / 3600
    world = summary['environment']['world_size']
    check(summary['fits'] == len(fits) and epochs == summary['observed_epochs_per_fit']['sum']
          and math.isclose(hours, summary['total_training_loop_hours'], rel_tol=1e-12)
          and math.isclose(hours*world, summary['training_loop_nominal_allocated_npu_hours'], rel_tol=1e-12),
          'Retained training totals do not agree')
    check(runs.is_dir(), 'Runs directory missing: '+str(runs))
    available = {suite for suite in SUITES if (runs/suite).is_dir()}
    check(available, 'No original study suite under '+str(runs))
    checked = 0; observed = 0; verified_best = []; local_last = []; loop_seconds = []
    for r,(suite,fold,method,seed) in zip(fits,actual):
        if suite not in available:
            continue
        rel = Path(suite)/fold/(method+f'_s{seed}'); fit = runs/rel
        def read(name):
            return json.loads((fit/name).read_text(encoding='utf-8'))
        status, cal, held, init = [read(k+'.json') for k in ('status','calibration','holdout','initialization')]
        check(status['status'] == 'completed' and status['identity'] == cal['identity'] == r['fit_identity']
              and status['checkpoint_epoch'] == cal['checkpoint_epoch'] == int(r['selected_epoch_zero_based'])
              and cal['best_sha256'] == held['best_sha256'] == r['best_pt_sha256']
              and init['parameters_sha256'] == r['initialization_parameters_sha256'] and init['initialization_seed'] == seed,
              f'Fit identity/selection mismatch: {rel}')
        logs = [json.loads(q.read_text(encoding='utf-8')) for q in sorted(fit.glob('epoch_*.json'))]
        n = int(r['observed_epochs']); seconds = [e['seconds'] for e in logs]
        check([e['epoch'] for e in logs] == list(range(n)) and all(math.isfinite(v) and v>0 for v in seconds)
              and math.isclose(math.fsum(seconds), float(r['epoch_training_loop_seconds_sum']), rel_tol=1e-12),
              f'Epoch counts or durations differ: {rel}')
        if checkpoint_root is not None:
            best = checkpoint_root/rel/'best.pt'; last = checkpoint_root/rel/'last.pt'
            if best.is_file():
                check(digest(best) == cal['best_sha256'], f'Selected checkpoint hash differs: {rel}')
                verified_best.append(best.stat().st_size)
            if last.is_file(): local_last.append(last.stat().st_size)
        checked += 1; observed += n; loop_seconds.extend(seconds)
    return dict(status='matched_all_study_fits' if checked==len(fits) else 'matched_available_suites_only',
                retained_fits=len(fits), retained_epochs=epochs, retained_loop_hours=hours,
                retained_allocated_device_hours=hours*world, directly_checked_fits=checked,
                directly_checked_epochs=observed, directly_checked_loop_hours=math.fsum(loop_seconds)/3600,
                checked_suites=sorted(available), absent_suites=sorted(set(SUITES)-available),
                checkpoint_directory_supplied=checkpoint_root is not None,
                hash_verified_best_files=len(verified_best), hash_verified_best_bytes=sum(verified_best),
                local_unverified_last_files=len(local_last), local_unverified_last_bytes=sum(local_last))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    inputs=p.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--reports',type=Path,help='Directory with both original report ZIPs')
    inputs.add_argument('--verify-runs',type=Path,help='Check retained fit logs under the runs directory in place')
    p.add_argument('--records',type=Path,default=Path('result/training_records'),help='Retained fit CSV and summary (verification mode)')
    p.add_argument('--output',type=Path,help='New training tables directory (report extraction only)')
    p.add_argument('--checkpoint-root',type=Path,help='Optional runs/ directory for checkpoint byte measurements')
    a=p.parse_args()
    if a.verify_runs is not None:
        if a.output is not None:p.error('--verify-runs is read-only and prints JSON; omit --output')
        print(json.dumps(verify_runs(a.verify_runs,a.records,a.checkpoint_root),indent=2,allow_nan=False))
    else:
        if a.output is None:p.error('--reports requires --output')
        extract(a.reports,a.output,a.checkpoint_root)
