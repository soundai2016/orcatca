"""Synthetic integrity and recovery checks; no model or corpus needed."""
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from check_analysis_inputs import inspect
from run_layout import SUITE, check_run, compare_holdout, fits, layout
from study_reports import allowed


def put(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_bytes(gzip.compress(('\n'.join(map(json.dumps,value))+'\n').encode()) if p.suffix=='.gz' else json.dumps(value).encode())


class Inputs(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);self.root=Path(tmp.name)
        self.store=self.root/'runs/suite';self.methods=['a','b'];self.protocol=self.root/'state/protocol'
        cfg=dict(name='suite',experiment=dict(folds=1,methods=self.methods,seeds=[0]),evaluation=dict(fpr_targets=[.01]))
        put(self.store/'config.json',dict(config=cfg));(self.store/'config.yaml').write_text(yaml.safe_dump(cfg))
        roles={}
        for role in ('calibrate','holdout'):
            roles[role]=[dict(id=role+cl,raw_class=cl,label=int(cl=='KW'),split=role,group=role,site=role,recording=role+cl,block=role+cl,audio=role+cl+'.wav',start_frame=0) for cl in ('AB','HW','KW')]
        rows=roles['calibrate']+roles['holdout']; m=self.protocol/'fold.gz';put(m,rows)
        put(self.protocol/'complete.json',dict(folds=[dict(manifest=m.name)],files={m.name:hashlib.sha256(m.read_bytes()).hexdigest()}))
        for method in self.methods:
            run=self.store/f'fold00/{method}_s0';self.runpath=run
            put(run/'status.json',dict(status='completed'));put(run/'calibration.json',dict(operating_thresholds={'fpr_0.01':.9}))
            put(run/'holdout.json',dict(operating={'fpr_0.01':{'overall':{'recall_kw':.5}}}))
            for role,filename in [('calibrate','calibration'),('holdout','holdout')]:put(run/(filename+'.jsonl.gz'),[dict(r,p=.5) for r in roles[role]])
    def check(self,src='runs/suite',verify=True):return inspect(self.root,src,methods=self.methods,verify=verify)
    def test_verified_directory_and_report(self):
        self.assertEqual(self.check()['status'],'inputs_verified');hashes={}
        for prefix in ('runs/','result/runs/'):
            hashes={}
            with zipfile.ZipFile(self.root/'report.zip','w') as z:
                for directory,location in [(self.store,prefix),(self.protocol,'state/protocol/')]:
                    for p in directory.rglob('*'):
                        if p.is_file():
                            n=location+p.relative_to(directory).as_posix();z.write(p,n);hashes[n]=hashlib.sha256(p.read_bytes()).hexdigest()
                z.writestr('REPORT_SHA256.json',json.dumps(hashes))
            self.assertEqual(self.check('report.zip')['status'],'inputs_verified')
        with zipfile.ZipFile(self.root/'report.zip','a') as z:z.write(self.store/'config.yaml','runs/config.yaml')
        with self.assertRaises(ValueError):self.check('report.zip')
    def test_partial_predictions_still_allow_recall(self):
        (self.runpath/'calibration.jsonl.gz').unlink()
        report=self.check();self.assertEqual(report['status'],'incomplete_or_invalid');r=report['fits'][1]
        self.assertIn('calibration.jsonl.gz',r['issues'][0]);self.assertEqual(r['archived_recall']['fpr_0.01']['value'],.5)
    def test_quick_does_not_claim_verification(self):
        p=self.runpath/'holdout.jsonl.gz'; rows=[json.loads(x) for x in gzip.decompress(p.read_bytes()).splitlines()];rows[0]['group']='wrong';put(p,rows)
        self.assertEqual(self.check(verify=False)['status'],'inputs_present_unverified')
        self.assertIn('mismatch',self.check()['fits'][1]['issues'][0])


class MigratedLayout(unittest.TestCase):
    def test_aggregate_cache_and_request_are_not_prediction_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for f, m, s in fits('primary'):
                fit = root/'runs'/SUITE['primary']/f/f'{m}_s{s}'
                for name in ('holdout.jsonl.gz','calibration.jsonl.gz'):
                    put(fit/name, [{'p':.8}])
            for f, m, s in fits('reference'):
                put(root/'result/analysis/reference/holdouts'/f/f'{m}_s{s}'/'holdout.json',{})
            put(root/'state/requests'/(SUITE['reference']+'.yaml'), {})
            put(root/'result/analysis/reanalysis/failure.json', {'exception':'FileExistsError','message':'test'})
            got = layout(root)
            self.assertEqual(got['runs']['primary']['window_score_pairs'],18)
            self.assertEqual(got['runs']['reference']['window_score_pairs'],0)
            self.assertEqual(got['summary_files']['reference_holdout_cache'],18)
            self.assertIn('Missing window inputs in inspected locations: reference.',got['full_replay'])
            self.assertEqual(got['previous_failed_analysis']['exception'],'FileExistsError')
            self.assertFalse(got['figure_inputs_present'])
            self.assertFalse(got['rendered_files_present'])
            self.assertEqual(len(got['missing_rendered_files']),4)

    def test_moved_primary_must_match_every_trace_member(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root/'runs'/SUITE['primary']
            cfg = {'name': SUITE['primary']}
            trace = {}
            for key in allowed('primary','result/runs/'):
                path = (root/key if key.startswith('state/') else run/key[len('result/runs/'):])
                data = (json.dumps(cfg).encode() if key.endswith('config.yaml') else
                        json.dumps({'config':cfg}).encode() if key.endswith('config.json') else b'archived')
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(data)
                trace['primary:'+key] = hashlib.sha256(data).hexdigest()
            self.assertEqual(check_run(root,'primary',run,trace),78)
            changed = next(run.glob('fold00/*/holdout.jsonl.gz'))
            changed.write_bytes(b'different score bytes')
            with self.assertRaisesRegex(ValueError,'Saved member differs'):
                check_run(root,'primary',run,trace)

    def test_holdout_must_match_model_counts_cutoffs_and_rates(self):
        actual = dict(threshold=.8, fp=2, tn=6, tp=3, fn=2, false_positive_rate=.25, recall_kw=.6)
        point = dict(procedure='full_calibration',status='estimated',threshold='.8',target_fp='2',target_tp='3',
                     target_fpr='.25',target_recall='.6',target_negative='8',target_positive='5')
        rates = {a: dict(point) for a in ('0.01', '0.05')}
        saved = dict(manifest_sha256='p',best_sha256='w',metrics={'macro_pauc':.74},
                     operating={f'fpr_{a}':{'overall':dict(actual)} for a in rates})
        compare_holdout(saved,rates,{'macro_pauc':.74},'p','w')
        with self.assertRaisesRegex(ValueError,'checkpoint identity'):
            compare_holdout(saved,rates,{'macro_pauc':.74},'p','other')
        for field, wrong in [('threshold','.81'),('target_negative','9'),('target_recall','.61')]:
            altered = {a: dict(v) for a,v in rates.items()}
            altered['0.05'][field] = wrong
            with self.subTest(field=field), self.assertRaises(ValueError):
                compare_holdout(saved,altered,{'macro_pauc':.74},'p','w')
