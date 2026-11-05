"""A migrated control suite may use a relocated protocol, but never different input bytes."""
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from orcatca.analysis import run
from orcatca.result_store import CombinedStore, ReplayTrace, ResultStore
from orcatca.utils import atomic_json, sha256, write_rows


class MigratedStudy(unittest.TestCase):
    methods = ['binary', 'control']

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        protocol = self.root / 'state/protocol'
        protocol.mkdir(parents=True)
        rows = []
        for role, group in [('calibrate', 'source'), ('holdout', 'target')]:
            for i in range(8):
                rec = f'{group}{i}'
                for j, label in enumerate(('AB','HW','KW')):
                    rows.append(dict(id=rec+label, recording=rec, block=rec, group=group, site=group,
                                     audio=rec+'.wav', start_frame=80000*j, split=role, raw_class=label,
                                     label=int(label=='KW'), time_metadata_available=False, recording_utc=None,
                                     block_utc=None, recording_duration=15.))
        manifest = protocol/'fold00.jsonl.gz'
        write_rows(manifest, rows)
        complete = protocol/'complete.json'
        atomic_json(complete, dict(folds=[dict(manifest=manifest.name)], files={manifest.name:sha256(manifest)}))
        self.inputs = {}
        for role, method in [('reference','binary'),('primary','control')]:
            suite = self.root/'runs'/role
            suite.mkdir(parents=True)
            cfg=dict(name=role, data={}, cache={}, augmentation={}, runtime={},
                     experiment=dict(folds=1, seeds=[17], methods=[method]),
                     evaluation=dict(fpr_targets=[.2], min_negative_recordings=2,
                                     min_negative_blocks=2, required_negative_subtypes=['AB','HW']))
            if role == 'primary':
                cfg['input_reference']=dict(protocol_complete='state/v220/protocol/complete.json',
                    protocol_sha256=sha256(complete),
                    source_config_sha256=self.inputs['reference:runs/config.yaml'])
            # JSON is a valid YAML serialization.
            atomic_json(suite/'config.yaml',cfg)
            atomic_json(suite/'config.json',dict(config=cfg))
            fit=suite/'fold00'/f'{method}_s17'
            atomic_json(fit/'status.json',dict(status='completed'))
            atomic_json(fit/'calibration.json',dict(operating_thresholds={'fpr_0.2':.88}))
            for target, name in [('calibrate','calibration'),('holdout','holdout')]:
                write_rows(fit/(name+'.jsonl.gz'),[dict(r,p=((i*7)%19)/20) for i,r in enumerate(rows) if r['split']==target])
            prefix='result/runs/' if role=='primary' else 'runs/'
            files={prefix+f.relative_to(suite).as_posix():f for f in suite.rglob('*') if f.is_file()}
            files.update({'state/protocol/'+f.name:f for f in protocol.iterdir()})
            hashes={k:sha256(v) for k,v in files.items()}
            self.inputs.update({role+':'+k:v for k,v in hashes.items()})
            if role=='reference':
                with zipfile.ZipFile(self.root/'reference_selected.zip','w') as archive:
                    for k,v in files.items():archive.write(v,k)
                    archive.writestr('REPORT_SHA256.json',json.dumps(hashes))
                import shutil
                shutil.rmtree(suite)
        self.trace=dict(status='completed', methods=self.methods, seeds=[17], analysis_seed=23,
                        bootstrap_repeats=10, crossfit_repeats=1, used_file_sha256=self.inputs)
        atomic_json(self.root/'expected_trace.json',self.trace)

    def replay(self):
        return run(self.root, 'runs/primary', 'result/analysis/replay', self.methods, 10, 1, 23,
                   'reference_selected.zip', 'expected_trace.json')

    def test_mixed_inputs_replay_full_trace_after_relocation(self):
        archive = self.replay()
        meta = json.loads((archive.parent/'analysis.json').read_bytes())
        self.assertEqual(meta['expected_trace']['checked_members'], {'primary':8,'reference':8})
        self.assertEqual(meta['inputs']['primary']['kind'], 'suite_directory')
        self.assertEqual(meta['inputs']['primary']['protocol'], str(self.root/'state/protocol'))
        self.assertEqual(meta['inputs']['reference']['kind'], 'report')
        self.assertEqual(meta['folds'][0]['methods'],2)
        self.assertTrue((archive.parent/'transfer_gap_summary.csv').is_file())
        # The new location-independent trace can also verify a subsequent replay.
        trace=ReplayTrace(archive.parent/'analysis.json', self.methods, 10, 1, 23)
        original=ReplayTrace(self.root/'expected_trace.json', self.methods, 10, 1, 23)
        self.assertEqual(trace.roles, original.roles)

    def test_relocated_complete_requires_original_digest(self):
        p=self.root/'state/protocol/complete.json'
        p.write_bytes(p.read_bytes()+b' ')
        with self.assertRaisesRegex(ValueError,'Referenced protocol changed'):
            self.replay()
        self.assertFalse((self.root/'result/analysis/replay/analysis.json').exists())

    def test_present_but_changed_original_protocol_not_replaced_with_copy(self):
        p=self.root/'state/v220/protocol/complete.json'
        p.parent.mkdir(parents=True)
        p.write_text('{}')
        with ResultStore(self.root,'runs/primary') as primary:
            with self.assertRaisesRegex(ValueError,'Referenced protocol changed'):
                primary.protocol(0)

    def test_changed_score_is_rejected_even_if_all_ids_and_labels_match(self):
        p=self.root/'runs/primary/fold00/control_s17/holdout.jsonl.gz'
        first=self.inputs['primary:result/runs/fold00/control_s17/holdout.jsonl.gz']
        import gzip
        rows=[json.loads(line) for line in gzip.decompress(p.read_bytes()).splitlines()]
        rows[0]['p'] = .987
        write_rows(p,rows)
        self.assertNotEqual(sha256(p),first)
        with self.assertRaisesRegex(ValueError,'mismatched expected study input'):
            self.replay()

    def test_extra_expected_member_and_duplicate_alias_are_rejected(self):
        self.inputs['primary:result/runs/unused.json']='0'*64
        atomic_json(self.root/'expected_trace.json',self.trace)
        with self.assertRaisesRegex(ValueError,'were not consumed'):
            self.replay()
        self.assertFalse((self.root/'result/analysis/replay/analysis.json').exists())
        self.inputs['primary:runs/config.yaml']=self.inputs['primary:result/runs/config.yaml']
        atomic_json(self.root/'expected_trace.json',self.trace)
        with self.assertRaisesRegex(ValueError,'Duplicate expected study member'):
            ReplayTrace(self.root/'expected_trace.json', self.methods, 10, 1, 23)

    def test_summary_only_reference_and_absent_methods_are_rejected(self):
        p=self.root/'result/analysis/reference/holdouts/fold00/binary_s17/holdout.json'
        atomic_json(p,dict(operating={}))
        with self.assertRaisesRegex(ValueError,'cannot replace window scores'):
            CombinedStore(self.root,'runs/primary','result/analysis/reference')
        with self.assertRaisesRegex(ValueError,'Missing methods: binary'):
            run(self.root,'runs/primary','result/analysis/guarded',self.methods,10,1,23)
        self.assertFalse((self.root/'result/analysis/guarded').exists())

if __name__=='__main__':
    unittest.main()
