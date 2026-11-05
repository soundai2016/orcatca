"""Offline integrity tests for staging transfer results and explicitly selected checkpoints."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('study_release',ROOT/'scripts/huggingface.py')
hf=importlib.util.module_from_spec(spec);spec.loader.exec_module(hf)


class ReleaseTests(unittest.TestCase):
    def test_results_need_no_runs_or_score_archives(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)/'project';root.mkdir()
            for name,path in hf.study_files(ROOT).items():
                rel=path.relative_to(ROOT);dest=root/rel
                dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
            # Stale legacy renders never stand in for a missing flat figure.
            top=root/'result/Fig1.pdf';legacy=root/'result/figures/Fig1.pdf'
            legacy.parent.mkdir(parents=True);top.rename(legacy)
            with self.assertRaisesRegex(FileNotFoundError,'Required release input missing'):
                hf.study_files(root)
            legacy.rename(top)
            other=root/'result/analysis/transfer_zzzz';other.mkdir()
            (other/'crossfit_summary.csv').write_text('do not select me')
            for extra in ('prediction_inputs.zip','analysis_result.zip'):
                (root/hf.CANONICAL/extra).write_bytes(b'unstaged')
            stage=Path(t)/'release'
            count,includes_results,_=hf.prepare(root,stage,[],False);self.assertTrue(includes_results);self.assertEqual(count,0)
            names=hf.verify(stage)
            self.assertFalse(any(n.endswith(('.pt','.zip','.gz')) for n in names))
            self.assertEqual((stage/'analysis/crossfit_summary.csv').read_bytes(),(ROOT/hf.CANONICAL/'crossfit_summary.csv').read_bytes())
            # Attribution is calculated separately; stale CSVs must not pass.
            stale=root/'result/figure_data/derived/label_mix_sensitivity.csv';stale.write_text('wrong figure data')
            with self.assertRaisesRegex(ValueError,'does not match'): hf.prepare(root,Path(t)/'fail',[],False)
            self.assertFalse((Path(t)/'fail').exists())

    def test_stale_render_or_example_cannot_be_staged(self):
        # This test exercises digest binding; no sound or graphic data are synthesized.
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); figures = root/'figures'; assets = root/hf.EXAMPLE_DIR
            figures.mkdir(); assets.mkdir(parents=True)
            for n in hf.FIGURES: (figures/n).write_bytes(n.encode())
            for n in hf.EXAMPLE_FILES: (assets/n).write_bytes(n.encode())
            ev = dict(figure_sha256={n:hf.sha(figures/n) for n in hf.FIGURES},
                      acoustic_context=dict(mode='measured', asset_sha256={n:hf.sha(assets/n) for n in hf.EXAMPLE_FILES}))
            self.assertEqual(len(hf.figure_files(root,ev,figures)),6)
            # When rendering used a local extraction, stage those exact files
            # under the portable release asset path, independent of bundled copies.
            source_dir=root/'result/acoustic_examples'
            shutil.copytree(assets,source_dir)
            ev['acoustic_context']['asset_directory']='result/acoustic_examples'
            mapped=hf.figure_files(root,ev,figures)
            self.assertEqual(mapped['assets/acoustic_examples/features.npz'],source_dir/'features.npz')
            self.assertEqual(hf.figure_files(root,ev,figures,staged=True)['assets/acoustic_examples/features.npz'],assets/'features.npz')
            del ev['acoustic_context']['asset_directory']
            (assets/'features.npz').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'Acoustic input differs'):
                hf.figure_files(root,ev,figures)
            ev['acoustic_context']={'mode':'none'}
            self.assertEqual(set(hf.figure_files(root,ev,figures)),{'figures/'+n for n in hf.FIGURES})
            (figures/'Fig2.pdf').write_bytes(b'old render')
            with self.assertRaisesRegex(ValueError,'Rendered figure does not match'):
                hf.figure_files(root,ev,figures)

    def test_named_checkpoint_selection_is_preserved(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);base=root/'runs/demo/fold01'
            for seed in (17,29):
                fit=base/f'kw_s{seed}';fit.mkdir(parents=True)
                (fit/'best.pt').write_text(f'fixture-{seed}');(fit/'holdout.json').write_text('{}')
                (fit/'holdout.jsonl.gz').write_text('not part of the release')
            with self.assertRaisesRegex(FileNotFoundError,'Completed transfer'): hf.prepare(root,root/'none',[],False)
            for n,all_seeds in ((1,False),(2,True)):
                stage=root/f'stage{n}';weights,includes_results,_=hf.prepare(root,stage,['demo'],all_seeds)
                self.assertEqual((weights,includes_results),(n,False));self.assertEqual(sum(k.endswith('.pt') for k in hf.verify(stage)),n)
                self.assertFalse(any(stage.rglob('*.gz')))
            with self.assertRaisesRegex(ValueError,'Invalid suite'): hf.prepare(root,root/'bad',['../wrong'],False)

    def test_existing_stage_rejects_extra_changed_and_manifest_listed_score_files(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);fit=root/'runs/demo/fold00/kw_s17';fit.mkdir(parents=True)
            (fit/'best.pt').write_bytes(b'fixture1');out=root/'stage';hf.prepare(root,out,['demo'],False)
            extra=out/'prediction_inputs.zip';extra.write_bytes(b'not allowed')
            with self.assertRaisesRegex(ValueError,'extra files'): hf.verify(out)
            m=out/hf.MANIFEST;original=m.read_bytes();d=json.loads(original)
            d['sha256'][extra.name]=hashlib.sha256(extra.read_bytes()).hexdigest();m.write_text(json.dumps(d))
            with self.assertRaisesRegex(ValueError,'Unexpected release path'): hf.verify(out)
            extra.unlink();m.write_bytes(original);(out/'README.md').write_text('not the reviewed version')
            with self.assertRaisesRegex(ValueError,'checksum mismatch'): hf.verify(out)


if __name__=='__main__': unittest.main()
