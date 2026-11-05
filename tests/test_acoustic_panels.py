"""Audio panels must preserve the frequency-by-time orientation and shared scale."""
import json
import hashlib
import shutil
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import plot
import acoustic_panels as panels
from acoustic_examples import load_examples


class AcousticPanelTests(unittest.TestCase):
    def test_measured_mesh_uses_correct_label_grid_common_scale_and_fixed_stat_size(self):
        ex=load_examples(ROOT/'assets/acoustic_examples')
        norms=[]
        for number, labels in ((1,('KW','HW','AB')), (2,())):
            fig=plot.plt.figure(figsize=(6.25,4)); old=fig.add_axes([.1,.2,.5,.5])
            _,_,width,height=old.get_window_extent().bounds
            try:
                panels.add_band(fig,number,ex)
                np.testing.assert_allclose(old.get_window_extent().bounds[2:],[width,height])
                axes={a.get_label():a for a in fig.axes}
                if number==1:
                    self.assertEqual([a.get_label() for a in fig.axes if a.get_label().startswith('signal_')],
                                     ['signal_KW','signal_HW','signal_AB'])
                for label in labels:
                    mesh,=axes['signal_'+label].collections
                    np.testing.assert_array_equal(mesh.get_array(),ex['labels'][label]['relative_power_db'])
                    self.assertEqual(axes['signal_'+label].get_yscale(),'log')
                    self.assertEqual(axes['signal_'+label].get_xlim(),(0,5))
                    if 'annotation_start_s' in ex['labels'][label]:
                        lines=[line for line in axes['signal_'+label].lines if line.get_gid()==label+'_annotation_span']
                        self.assertEqual(len(lines),1)
                        expected_span=[ex['labels'][label][k] for k in ('annotation_start_s','annotation_stop_s')]
                        np.testing.assert_allclose(lines[0].get_xdata(),expected_span)
                        self.assertGreater(lines[0].get_ydata()[0],1.)
                    norms.append(mesh.norm)
                if number==2:
                    self.assertNotIn('sound_power_color_scale',axes)
                    profile,=axes['signal_spectrum'].lines
                    np.testing.assert_allclose(profile.get_xdata(),ex['frequency_hz']/1000.)
                    # Evaluate in natural log units as an independent raw-feature reference.
                    source=ex['labels']['KW']['relative_power_db'].astype(float)
                    expected=np.log(np.exp(source*np.log(10.)/10.).mean(axis=1))*10./np.log(10.)
                    np.testing.assert_allclose(profile.get_ydata(),expected)
                    self.assertGreater(np.max(np.abs(profile.get_ydata()-source.mean(axis=1))),1.)
                    self.assertLess(float(profile.get_ydata().max()),0.)
                    self.assertGreaterEqual(float(profile.get_ydata().min()),axes['signal_spectrum'].get_ylim()[0])
                    self.assertLessEqual(float(profile.get_ydata().max()),axes['signal_spectrum'].get_ylim()[1])
                    path=axes['signal_wave'].collections[0].get_paths()[0]
                    self.assertEqual(path.vertices[:,1].max(),ex['labels']['KW']['envelope_max'].max())
                    self.assertEqual(path.vertices[:,1].min(),ex['labels']['KW']['envelope_min'].min())
            finally:
                plot.plt.close(fig)
        self.assertTrue(all(norm is norms[0] for norm in norms))

    def test_freshly_extracted_features_take_priority_and_corrupt_ones_are_not_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'run';run.mkdir(); directory=root/'result/acoustic_examples'
            bundled=root/'assets/acoustic_examples'
            shutil.copytree(ROOT/'assets/acoustic_examples',bundled)
            digest=json.loads((bundled/'examples.json').read_text())['protocol_sha256']
            (run/'analysis.json').write_text(json.dumps({'used_file_sha256':{
                'primary:state/protocol/complete.json':digest}}))
            self.assertEqual(panels.context(root,None,False,run)[1]['asset_directory'],'assets/acoustic_examples')
            shutil.copytree(bundled,directory)
            self.assertEqual(panels.context(root,None,False,run)[1]['asset_directory'],'result/acoustic_examples')
            (directory/'features.npz').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'SHA-256'):
                panels.context(root,None,False,run)
            self.assertEqual(panels.context(root,bundled,False,run)[1]['asset_directory'],'assets/acoustic_examples')
            self.assertIsNone(panels.context(root,None,True,run)[0])

    def test_recognized_old_cache_and_unknown_or_redirected_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'run';run.mkdir();directory=root/'result/acoustic_examples'
            bundled=root/'assets/acoustic_examples'
            shutil.copytree(ROOT/'assets/acoustic_examples',bundled)
            digest=json.loads((bundled/'examples.json').read_text())['protocol_sha256']
            (run/'analysis.json').write_text(json.dumps({'used_file_sha256':{
                'primary:state/protocol/complete.json':digest}}))
            directory.mkdir(parents=True)
            (directory/'examples.json').write_text('{}')
            (directory/'features.npz').write_bytes(b'earlier feature fixture')
            old={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir()}
            with patch('acoustic_examples.PREVIOUS',old):
                _,info=panels.context(root,None,False,run)
                self.assertTrue(info['previous_generated_bundle_skipped'])
                self.assertEqual(info['asset_directory'],'assets/acoustic_examples')
                with self.assertRaises((KeyError,ValueError)):
                    panels.context(root,directory,False,run)
                (directory/'features.npz').write_bytes(b'changed local copy')
                with self.assertRaises((KeyError,ValueError)):
                    panels.context(root,None,False,run)
            shutil.rmtree(directory);directory.symlink_to(bundled,target_is_directory=True)
            with self.assertRaisesRegex(ValueError,'Redirected'):
                panels.context(root,None,False,run)

    def test_examples_from_another_protocol_are_not_attached(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp)
            (run/'analysis.json').write_text(json.dumps({'used_file_sha256':{
                'primary:state/protocol/complete.json':'0'*64}}))
            self.assertIsNone(panels.context(ROOT,None,False,run)[0])
            with self.assertRaisesRegex(ValueError,'different run protocol'):
                panels.context(ROOT,ROOT/'assets/acoustic_examples',False,run)

if __name__=='__main__': unittest.main()
