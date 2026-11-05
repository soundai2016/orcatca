"""Check feature integrity and annotation identity without raw audio."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import io
import wave
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import acoustic_examples as ae


class AcousticExamplesTests(unittest.TestCase):
    def setUp(self):
        self.assets = ROOT/'assets/acoustic_examples'

    def test_common_reference_preserves_between_window_contrast(self):
        ex = ae.load_examples(self.assets)
        peaks = {k: float(v['relative_power_db'].max()) for k,v in ex['labels'].items()}
        self.assertEqual(max(peaks.values()),0.)
        self.assertLess(min(peaks.values()),0.)
        # Non-overlapping native sample intervals on a single recorder clock.
        windows=sorted((v['start_frame'],v['stop_frame_exclusive']) for v in ex['labels'].values())
        self.assertTrue(all(b-a==5*ex['native_rate_hz'] for a,b in windows))
        self.assertTrue(all(one[1]<=two[0] for one,two in zip(windows,windows[1:])))
        self.assertEqual((ex['frequency_hz'][0],ex['frequency_hz'][-1]), (218.75,7500.))
        self.assertAlmostEqual(ex['time_s'][0], 0.016)
        self.assertAlmostEqual(ex['time_s'][-1], 4.976)
        self.assertTrue(np.allclose(np.diff(ex['time_s']), .01))

    def test_rejects_mutated_feature_and_wrong_location_even_with_unchanged_scores(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'bundle';shutil.copytree(self.assets,p)
            j=p/'examples.json'; original=j.read_bytes(); d=json.loads(original)
            d['labels']['HW']['start_frame']+=1;j.write_text(json.dumps(d))
            with self.assertRaisesRegex(ValueError, 'window identity'): ae.load_examples(p)
            j.write_bytes(original)
            with (p/'features.npz').open('ab') as stream: stream.write(b'changed')
            with self.assertRaisesRegex(ValueError,'SHA-256 mismatch'): ae.load_examples(p)

    def test_writes_never_target_data_or_protocol(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            # A missing recording must not trigger a simulated substitute.
            target=root/'result/features'
            with self.assertRaises(FileNotFoundError): ae.extract(root,target)
            self.assertFalse(target.exists())
            for path in (root/ae.RAW/'figures',root/'state'/'protocol'/'out',root/'runs'/'out'):
                with self.subTest(path=path), self.assertRaisesRegex(ValueError,'outside inputs'):
                    ae.extract(root,path)
                self.assertFalse(path.exists())

    def test_pristine_prior_pair_is_backed_up_and_modified_pair_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);old=root/'acoustic_examples';old.mkdir()
            for n in ('examples.json','features.npz'):(old/n).write_bytes(n.encode())
            with patch.object(ae,'PREVIOUS',{p.name:ae.sha(p) for p in old.iterdir()}):
                self.assertTrue(ae.is_previous_bundle(old))
                stage=root/'stage';shutil.copytree(self.assets,stage)
                (old/'features.npz').write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'changed during extraction'):
                    ae.install_features(stage,old,True)
                self.assertEqual((old/'features.npz').read_bytes(),b'changed')
                (old/'features.npz').write_bytes(b'features.npz')
                ae.install_features(stage,old,True)
                self.assertTrue(ae.is_previous_bundle(root/'acoustic_examples_previous'))
                self.assertEqual(ae.sha(old/'features.npz'),ae.sha(self.assets/'features.npz'))

    def test_listening_copy_common_gain_and_spacing(self):
        windows={k:np.full(80000,v) for k,v in [('KW',1.),('HW',.5),('AB',.25)]}
        data,gain=ae.encode_listening_copy(windows)
        self.assertEqual(gain,.8)
        with wave.open(io.BytesIO(data),'rb') as stream:
            self.assertEqual((stream.getframerate(),stream.getnframes()),(16000,256000))
            pcm=np.frombuffer(stream.readframes(256000),dtype='<i2')
        self.assertTrue(np.all(pcm[80000:88000]==0) and np.all(pcm[168000:176000]==0))
        self.assertLessEqual(abs(int(pcm[0])-4*int(pcm[176000])),2)
        self.assertIn(ae.LICENSE_URL.encode('ascii'),data)

    def test_quiet_bins_keep_relative_power_before_display_scaling(self):
        x=(1e-7*np.sin(2*np.pi*1000*np.arange(80000)/16000)).astype('float32')
        a=ae.linear_spectral_power(x);b=ae.linear_spectral_power(2*x)
        self.assertGreater(float(a.max()),0.)
        self.assertLess(float(b.max()),1e-12)  # Both would be identical with the absolute training floor.
        self.assertAlmostEqual(10*np.log10(float(b.max())/float(a.max())),6.0206,places=4)
