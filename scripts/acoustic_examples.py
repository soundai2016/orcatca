#!/usr/bin/env python3
"""Extract display features from three annotated training windows in one recording.

Run from the project root: python scripts/acoustic_examples.py
The original recording and protocol files are read only. No classifier scores or weights are used.
"""
from __future__ import annotations
import argparse
import csv
import io
import wave
import gzip
import hashlib
import json
from pathlib import Path
import sys
import struct
from tempfile import TemporaryDirectory

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = 'dclde_2027_killer_whales'
RECORDING = 'simres/audio/eastpoint/ICLISTENHF6156_20220711T152500.000Z_20220711T152959.996Z.flac'
AUDIO_SHA = '98ea242436a24492089c566dc7800bb1e6f18a00a5e89df711700525d56f8965'
PROTOCOL_SHA = '73f074bfba2be47cfc953a38c4ab5d8800742f4ec7d973261bfcd42624b47921'
ANNOTATIONS_SHA = '407f1c0f7d95a39c7bace99bbef40646e2fe62403c6237c20ddf374e88e21b15'
FOLD_SHA = 'fa10227985bd98154803df52defcf059e53fb48514276b167effb7ffb55545a8'
RECORDS_SHA = '6aa18ce74dd4937ef55e1ffd2b4005973e6d74e7cbd4956fd738f6e014fe2b4e'
IDENTITY = {'KW': ('05c49aacce0f41b57b25d51a', 880518, 178077),
            'HW': ('6676d6d03998bc5aa227751d', 31036826, 178158),
            'AB': ('401d7a8ec7623b07f86defb3', 18559373, 178110)}
CFG = dict(seconds=5., sample_rate=16000, n_fft=512, hop=160, min_hz=200., max_hz=7500.)
SCHEMA = 'orcatca-acoustic-windows-2'
SITE = 'SIMRES::Tekteksen'
CONTRIBUTOR = 'SIMRES'
GEOMETRY = (128000,38400000,38519910)  # native Hz, sample frames, encoded bytes
BOXES = {'KW':(9.0092,9.7489,309.2,309.2), 'HW':(244.5669,245.3835,276.657,276.657),
         'AB':(144.9943,149.9959,0.,0.)}
SELECTION = ('The search used the three smallest encoded fold00 training recordings containing KW, HW and AB. '
             'For each annotation with at least 20 flank frames (0.1 s guards), subtract the per-frequency '
             'median flank log power, take the frequency maximum (0.3–6 kHz) for each annotated frame, '
             'then take the median across those frames. Select the highest-contrast eligible KW; use the '
             'highest-contrast HW and earliest nonoverlapping AB from that same file. '
             'Ranking used the training frontend log-power floor. Illustrative, not a species-average signature; '
             'no classifier scores are used.')
CREDIT = 'Saturna Island Marine Research & Education Society (SIMRES), DCLDE 2027, NOAA NCEI, doi:10.25921/15ey-mh50'
LICENSE = 'CC BY-SA 4.0'
LICENSE_URL = 'https://creativecommons.org/licenses/by-sa/4.0/'
DISPLAY_POLICY = 'Window-normalized linear FFT power, referenced to the three-window maximum before the relative display floor; the absolute training floor is not applied.'
POWER_SCALE = 'dB relative to the largest in-band time-frequency bin across all three windows; not SPL'
WAVE_SCALE = 'signed digital amplitude relative to each window maximum absolute sample'
POWER_STEP = 0.5
CLIP_DB = -120.
ENV_BLOCK = 320   # 20 ms min/max bins
PREVIOUS = {'examples.json':'a04d56735bf134d6b7ba6f883a38875697fc234a8ee233c5adbebb600e9946dc',
            'features.npz':'c59dc184a212dd920dfe31e282be30d23eac80764e299868ab03f8de68871b5b'}


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def require(check, msg):
    if not check:
        raise ValueError(msg)


def is_previous_bundle(directory):
    """Recognize only the unmodified, previously distributed pair of files."""
    d=Path(directory)
    return (d.is_dir() and d.resolve()==d.absolute() and
            {p.name for p in d.iterdir()}==set(PREVIOUS) and
            all((d/n).is_file() and not (d/n).is_symlink() and sha(d/n)==h for n,h in PREVIOUS.items()))


def install_features(stage, output, previous):
    """Commit a complete verified pair. Unrelated local data are never replaced."""
    if previous:
        backup=output.with_name(output.name+'_previous')
        require(is_previous_bundle(output), 'Earlier features changed during extraction')
        require(not backup.exists() and not backup.is_symlink(), 'Backup path exists: '+str(backup))
        output.rename(backup)
        try:
            stage.rename(output)
        except BaseException:
            if not output.exists() and not output.is_symlink(): backup.rename(output)
            raise
        print('Preserved earlier verified display features in',backup)
    elif output.exists():
        if (output.is_symlink() or {p.name for p in output.iterdir()}!={'features.npz','examples.json'} or
                any((output/n).is_symlink() or (stage/n).read_bytes()!=(output/n).read_bytes()
                    for n in ('features.npz','examples.json'))):
            raise FileExistsError(f'Existing acoustic files differ from the new extraction: {output}')
    else:
        stage.rename(output)


def axes():
    hz = np.fft.rfftfreq(CFG['n_fft'], 1 / CFG['sample_rate'])
    bins = hz[(hz >= CFG['min_hz']) & (hz <= CFG['max_hz'])]
    n = int(CFG['seconds'] * CFG['sample_rate'])
    times = (np.arange(0, n-CFG['n_fft']+1, CFG['hop'])+CFG['n_fft']/2) / CFG['sample_rate']
    env_t = (np.arange(0, n, ENV_BLOCK)+ENV_BLOCK/2) / CFG['sample_rate']
    return bins, times, env_t


def source_rows(root):
    proto = root / 'state' / 'protocol'
    require(sha(proto/'complete.json') == PROTOCOL_SHA, 'Frozen protocol SHA-256 mismatch')
    complete = json.loads((proto/'complete.json').read_bytes())
    require(complete['files']['fold00.jsonl.gz'] == FOLD_SHA and
            sha(proto/'fold00.jsonl.gz') == FOLD_SHA, 'Fold manifest SHA-256 mismatch')
    require(complete['files']['recordings.json'] == RECORDS_SHA and
            sha(proto/'recordings.json') == RECORDS_SHA, 'Recording inventory SHA-256 mismatch')
    record = json.loads((proto/'recordings.json').read_bytes())[RECORDING]
    require((record['sha256'],record['channels'],record['rate'],record['frames'],record['size']) ==
            (AUDIO_SHA, 1, *GEOMETRY), 'Recording identity or native geometry differs')
    rows = {}
    chosen = {key: (label,start,line) for label,(key,start,line) in IDENTITY.items()}
    with gzip.open(proto/'fold00.jsonl.gz','rt',encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            if r['id'] not in chosen:
                continue
            label,start,annotation = chosen[r['id']]
            require(label not in rows and r['split'] == 'train' and r['recording'] == AUDIO_SHA and
                    r['audio'] == RECORDING and r['site'] == SITE and r['provider'] == CONTRIBUTOR and
                    (r['raw_class'],r['start_frame'],r['annotation_line'],r['label']) ==
                    (label,start,annotation,int(label=='KW')), 'Window identity or role differs')
            rows[label] = r
    require(rows.keys() == IDENTITY.keys(), 'Training window missing from frozen fold')
    annotations=root/RAW/'Annotations.csv'
    require(complete.get('annotations_sha256')==ANNOTATIONS_SHA and sha(annotations)==ANNOTATIONS_SHA,
            'Aggregate annotation table SHA-256 mismatch')
    wanted={r['annotation_line']:label for label,r in rows.items()}; seen={}
    with annotations.open(encoding='utf-8-sig',newline='') as stream:
        for line,r in enumerate(csv.DictReader(stream),2):
            if line not in wanted:continue
            label=wanted[line]
            def number(v): return None if v.strip().upper() in ('','NA','NAN') else float(v)
            box=tuple(number(r[k]) for k in ('FileBeginSec','FileEndSec','LowFreqHz','HighFreqHz'))
            require(r['Soundfile']==Path(RECORDING).name and r['ClassSpecies'].upper()==label and
                    r['Provider']==CONTRIBUTOR and (r['KW']=='1')==(label=='KW') and
                    (label!='KW' or r['KW_certain']=='1') and box==BOXES[label],
                    'Archived annotation bounds or original class differ: '+label)
            seen[label]=line
            if len(seen)==len(wanted):break
    require(set(seen)==set(rows),'Missing archived annotation extent')
    return record, rows


def annotation_box(label,start):
    start_s=start/GEOMETRY[0]; a,b,low,high=BOXES[label]
    lo=max(0.,a-start_s);hi=min(CFG['seconds'],b-start_s)
    require(np.isfinite([lo,hi]).all() and 0<=lo<hi<=CFG['seconds'],'Annotation outside display window')
    return dict(source_annotation_start_s=a,source_annotation_stop_s=b,annotation_start_s=round(lo,9),
                annotation_stop_s=round(hi,9),source_annotation_low_hz=low,source_annotation_high_hz=high)


def linear_spectral_power(x):
    """Training FFT geometry, preserving measured power before display scaling."""
    from scipy.fft import rfft
    n,hop = CFG['n_fft'],CFG['hop']
    taper=np.hanning(n+1)[:-1].astype(np.float32)
    frames=np.lib.stride_tricks.sliding_window_view(x,n)[::hop]
    spec=rfft(frames*taper,axis=1,workers=1)
    bins=np.fft.rfftfreq(n,1/CFG['sample_rate'])
    keep=(bins>=CFG['min_hz'])&(bins<=CFG['max_hz'])
    power=(spec.real**2+spec.imag**2)/float(taper.sum()**2)
    return np.ascontiguousarray(power[:,keep].T)


def encode_listening_copy(windows):
    """KW, HW, AB; half-second silence between windows and one common playback gain."""
    order=('KW','HW','AB');arrays=[np.asarray(windows[k],dtype=np.float64) for k in order]
    require(all(w.shape==(80000,) and np.isfinite(w).all() for w in arrays),'Invalid listening window')
    peak=max(float(np.max(np.abs(w))) for w in arrays);require(peak>0.,'Silent listening windows')
    gain=0.8/peak; blank=np.zeros(8000,dtype=np.float64)
    samples=np.concatenate([arrays[0],blank,arrays[1],blank,arrays[2]])
    pcm=np.rint(samples*gain*32767.).astype('<i2')
    buf=io.BytesIO()
    with wave.open(buf,'wb') as f:
        f.setnchannels(1);f.setsampwidth(2);f.setframerate(16000);f.writeframes(pcm.tobytes())
    def chunk(key, data):
        return key + struct.pack('<I',len(data)) + data + (b'\0' if len(data)%2 else b'')
    tags={b'INAM':'Annotated killer whale, humpback whale and background examples',b'IART':CREDIT,
          b'ICOP':LICENSE+'; '+LICENSE_URL,
          b'ICMT':f'Three 5-s windows: KW at 0 s, HW at 5.5 s, AB at 11 s. Resampled to 16 kHz with a common playback gain {gain:.8g}; not SPL.'}
    info=b'INFO'+b''.join(chunk(k,v.encode('ascii')+b'\0') for k,v in tags.items())
    data=buf.getvalue()+chunk(b'LIST',info)
    return data[:4]+struct.pack('<I',len(data)-8)+data[8:],gain


def save_listening_copy(root,windows):
    path=root/'result/acoustic_comparison.wav'
    require(path.resolve()==path,'Redirected listening-copy output')
    data,gain=encode_listening_copy(windows)
    if path.exists():
        require(path.is_file() and path.read_bytes()==data,'Listening-copy output already differs: '+str(path))
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as stream:stream.write(data)
    print(f'Listening copy: {path} (KW, HW, AB; common playback gain {gain:.8g})')


def extract(root, output, listening_copy=False):
    root = Path(root).resolve(strict=True)
    output = Path(output).absolute()
    require(output.resolve()==output, 'Redirected acoustic output path')
    protected = [root/k for k in (RAW,'state','runs')]
    require(all(not output.is_relative_to(p.resolve()) and not p.resolve().is_relative_to(output)
                for p in protected), 'Choose a new output directory outside inputs')
    previous = output==root/'result/acoustic_examples' and is_previous_bundle(output)
    if output.exists() and not previous:
        require(output.is_dir() and {p.name for p in output.iterdir()}=={'features.npz','examples.json'},
                'Existing acoustic directory is incomplete or contains unrelated files: '+str(output))
        load_examples(output)
    record, rows = source_rows(root)
    # Extraction shares the training frontend; reading the small bundle needs NumPy only.
    sys.path.insert(0, str(ROOT/'src'))
    from orcatca.audio import load_wave
    from orcatca.utils import raw_path
    flac = raw_path(root, RECORDING); st = flac.stat()
    require(st.st_size == record['size'] and sha(flac) == AUDIO_SHA,
            'Source recording bytes differ from the frozen recording')
    # A restored copy can have a new mtime; full byte identity was checked above.
    record = dict(record, mtime_ns=st.st_mtime_ns)
    arrays = {}; power = {}; detail = {}; listening = {}
    for label,(sid,start,line) in IDENTITY.items():
        x = load_wave(root,rows[label],record,CFG); peak = float(np.max(np.abs(x)))
        require(peak>0 and np.isfinite(peak),'Silent or invalid training window: '+label)
        if listening_copy: listening[label]=x
        power[label] = linear_spectral_power(x)
        scaled = (x / peak).reshape(-1, ENV_BLOCK)
        arrays[label+'_envelope_min'] = scaled.min(axis=1).astype('<f4')
        arrays[label+'_envelope_max'] = scaled.max(axis=1).astype('<f4')
        detail[label] = dict(id=sid,start_frame=start,stop_frame_exclusive=start+5*record['rate'],
                             annotation_line=line,digital_peak=peak,**annotation_box(label,start))
    common = max(float(m.max()) for m in power.values())
    require(common>0 and np.isfinite(common),'Empty or invalid in-band energy')
    for key,m in power.items():
        relative_db=10*np.log10(np.maximum(m.astype(np.float64)/common,10.**(CLIP_DB/10)))
        arrays[key+'_attenuation'] = np.rint(np.clip(-relative_db,0.,-CLIP_DB)/POWER_STEP).astype('uint8')
    info = dict(schema=SCHEMA,recording=RECORDING,source_sha256=AUDIO_SHA,
                data_doi='10.25921/15ey-mh50',source_credit=CREDIT,data_license=LICENSE,data_license_url=LICENSE_URL,
                source_url='https://storage.googleapis.com/noaa-passive-bioacoustic/dclde/2027/'+RAW+'/'+RECORDING,
                contributor=CONTRIBUTOR,site=SITE,fold='fold00',split='train',
                protocol_sha256=PROTOCOL_SHA,fold_sha256=FOLD_SHA,recordings_sha256=RECORDS_SHA,
                selection=SELECTION,annotations_sha256=ANNOTATIONS_SHA,
                native_rate_hz=record['rate'],native_frames=record['frames'],audio_bytes=record['size'],
                duration_s=CFG['seconds'],labels=detail,frontend=CFG,
                window='periodic Hann, no centering or padding',power_scale=POWER_SCALE,
                reference_feature_db=float(10*np.log10(common)),display_power_policy=DISPLAY_POLICY,relative_floor_db=CLIP_DB,display_step_db=POWER_STEP,
                envelope_scale=WAVE_SCALE,envelope_bin_samples=ENV_BLOCK)
    output.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(prefix='.acoustic-',dir=output.parent) as tmp:
        stage=Path(tmp)/'bundle'; stage.mkdir()
        np.savez_compressed(stage/'features.npz',**arrays)
        info['features_sha256']=sha(stage/'features.npz')
        (stage/'examples.json').write_text(json.dumps(info,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        load_examples(stage)
        install_features(stage,output,previous)
    if listening_copy: save_listening_copy(root,listening)
    return info


def load_examples(directory):
    """Return verified digital envelopes and relative spectral power; never read audio."""
    d=Path(directory); j=d/'examples.json'; z=d/'features.npz'
    require(not j.is_symlink() and not z.is_symlink() and not d.is_symlink(), 'Redirected acoustic bundle member')
    info=json.loads(j.read_text(encoding='utf-8'))
    require((info['schema'],info['source_sha256'],info['recording'],info['fold'],info['split'],
             info['protocol_sha256'],info['fold_sha256'],info['recordings_sha256'],info['site'])==
            (SCHEMA,AUDIO_SHA,RECORDING,'fold00','train',PROTOCOL_SHA,FOLD_SHA,RECORDS_SHA,SITE),
            'Wrong source or fold for acoustic examples')
    require((info['source_credit'],info['data_license'],info['data_license_url'])==(CREDIT,LICENSE,LICENSE_URL), 'Source credit or license differs')
    require(info.get('annotations_sha256')==ANNOTATIONS_SHA and info['contributor']==CONTRIBUTOR, 'Annotation source differs')
    require(info['frontend']==CFG and info['display_power_policy']==DISPLAY_POLICY and info['power_scale']==POWER_SCALE and info['envelope_scale']==WAVE_SCALE
            and (info['display_step_db'],info['relative_floor_db'],info['envelope_bin_samples'])==(POWER_STEP,CLIP_DB,ENV_BLOCK)
            and (info['native_rate_hz'],info['native_frames'],info['audio_bytes'])==GEOMETRY and info['duration_s']==5., 'Acoustic scale or geometry differs')
    require(sha(z)==info['features_sha256'],'Acoustic feature SHA-256 mismatch')
    require(set(info['labels'])==set(IDENTITY),'Unexpected acoustic labels')
    hz,t,env_t = axes(); labels={}
    with np.load(z,allow_pickle=False) as data:
        keys={label+'_'+field for label in IDENTITY for field in ('attenuation','envelope_min','envelope_max')}
        require(len(data.files)==len(keys) and set(data.files)==keys,'Unexpected acoustic features')
        maximum = []
        for label,(sid,start,line) in IDENTITY.items():
            row = info['labels'][label]
            require((row['id'],row['start_frame'],row['stop_frame_exclusive'],row['annotation_line'])==
                    (sid,start,start+5*GEOMETRY[0],line),'Acoustic window identity differs: '+label)
            expected=annotation_box(label,start)
            require(all(row.get(k)==v for k,v in expected.items()),'Annotation extent differs: '+label)
            att,low,high=(data[label+'_'+k] for k in ('attenuation','envelope_min','envelope_max'))
            require(att.shape==(len(hz),len(t)) and att.dtype==np.uint8 and att.max()<=-CLIP_DB/POWER_STEP,
                    'Invalid spectral grid: '+label)
            require(low.shape==high.shape==(len(env_t),) and np.isfinite(low).all() and np.isfinite(high).all()
                    and np.all(low<=high) and low.min()>=-1 and high.max()<=1
                    and max(abs(low.min()),abs(high.max()))==1, 'Invalid peak envelope: '+label)
            maximum.append(att.min());labels[label]=dict(row,relative_power_db=-POWER_STEP*att.astype(np.float32),
                                                        envelope_min=low,envelope_max=high)
        require(min(maximum)==0,'Missing common spectral peak')
    return dict(info,labels=labels,time_s=t,frequency_hz=hz,envelope_time_s=env_t)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT,help='Project root containing state/protocol and '+RAW)
    p.add_argument('--output',type=Path,default=Path('result/acoustic_examples'),help='Feature directory; identical files are verified and reused')
    p.add_argument('--listening-copy', action='store_true',help='Also write result/acoustic_comparison.wav (KW, HW, AB; a common gain)')
    a=p.parse_args()
    try:
        data=extract(a.root,a.root/a.output,a.listening_copy)
    except (OSError,ValueError,KeyError,TypeError) as exc:
        p.exit(1,f'Acoustic export failed: {exc}\n')
    print(f"Verified {len(data['labels'])} training windows from {data['recording']} to {a.root / a.output}")

if __name__=='__main__': main()
