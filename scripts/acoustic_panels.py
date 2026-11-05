"""Training-window context for threshold and site comparisons."""
from pathlib import Path
import hashlib
import json
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.colorbar import ColorbarBase
from matplotlib.lines import Line2D


INK = '#142B38'
MUTED = '#455E68'
POWER = LinearSegmentedColormap.from_list('received_power', (
    (0., '#102332'), (.16, '#193947'), (.30, '#315A69'), (.42, '#668D90'),
    (.56, '#AEBBAB'), (.70, '#E3C98E'), (.84, '#F3DA8E'), (1., '#FFF6C4')))
SCALE = Normalize(-60., 0., clip=True)


def context(root, directory, disabled, analysis):
    """Automatically pair examples only with runs that identify their input protocol."""
    none = {'mode': 'none'}
    if disabled:
        return None, none
    explicit = directory is not None
    generated = root/'result/acoustic_examples'
    local = not explicit and (generated.exists() or generated.is_symlink())
    bundled = root/'assets/acoustic_examples'
    name = root / directory if explicit else generated if local else bundled
    if any(p.is_symlink() for p in (name, *name.parents)):
        raise ValueError('Redirected acoustic feature directory: '+str(name))
    if local and not name.is_dir():
        raise ValueError('Generated acoustic features are not a readable directory: '+str(name))
    previous = False
    if local:
        from acoustic_examples import is_previous_bundle
        previous = is_previous_bundle(name)
        if previous:
            if not bundled.is_dir() or any(p.is_symlink() for p in (bundled,*bundled.parents)):
                raise ValueError('Updated acoustic assets unavailable: '+str(bundled))
            name = bundled
    directory = name.resolve()
    if not explicit and (not directory.exists() or analysis is None):
        return None, none
    expected = None
    if analysis is not None:
        trace = json.loads((analysis/'analysis.json').read_text(encoding='utf-8'))
        expected = trace.get('used_file_sha256', {}).get('primary:state/protocol/complete.json')
    if not explicit and expected is None:
        return None, none
    from acoustic_examples import load_examples
    examples = load_examples(directory)
    if expected is not None and expected != examples['protocol_sha256']:
        if explicit or local:
            raise ValueError('Selected audio examples belong to a different run protocol')
        return None, none
    hashes = {n: hashlib.sha256((directory/n).read_bytes()).hexdigest()
              for n in ('examples.json','features.npz')}
    location = directory.relative_to(root.resolve()) if directory.is_relative_to(root.resolve()) else directory
    return examples, {'mode': 'measured', 'asset_directory': location.as_posix(), 'asset_sha256': hashes,
                      'recording': examples['recording'], 'site': examples['site'],
                      'labels': ['KW', 'HW', 'AB'],
                      'window_ids': {k: examples['labels'][k]['id'] for k in ('KW','HW','AB')},
                      'power_limits_db': [float(SCALE.vmin), float(SCALE.vmax)],
                      'reference': 'Shared spectral-power maximum of the three training windows',
                      'examples_used_for_metrics': False,
                      **({'previous_generated_bundle_skipped': True} if previous else {})}


def add_band(fig, number, ex):
    """Append measured acoustics without compressing any statistical data axis."""
    if ex is None:
        return
    old_w, old_h = fig.get_size_inches()
    extra = .16 if number == 1 else 0.
    band = 1.37 + extra
    total = old_h + band
    scale = old_h/total
    offset = band/total
    legends = [(leg, leg.get_bbox_to_anchor().transformed(fig.transFigure.inverted()).bounds)
               for leg in fig.legends]
    before = [(ax, ax.get_position().bounds) for ax in fig.axes]
    fig.set_size_inches(old_w, total, forward=True)
    for ax, (x,y,w,h) in before:
        ax.set_position([x, offset + scale*y, w, scale*h])
    for txt in fig.texts:
        x, y = txt.get_position()
        txt.set_position((x, offset + scale*y))
    for leg, (x,y,w,h) in legends:
        leg.set_bbox_to_anchor((x, offset+scale*y, w, scale*h), transform=fig.transFigure)
    def text(x, y_inches, value, **kw):
        fig.text(x,y_inches/total,value,fontsize=kw.pop('fontsize',8),
                 color=kw.pop('color',INK),va='center',**kw)
    fig.add_artist(Line2D([.015,.985],[offset-.030/total]*2,color='#C8D4D3',
                          lw=.65,transform=fig.transFigure))
    text(.014,1.155 + extra, 'd' if number==1 else 'e',fontsize=10,fontweight='bold')
    text(.050,1.155 + extra,'Annotated training windows' if number==1 else 'Killer whale training window',fontsize=9)
    spans = {k:(v['annotation_start_s'],v['annotation_stop_s']) for k,v in ex['labels'].items()
             if 'annotation_start_s' in v and 'annotation_stop_s' in v}
    if number==1 and spans:
        y=(1.155+extra)/total
        fig.add_artist(Line2D([.354,.380],[y,y],transform=fig.transFigure,
                              color='#354E58',lw=1.3,solid_capstyle='butt'))
        text(.391,1.155+extra,'Label span',color=MUTED)
    if number==1:
        # All three meshes use the same reference and transform; brightness is comparable.
        text(.550,1.155 + extra,'Power (dB re shared peak)',ha='left')
        color_ax=fig.add_axes([.822,(1.155 + extra)/total,.137,.064/total],label='sound_power_color_scale')
        cb=ColorbarBase(color_ax,cmap=POWER,norm=SCALE,ticks=[-60,-30,0],orientation='horizontal')
        cb.outline.set_visible(False)
        cb.ax.tick_params(axis='x',length=1.8,width=.55,pad=1,labelsize=8,labelcolor=MUTED)
        cb.ax.set_xticklabels(['−60','−30','0'])
    else:
        text(.965,1.155 + extra,'Same target window as Fig. 1(d)',ha='right',color=MUTED)
    t=np.asarray(ex['time_s']);freq=np.asarray(ex['frequency_hz'])/1000
    dt=float(np.diff(t).mean()); df=float(np.diff(freq).mean())
    extent=(t[0]-dt/2,t[-1]+dt/2,freq[0]-df/2,freq[-1]+df/2)
    panels = [('KW','Target: killer whale'),('HW','Negative: humpback whale'),('AB','Negative: background')] if number==1 else [
              ('wave','Waveform envelope'), ('spectrum','Time-averaged spectrum')]
    for idx,(key,title) in enumerate(panels):
        if number==1:
            left=.105+.300*idx; width=.269
        else:
            left=.108 if idx==0 else .586; width=.367
        text(left,.915+extra,title,ha='left',fontweight='semibold' if key=='KW' else 'normal',
             color={'KW':'#145D7E','HW':'#007A70','AB':'#A64614'}.get(key,INK))
        ax=fig.add_axes([left,.350/total,width,(.469+extra)/total],label='signal_'+key)
        ax.tick_params(labelsize=8,width=.55,length=2.0,pad=2,labelcolor=MUTED,color=MUTED)
        ax.spines[:].set_visible(False)
        ax.spines[['left','bottom']].set_visible(True)
        for edge in ('left','bottom'):
            ax.spines[edge].set_linewidth(.58);ax.spines[edge].set_color(MUTED)
        if key=='spectrum':
            # The inputs are half-dB display samples before visible-range clipping.
            # Average power in linear units, not dB; retain the common reference.
            db=ex['labels']['KW']['relative_power_db'].astype(np.float64)
            mean_db=10*np.log10(np.power(10.,db/10).mean(axis=1))
            profile,=ax.plot(freq,mean_db,color='#145D7E',lw=1.05)
            profile.set_gid('KW_mean_spectral_power')
            ax.set_xscale('log');ax.minorticks_off();ax.set_xlim(freq[0],freq[-1]);ax.set_xticks([.25,1,4,7.5],['0.25','1','4','7.5'])
            ax.set_xlabel('Frequency (kHz)',fontsize=8,labelpad=2,color=INK)
            lo = 10*np.floor(mean_db.min()/10); hi = min(0.,10*np.ceil(mean_db.max()/10))
            lo = min(lo, hi-20.)
            ax.set_ylim(lo,hi);ax.set_yticks([lo, (lo+hi)/2,hi]);ax.set_ylabel('dB re peak',fontsize=8,labelpad=1,color=INK)
            continue
        ax.set_xlim(0,5)
        if number==1:
            ax.set_xticks([0,2.5,5],['0','2.5','5'])
            if idx==1:
                ax.set_xlabel('Time in window (s)',fontsize=8,labelpad=2,color=INK)
        else:
            ax.set_xticks([0,1,2,3,4,5]);ax.set_xlabel('Time in window (s)',fontsize=8,labelpad=2,color=INK)
        if key=='wave':
            sample=ex['labels']['KW'];tx=np.asarray(ex['envelope_time_s']);dt_env=float(np.diff(tx).mean())
            edges=np.r_[tx-dt_env/2,tx[-1]+dt_env/2]
            ax.axhline(0,color='#D5DFE0',lw=.55)
            envelope=ax.fill_between(edges,np.r_[sample['envelope_min'],sample['envelope_min'][-1]],
                        np.r_[sample['envelope_max'],sample['envelope_max'][-1]],step='post',
                        color='#145D7E',linewidth=.1,edgecolor='#145D7E')
            envelope.set_gid('KW_measured_amplitude_range')
            ax.set_ylim(-1,1);ax.set_yticks([-1,0,1]);ax.set_ylabel('Rel. amp.',fontsize=8,labelpad=1,color=INK)
        else:
            im=ax.pcolormesh(np.r_[t-dt/2,t[-1]+dt/2],np.r_[freq-df/2,freq[-1]+df/2],
                             ex['labels'][key]['relative_power_db'],cmap=POWER,norm=SCALE,
                             shading='flat',rasterized=True,antialiased=False)
            ax.set_yscale('log');ax.minorticks_off()
            im.set_gid(key+'_measured_power');ax.set_ylim(extent[2],extent[3]);ax.set_yticks([.25,1,7.5],['0.25','1','7.5'])
            if key in spans:
                a,b=spans[key]
                if not (0 <= a < b <= 5): raise ValueError('Invalid annotation span: '+key)
                # Use native window time; place the span above the mesh, away from signals.
                bar,=ax.plot([a,b],[1.035,1.035],transform=ax.get_xaxis_transform(),clip_on=False,
                             color='#354E58',lw=1.3,solid_capstyle='butt')
                bar.set_gid(key+'_annotation_span')
            if idx==0:
                ax.set_ylabel('kHz',fontsize=8,labelpad=2,color=INK)
            else:
                ax.tick_params(axis='y',labelleft=False)
