"""Site-standardized pAUC calculated separately per fit, never pooling scores."""
from collections import defaultdict
from statistics import fmean

import numpy as np
from sklearn.metrics import roc_auc_score

from .utils import write_csv


def score_sites(fold, rows, scores, methods, seeds, max_fpr):
    """Compute McClish standardized partial ROC area using all scores/ties."""
    y = np.asarray([r['label'] for r in rows]); p = np.asarray(scores)
    if p.shape != (len(methods),len(seeds),len(y)) or not 0 < max_fpr <= 1:
        raise ValueError('Unexpected score tensor or pAUC upper limit')
    groups = defaultdict(list)
    for i,row in enumerate(rows):groups[row['site']].append(i)
    out=[]
    for site,idx in sorted(groups.items()):
        yy=y[idx]; positives=int(yy.sum()); negatives=len(idx)-positives
        for mi,method in enumerate(methods):
            for si,seed in enumerate(seeds):
                pred=p[mi,si,idx]; dual=positives>0 and negatives>0
                out.append(dict(fold=fold,method=method,seed=seed,site=site,
                    negative_windows=negatives,positive_windows=positives,
                    pauc=float(roc_auc_score(yy,pred,max_fpr=max_fpr)) if dual else None,
                    auc=float(roc_auc_score(yy,pred)) if dual else None,pauc_max_fpr=max_fpr))
    return out


def rank_tables(rows):
    """Exactly equal weight per eligible site, per seed, and per observed fold."""
    byfit=defaultdict(list);seen=set();shapes={}
    for r in rows:
        fit=r['fold'],r['method'],r['seed'];key=fit+(r['site'],)
        if key in seen:raise ValueError('Duplicate site/fit: '+str(key))
        seen.add(key);byfit[fit].append(r)
    fit_metrics=[];sensitivity=[]
    for minimum in (1,20):
        grouped=defaultdict(list)
        for (fold,method,seed),rs in sorted(byfit.items()):
            support=sorted((r['site'],r['positive_windows'],r['negative_windows']) for r in rs)
            if shapes.setdefault(fold,support)!=support:raise ValueError('Site support differs across fits: '+fold)
            dual=[r for r in rs if r['positive_windows'] and r['negative_windows']>=minimum]
            if any(r[k] is None or not np.isfinite(r[k]) or not 0<=r[k]<=1 for r in dual for k in ('pauc','auc')):
                raise ValueError('Eligible site lacks a valid rank metric')
            p=fmean(r['pauc'] for r in dual) if dual else None
            if minimum==1:
                fit_metrics.append(dict(fold=fold,method=method,seed=seed,macro_pauc=p,
                    macro_auc=fmean(r['auc'] for r in dual) if dual else None))
            grouped[fold,method].append((p,len(dual)))
        for (fold,method),vals in sorted(grouped.items()):
            sensitivity.append(dict(fold=fold,method=method,macro_pauc=fmean(p for p,_ in vals) if vals[0][1] else None,
                                    sites=vals[0][1],min_negative=minimum))
    macro=defaultdict(list)
    for r in sensitivity:
        if r['min_negative']==1:macro[r['method']].append(r['macro_pauc'])
    main=[dict(method=m,macro_pauc=fmean(ps) if None not in ps else None) for m,ps in sorted(macro.items())]
    return dict(site_metrics=sorted(rows,key=lambda r:(r['fold'],r['method'],r['seed'],r['site'])),
                fit_metrics=fit_metrics,support_sensitivity=sensitivity,main_results=main)


def write_rank_tables(output,rows):
    for table,values in rank_tables(rows).items():write_csv(output/(table+'.csv'),values)
