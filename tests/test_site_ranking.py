import unittest
import numpy as np
from orcatca.site_ranking import score_sites,rank_tables

class SiteRanking(unittest.TestCase):
    def test_ties_filter_and_equal_site_not_window(self):
        # One tiny site ranks perfectly; a large site is all tied (pAUC=.5).
        # The third, all-negative site must never enter a rank mean.
        rows=[dict(site=s,label=y) for s,labels in [('tiny',[0,1]),('large',[0]*21+[1]*3),('one_class',[0]*25)] for y in labels]
        score=np.asarray([0,1]+[.5]*(len(rows)-2))[None,None,:]
        ranks=rank_tables(score_sites('f',rows,score,['bin'],[17],.1))
        self.assertEqual(ranks['fit_metrics'][0]['macro_pauc'],.75)
        sens=ranks['support_sensitivity'];self.assertEqual([(r['min_negative'],r['sites'],r['macro_pauc']) for r in sens],[(1,2,.75),(20,1,.5)])
        self.assertTrue(any(r['pauc'] is None for r in ranks['site_metrics']))
    def test_equal_collections_and_missing_support(self):
        rows=[]
        for f,rs in [('a',[(1,.6),(21,.8)]),('b',[(2,.9)])]:
            for seed in [10,22]:
                for j,(n,p) in enumerate(rs):rows.append(dict(fold=f,method='m',seed=seed,site=str(j),positive_windows=2,negative_windows=n,pauc=p,auc=p))
        result=rank_tables(rows);self.assertAlmostEqual(result['main_results'][0]['macro_pauc'],.8)
        self.assertIsNone(next(r['macro_pauc'] for r in result['support_sensitivity'] if r['fold']=='b' and r['min_negative']==20))
