import pandas as pd, json, numpy as np
R='/home/abreham/research/LiteratureAnnotator/weak_supervision/data/'
df=pd.read_parquet(R+'samples.parquet')
pm=json.load(open(R+'pubmed_meta.json'))
paper=df.groupby('pmid').agg(stratum=('stratum','first'),chars=('rendered_chars','first'),full=('is_full_text','first'),trunc=('truncated','first'))
print(paper.stratum.value_counts())
print(paper.chars.describe(percentiles=[.1,.25,.5,.75,.9,.95,.99]))
print('full text',paper.full.value_counts().to_dict(),'truncated',paper.trunc.value_counts().to_dict())
print(paper.groupby('stratum').chars.median())
# majority label + agreement
def maj(s):
    vc=s.dropna().value_counts(); return vc.index[0] if len(vc) else None
g=df.groupby(['pmid','criterion']).verdict
lab=g.agg(maj).unstack(); agree=g.agg(lambda s: s.value_counts().iloc[0]/len(s)).unstack()
for c in lab.columns: print(c, lab[c].value_counts().to_dict(), 'unanimous', round((agree[c]==1).mean(),3))
print('papers with any disagreement', (agree<1).any(axis=1).sum(), 'n disagreeing criteria dist', (agree<1).sum(axis=1).value_counts().sort_index().to_dict())
pt=pd.Series({k:v.get('pub_types',[]) for k,v in pm.items()})
from collections import Counter; print(Counter(t for l in pt for t in l).most_common(15))
for f in ['study_v1/selection/smoke_pmids.txt','data_download/strata/pilot_2026_10_sample33.txt']:
    ids=open('/home/abreham/research/LiteratureAnnotator/.worktrees/step0-generation/'+f).read().split()
    print(f, len(ids), 'in pilot', sum(i in paper.index for i in ids), paper.reindex(ids).stratum.value_counts().to_dict())
print(paper.loc[[i for i in ['33386803','37570596','37252737'] if i in paper.index]])
paper.join(lab).join(agree.add_prefix('agr_')).to_csv('/home/abreham/research/LiteratureAnnotator/.worktrees/step0-generation/study_v1/selection/pool_summary.csv')
