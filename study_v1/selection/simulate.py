import pandas as pd, numpy as np
p=pd.read_csv('study_v1/selection/pool_summary.csv',dtype={'pmid':str}).set_index('pmid')
for c in [f'criterion_{i}' for i in range(1,7)]: p[c]=p[c].astype(str).str.lower()
C=[f'criterion_{i}' for i in range(1,7)]
p['n_dis']=(p[[f'agr_{c}' for c in C]]<1).sum(axis=1)
OUT={'opus':(2,2000,10),'sol':(1,1500,5),'gemini':(1,800,6),'deepseek':(.035,800,.149),'qwen':(1.15,1300,3.45),'glm':(.28,2000,1.249)}
def cost(chars):  # s00, all six models, no cache hits
    tin=6*(chars/4.8+1300); return sum(tin*i/1e6+6*o*r/1e6 for i,o,r in OUT.values())
p['cost']=cost(p.chars)
print('pool mean cost/paper $%.3f, 100 mean-length papers $%.0f'%(p.cost.mean(),100*p.cost.mean()))
def report(name,s):
    row={'design':name,'mean_kchars':round(s.chars.mean()/1e3,1),'s00_$':round(s.cost.sum()),'any_dis':int((s.n_dis>0).sum())}
    for c in C:
        vc=s[c].value_counts(); row[c[-1]]='/'.join(str(vc.get(k,0)) for k in ['true','false','not_applicable'][:3 if c in('criterion_3','criterion_6') else 2])
    return row
rng=np.random.default_rng(20261126); rows=[]
cap=p[p.chars<=80000]
for name,pool in [('random_all',p),('random_cap80k',cap)]:
    rows.append(report(name,pool.sample(100,random_state=1)))
eq=cap.groupby('stratum',group_keys=False).apply(lambda g:g.sample(min(len(g),9),random_state=1)); rows.append(report('equal_per_stratum_cap80k',eq.head(100)))
# disagreement-enriched: half from papers with any disagreement
d=cap[cap.n_dis>0].sample(50,random_state=1); a=cap[cap.n_dis==0].sample(50,random_state=1); rows.append(report('50pct_disagree_cap80k',pd.concat([d,a])))
rows.append(report('POOL(679)',p))
print(pd.DataFrame(rows).to_string(index=False))
print('papers >80k chars:',(p.chars>80000).sum(), ' >60k:',(p.chars>60000).sum())
print(p.groupby('stratum').agg(n=('cost','size'),dis=('n_dis',lambda x:(x>0).mean())).round(2))
