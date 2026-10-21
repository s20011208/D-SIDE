import numpy as np
from dside._backend.global_consensus import (shortest_mass_interval, correlation_partition,
    partitions, select_global_mixture, mixture_parts, generate)


def test_interval_matches_brute_force():
    rng=np.random.default_rng(40)
    x=np.sort(rng.normal(size=20));w=rng.uniform(.1,1,size=20);w/=w.sum()
    actual=shortest_mass_interval(x,w,.5)
    candidates=[(x[j]-x[i],x[i],x[j]) for i in range(20) for j in range(i,20) if w[i:j+1].sum()>=.5-1e-12]
    assert actual==min(candidates)[1:]


def test_partition_does_not_need_cluster_count():
    S=np.full((8,8),.05);S[:6,:6]=.95;np.fill_diagonal(S,1.)
    clusters,audit=correlation_partition(S)
    assert clusters==[[0,1,2,3,4,5],[6],[7]]
    assert audit['partitions_evaluated']==4140
    assert len(list(partitions(8)))==4140


def test_missing_is_unknown_and_no_evidence_does_not_merge():
    S=np.full((5,5),np.nan);np.fill_diagonal(S,1)
    assert correlation_partition(S)[0]==[[0],[1],[2],[3],[4]]


def test_global_mixture_score_is_likelihood_derivative():
    z=np.array([-.9,.1,.4,1.3]);p=np.array([.3,.25,.6]);eps=1e-6
    plus=p.copy();minus=p.copy();plus[2]+=eps;minus[2]-=eps
    derivative=(mixture_parts(z,plus)[0]-mixture_parts(z,minus)[0])/(2*eps)
    gamma=mixture_parts(z,p)[1]
    assert np.allclose(derivative,(gamma-p[2])/(p[2]*(1-p[2])),rtol=1e-6)


def test_pair_balance_and_own_rows_have_no_effect():
    rng=np.random.default_rng(112)
    rows=[]
    for g in range(5):
        for h in range(g+1,5):
            z=rng.normal(.3,.08,30) if h<4 else rng.uniform(-1.3,1.3,30)
            rows += [dict(g=g,h=h,statistic=float(np.tan(v)),weight=1.) for v in z]
    a=select_global_mixture(rows,5)
    b=select_global_mixture(rows+[r.copy() for r in rows if (r['g'],r['h'])==(0,1)]
        +[dict(g=0,h=0,statistic=100.,weight=1000.)],5)
    assert a['clusters']==b['clusters']
    assert np.allclose(list(a['peak'].values()),list(b['peak'].values()),atol=1e-6)
    assert np.allclose(a['support'],b['support'],atol=1e-6)
