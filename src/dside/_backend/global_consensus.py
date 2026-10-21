"""Global labelled scalar consensus without any within-source reference centre.

The selector uses a pair-balanced empirical distribution, a shortest-mass
interval and exact weighted correlation clustering. All objectives are finite
sample descriptive criteria; dependent ratio rows do not justify iid inference.
"""
from dataclasses import replace
from itertools import combinations
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.special import ndtr
from .core import Trial, GreenModel
from .baselines import IndependentModel
from .scalar_sources import selector_smoothing
from .scalar_precision import precision_pairs


def rhs_truth(group, x):
    x = np.asarray(x)
    stiffness, nonlinear = (1., .45) if group < 6 else ((1.9, -.9) if group == 6 else (.65, -.65))
    return -stiffness*x + nonlinear*np.sin(1.7*x)-.1*x**3/(1+x*x)


def damping_truth(config):
    return np.array(config['good_damping']+config['bad_damping'])


def generate(config, seed, split='train'):
    """Exactly one observed training trajectory per source, independent RNG streams."""
    assert config['groups'] == 8 and config['train_trials'] == 1
    answer=[]
    damping=damping_truth(config)
    count=1 if split=='train' else config['test_trials']
    duration=config['duration'] if split=='train' else config['forecast_horizon']
    for group in range(8):
        for i in range(count):
            rng=np.random.default_rng(np.random.SeedSequence([seed, 8201, group, int(split!='train'), i]))
            initial=rng.uniform(-1.25,1.25,2)
            sol=solve_ivp(lambda t,z: [z[1],rhs_truth(group,z[0])-damping[group]*z[1]],
                (0,duration),initial,method='DOP853',dense_output=True,rtol=1e-10,atol=1e-12)
            if not sol.success: raise RuntimeError(sol.message)
            times=np.r_[0.,np.sort(rng.uniform(0,duration,config['observations']-2)),duration]
            y=sol.sol(times)[0]+rng.normal(0,config['noise_sd'],len(times))
            tt=np.linspace(0,duration,101)
            answer.append(Trial(group,f'{split}-g{group}-t{i}',times,y,initial,tt,sol.sol(tt)[0]))
    return answer


def shortest_mass_interval(values, weights, mass=.5):
    """Exact narrowest data-endpoint interval with at least the requested mass."""
    x,w=np.asarray(values,float),np.asarray(weights,float)
    if not 0 < mass <= 1 or len(x)==0 or x.shape!=w.shape or np.any(w<0) or w.sum()<=0:
        raise ValueError('Invalid weighted empirical distribution')
    if not np.isfinite(x).all() or not np.isfinite(w).all(): raise ValueError('Nonfinite input')
    order=np.argsort(x,kind='stable');x=x[order];w=w[order]/w.sum()
    c=np.r_[0.,np.cumsum(w)];best=None
    for left in range(len(x)):
        target=c[left]+mass
        if target > 1+1e-12: break
        right=min(len(x)-1,max(left,int(np.searchsorted(c,target-1e-12))-1))
        candidate=(float(x[right]-x[left]),float(x[left]),float(x[right]))
        if best is None or candidate<best:best=candidate
    return best[1],best[2]


def partitions(n):
    """Canonical set partitions; Bell(8)=4140, no assumed number of clusters."""
    labels=np.zeros(n,dtype=int)
    def walk(i,maximum):
        if i==n:
            yield labels.copy();return
        for k in range(maximum+2):
            labels[i]=k
            yield from walk(i+1,max(maximum,k))
    yield from walk(1,0)


def correlation_partition(support, threshold=.5):
    """Maximize sum_{same cluster, observed g<h} (support[g,h]-threshold).

    This is the weighted-agreement correlation-clustering criterion up to a
    partition-independent constant. Missing comparisons give zero evidence.
    Ties prefer fewer merged pairs. Exhaustive search is only for <=9 sources.
    """
    S=np.asarray(support,float);n=len(S)
    if S.shape!=(n,n) or n>9 or not np.allclose(S,S.T,equal_nan=True):
        raise ValueError('Symmetric support matrix for at most nine sources required')
    g,h=np.triu_indices(n,1);v=np.nan_to_num(S[g,h]-threshold,nan=0.)
    best=None;objectives=[]
    for labels in partitions(n):
        together=labels[g]==labels[h]
        value=float(v@together)
        candidate=(value,-int(together.sum()))
        if best is None or candidate>best[0]:best=(candidate,labels.copy())
        objectives.append(value)
    labels=best[1]
    clusters=[np.flatnonzero(labels==k).tolist() for k in np.unique(labels)]
    return clusters,dict(objective=best[0][0],partitions_evaluated=len(objectives),
                         second_best_objective=float(sorted(objectives)[-2]))


def select_from_rows(rows, groups, mass=.5, threshold=.5):
    """Only cross-source samples enter the global reference and pair support."""
    pairs={}
    for r in rows:
        g,h=r['g'],r['h']
        if g==h:continue
        pairs.setdefault((g,h),[]).append(r)
    if not pairs:
        return dict(clusters=[[g] for g in range(groups)],interval=None,support=np.full((groups,groups),np.nan),
                    rows=[],status='no_cross_source_evidence')
    z=[];weights=[];label=[];export=[]
    for pair,rr in sorted(pairs.items()):
        local=np.array([r['weight'] for r in rr]);local=local/local.sum()
        for r,w in zip(rr,local):
            z.append(np.arctan(r['statistic']));weights.append(w/len(pairs));label.append(pair)
            export.append(dict(r,z=float(z[-1]),pair_weight=float(w),global_weight=float(weights[-1])))
    z,w=np.array(z),np.array(weights)
    lo,hi=shortest_mass_interval(z,w,mass)
    S=np.full((groups,groups),np.nan);np.fill_diagonal(S,1.)
    for pair in pairs:
        ii=np.array([p==pair for p in label]);ww=w[ii]
        S[pair]=S[pair[::-1]]=float(np.sum(ww*((z[ii]>=lo)&(z[ii]<=hi)))/ww.sum())
    clusters,fit=correlation_partition(S,threshold)
    for r in export:r['inside']=bool(lo<=r['z']<=hi)
    return dict(clusters=clusters,interval=[lo,hi],support=S,rows=export,status='ok',
        mass=float(np.sum(w*((z>=lo)&(z<=hi)))),requested_mass=mass,threshold=threshold,
        available_pairs=len(pairs),clustering=fit)


def select(raw,smooth,noise_sd,mass=.5,threshold=.5,trim=.02):
    sm=selector_smoothing(raw,smooth,'poly5')
    rows,diagnostics=precision_pairs(raw,sm,noise_sd,trim=trim)
    result=select_from_rows(rows,len(raw),mass,threshold)
    result['precision_diagnostics']=diagnostics
    result['all_precision_rows']=rows
    return result


def mixture_parts(z, parameters):
    """Truncated Gaussian peak plus uniform background on the atan domain."""
    mu,sigma,proportion=np.asarray(parameters,float)
    normalizer=ndtr((np.pi/2-mu)/sigma)-ndtr((-np.pi/2-mu)/sigma)
    log_peak=-.5*((np.asarray(z)-mu)/sigma)**2-np.log(sigma*np.sqrt(2*np.pi)*normalizer)
    log_background=-np.log(np.pi)
    log_mixture=np.logaddexp(np.log(proportion)+log_peak,np.log1p(-proportion)+log_background)
    responsibility=np.exp(np.log(proportion)+log_peak-log_mixture)
    return log_mixture,responsibility,log_peak


def select_global_mixture(rows,groups):
    """V3: global mixture score -> exact correlation clustering -> dominant core.

    Enrichment gamma-pi, not gamma-.5, is proportional to the derivative of
    log[pi*peak+(1-pi)*background] w.r.t. pi. It measures peak support relative
    to the GLOBAL mixture background without any source-specific centre.
    This is a weighted pseudo-likelihood; no iid likelihood tests/CI are used.
    """
    pairs={}
    for row in rows:
        if row['g']!=row['h']:pairs.setdefault((row['g'],row['h']),[]).append(row)
    if not pairs:
        return dict(clusters=[[g] for g in range(groups)],core=[],rows=[],status='no_cross_source_evidence')
    exported=[]
    for pair,rr in sorted(pairs.items()):
        weight=np.array([r['weight'] for r in rr]);weight/=weight.sum()
        for r,w in zip(rr,weight):
            exported.append(dict(r,z=float(np.arctan(r['statistic'])),pair_weight=float(w),global_weight=float(w/len(pairs))))
    z=np.array([r['z'] for r in exported]);w=np.array([r['global_weight'] for r in exported])
    fits=[]
    for initial_mu in np.linspace(-1.2,1.2,9):
        result=minimize(lambda v:-float(w@mixture_parts(z,v)[0]),[initial_mu,.25,.6],
            method='L-BFGS-B',bounds=[(-1.5,1.5),(.04,.7),(.05,.95)])
        fits.append(result)
    fit=min(fits,key=lambda r:r.fun)
    mu,sigma,proportion=fit.x
    log_mix,gamma,_=mixture_parts(z,fit.x)
    normalizer=ndtr((np.pi/2-mu)/sigma)-ndtr((-np.pi/2-mu)/sigma)
    radius=sigma*np.sqrt(max(0.,2*np.log(np.pi/(sigma*np.sqrt(2*np.pi)*normalizer))))
    interval=[max(-np.pi/2,float(mu-radius)),min(np.pi/2,float(mu+radius))]
    S=np.full((groups,groups),np.nan);np.fill_diagonal(S,proportion)
    for pair in pairs:
        ii=np.array([(r['g'],r['h'])==pair for r in exported])
        S[pair]=S[pair[::-1]]=float(w[ii]@gamma[ii]/w[ii].sum())
    partition,fit_cluster=correlation_partition(S,threshold=proportion)
    def score(cluster):return sum(S[g,h]-proportion for g,h in combinations(cluster,2) if np.isfinite(S[g,h]))
    eligible=[cluster for cluster in partition if len(cluster)>=3]
    core=max(eligible,key=lambda cl:(score(cl),len(cl))) if eligible else []
    # Only the main consensus is supported by the single-peak model. Other
    # partition blocks do NOT establish additional shared mechanisms.
    clusters=([core] if core else [])+[[g] for g in range(groups) if g not in core]
    for row,q in zip(exported,gamma):
        row.update(responsibility=float(q),enrichment=float(q-proportion),inside=bool(q>=proportion))
    scores=[]
    for g in range(groups):
        comparisons=[S[g,h]-proportion for h in core if h!=g and np.isfinite(S[g,h])]
        scores.append(dict(group=g,core_support_sum=float(sum(comparisons)),observed_core_pairs=len(comparisons)))
    return dict(status='ok',core=core,clusters=clusters,partition=partition,support=S,
        enrichment=S-proportion,rows=exported,interval=interval,
        peak=dict(mu=float(mu),sigma=float(sigma),proportion=float(proportion),
                  weighted_log_likelihood=float(w@log_mix),uniform_log_likelihood=float(-np.log(np.pi))),
        mixture_starts=[dict(success=bool(r.success),objective=float(r.fun),parameters=r.x.tolist()) for r in fits],
        available_pairs=len(pairs),clustering=fit_cluster,source_scores=scores,
        criterion='sum within-partition pair-mean responsibility minus global mixture proportion; main peak block only')


def select_mixture(raw,smooth,noise_sd,trim=.02):
    sm=selector_smoothing(raw,smooth,'poly5')
    rows,diagnostics=precision_pairs(raw,sm,noise_sd,trim=trim)
    result=select_global_mixture(rows,len(raw))
    result['precision_diagnostics']=diagnostics
    result['all_precision_rows']=rows
    return result


class ClusterGreen:
    """Shared RKHS RHS inside a selected cluster; separate bounded damping.

    Each cluster builds its own dictionary from its own data. Singletons use
    the already fitted independent model. Excluded sources cannot influence
    selected-cluster fitting through a global kernel dictionary.
    """
    def __init__(self,clusters,independent=None,**fit):
        self.clusters=clusters;self.independent=independent;self.fit_options=fit

    def fit(self,smooth):
        if self.independent is None:self.independent=IndependentModel(**self.fit_options).fit(smooth)
        self.models={};self.mapping={};self.outer_log=[];self.inner_diagnostics=[]
        for cluster in self.clusters:
            if len(cluster)==1:
                g=cluster[0];self.models[g]=self.independent.models[g];self.mapping[g]=0
                continue
            local={g:i for i,g in enumerate(cluster)}
            train=[replace(t,group=local[t.group]) for t in smooth if t.group in local]
            model=GreenModel(pool=True,shared_operator=False,**self.fit_options).fit(train)
            self.outer_log += [dict(r,cluster=cluster) for r in model.outer_log]
            self.inner_diagnostics.append(dict(cluster=cluster,**model.inner_diagnostics))
            for g in cluster:self.models[g]=model;self.mapping[g]=local[g]
        self.omega=np.array([self.models[g].omega[self.mapping[g],0] for g in range(len(smooth))])[:,None]
        return self

    def rhs(self,g,x):return self.models[g].rhs(self.mapping[g],x)
    def predict(self,g,t,initial):return self.models[g].predict(self.mapping[g],t,initial)
