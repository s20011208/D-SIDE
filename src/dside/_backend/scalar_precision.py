"""Observation-noise propagation for the unchanged scalar ratio feature.

Delta variances are reliability heuristics, not confidence intervals: matching
states are estimated and polynomial approximation error is not stochastic noise.
"""
from itertools import combinations
import numpy as np
from .core import green_matrices
from .scalar_sources import scalar_pairs,weighted_quantiles
from .scalar_evidence import average_evidence_clusters


def linear_coordinates(raw, sm, regularization=1e-7):
    T=sm.t[-1]
    obs=2*(raw.t-raw.t[0])/T-1
    Z=np.polynomial.chebyshev.chebvander(obs,5)
    pinv=np.linalg.pinv(Z)
    X=np.polynomial.chebyshev.chebvander(2*sm.t/T-1,5)@pinv
    derivative=np.array([np.polynomial.Chebyshev.basis(k,domain=[0,T]).deriv()(0) for k in range(6)])@pinv
    _,G=green_matrices(sm.t,[0.],2);_,V=green_matrices(sm.t,[],1)
    u,s,vt=np.linalg.svd(G,full_matrices=False)
    R=(vt.T*(s/(s*s+regularization*s[0]**2)))@u.T
    B=R@(X-X[0]-sm.t[:,None]*derivative)
    A=R@(V@X-sm.t[:,None]*X[0])
    return B,A,X


def precision_pairs(raw, smooth, noise_sd, trim=.02):
    # Source-balanced velocity scale; a high-data source cannot set this cutoff.
    groups=sorted({s.group for s in smooth})
    scale=np.median([np.sqrt(np.mean([np.mean(s.derivatives[:,1]**2)
        for s in smooth if s.group==g])) for g in groups])
    rows,di=scalar_pairs(smooth,trim=trim,denominator_fraction=0.)
    coordinate={}
    for r,s in zip(raw,smooth):
        B,A,X=linear_coordinates(r,s)
        b,a=B@r.y,A@r.y
        coordinate[s.name]=(s,B,A,X,np.gradient(b,s.t),np.gradient(a,s.t))
    result=[];rejections={"velocity":0,"denominator":0}
    for row in rows:
        matrices=[]
        reject=False
        for side in ["p","q"]:
            sm,B,A,X,bt,at=coordinate[row["trial_"+side]]
            time=row["time_"+side]
            v=float(np.interp(time,sm.t,sm.derivatives[:,1]))
            if abs(v)<.05*scale:
                reject=True;break
            def interp(M):
                index=min(int(time/(sm.t[1]-sm.t[0])),len(sm.t)-2)
                fraction=(time-sm.t[index])/(sm.t[index+1]-sm.t[index])
                return (1-fraction)*M[index]+fraction*M[index+1]
            xx=interp(X)
            bb=interp(B)-np.interp(time,sm.t,bt)/v*xx
            aa=interp(A)-np.interp(time,sm.t,at)/v*xx
            matrices.append((bb,aa))
        if reject:
            rejections["velocity"]+=1;continue
        (Bp,Ap),(Bq,Aq)=matrices
        same=row["trial_p"]==row["trial_q"]
        varD=noise_sd**2*(np.sum((Ap-Aq)**2) if same else np.sum(Ap**2)+np.sum(Aq**2))
        D,N=row["denominator"],row["numerator"]
        if abs(D)<max(.1*scale,2*np.sqrt(varD)):
            rejections["denominator"]+=1;continue
        denom=D*D+N*N
        lp=(-D*Bp-N*Ap)/denom;lq=(D*Bq+N*Aq)/denom
        varz=noise_sd**2*(np.sum((lp+lq)**2) if same else np.sum(lp**2)+np.sum(lq**2))
        result.append(dict(row,variance_z=float(varz),
            weight=float(row["weight"]/(varz+.05**2))))
    return result,dict(original=di,rejections=rejections,source_balanced_speed=float(scale))


def precision_distances(raw,smooth,groups,noise_sd,mode="precision",trim=.02):
    rows,di=precision_pairs(raw,smooth,noise_sd,trim)
    qs=(np.arange(41)+.5)/41
    samples={};uncertainty={}
    for g in range(groups):
        for h in range(g,groups):
            rr=[r for r in rows if r["g"]==g and r["h"]==h]
            if rr:
                w=np.array([r["weight"] for r in rr])
                samples[g,h]=weighted_quantiles(np.arctan([r["statistic"] for r in rr]),w,qs)
                uncertainty[g,h]=float(weighted_quantiles(
                    np.sqrt([r["variance_z"] for r in rr]),w,[.5])[0])
    # Scalar within-source estimates locate the source, without interpreting a
    # cross-source ratio as a shared true parameter.
    centers={}
    for g in range(groups):
        if (g,g) in samples:
            centers[g]=float(samples[g,g][20])
        else:
            local=[q[20] for pair,q in samples.items() if g in pair]
            if local:centers[g]=float(np.median(local))
    D=np.full((groups,groups),np.inf);np.fill_diagonal(D,0)
    evidence={}
    for g,h in combinations(range(groups),2):
        if g not in centers or h not in centers:continue
        own=abs(centers[g]-centers[h])
        if (g,h) in samples:
            residual=np.abs(samples[g,h]-.5*(centers[g]+centers[h]))
            cross=float(np.mean(residual))
            # Remove expected absolute Gaussian perturbation, WITHOUT
            # multiplying precision by the number of dependent pair rows.
            corrected=max(0.,cross-.8*uncertainty[g,h])
            distance=max(own,corrected)
            kind="cross_plus_within"
        else:
            distance=own;kind="provisional_within_only"
        if mode=="precision_raw":
            distance=own if (g,h) not in samples else max(own,cross)
        D[g,h]=D[h,g]=distance
        evidence[f"{g}_{h}"]=dict(distance=distance,kind=kind)
    return D,dict(pairs=rows,centers={str(k):v for k,v in centers.items()},
                  distances=D.tolist(),evidence=evidence,diagnostics=di)


def precision_clusters(raw,smooth,groups,noise_sd,mode="precision",trim=.02,thresholds=(.15,.3,.6)):
    D,info=precision_distances(raw,smooth,groups,noise_sd,mode,trim)
    return {str(t):average_evidence_clusters(D,t).tolist() for t in thresholds},info
