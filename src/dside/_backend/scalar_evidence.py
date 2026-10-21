"""Augment labelled scalar-ratio evidence with within-source distributions.

All statistics remain one dimensional. Disconnected same-damping sources are
not thereby identified as equal-force sources: weak links are provisional.
"""
from itertools import combinations
import numpy as np
from .scalar_sources import (scalar_pairs,weighted_quantiles,distribution_distances,
                             observed_complete_clusters)


def evidence_distances(trials, groups, trim=.02, mode="augmented"):
    rows,diagnostics = scalar_pairs(trials,trim=trim)
    probs=(np.arange(41)+.5)/41
    samples={}
    for g in range(groups):
        for h in range(g,groups):
            rr=[r for r in rows if r["g"]==g and r["h"]==h]
            if rr:
                samples[g,h]=weighted_quantiles(np.arctan([r["statistic"] for r in rr]),
                    np.array([r["weight"] for r in rr]),probs)
    profiles={}
    for g in range(groups):
        if (g,g) in samples:
            profiles[g]=samples[g,g]
        else:
            local=[q for key,q in samples.items() if g in key]
            if local:
                profiles[g]=np.median(local,axis=0)
    D=np.full((groups,groups),np.inf);np.fill_diagonal(D,0)
    kind={}
    for g,h in combinations(range(groups),2):
        if g not in profiles or h not in profiles:
            continue
        own=float(np.mean(np.abs(profiles[g]-profiles[h])))
        if mode=="within":
            d=own;reason="profile_only"
        elif mode=="augmented":
            if (g,h) in samples:
                cross=float(np.mean(np.abs(samples[g,h]-.5*(profiles[g]+profiles[h]))))
                d=max(own,cross);reason="cross_and_within"
            else:
                d=own;reason="provisional_profile_no_cross"
        else:
            raise ValueError(mode)
        D[g,h]=D[h,g]=d
        kind[f"{g}_{h}"]=dict(distance=d,evidence=reason)
    return D,dict(pairs=rows,profiles={str(g):q.tolist() for g,q in profiles.items()},
        distances=D.tolist(),edge_evidence=kind,diagnostics=diagnostics)


def average_evidence_clusters(D, threshold):
    """Average observed discrepancy avoids the single noisy-edge veto."""
    clusters=[[g] for g in range(len(D))]
    while True:
        candidates=[]
        for i,j in combinations(range(len(clusters)),2):
            edges=D[np.ix_(clusters[i],clusters[j])].ravel()
            finite=edges[np.isfinite(edges)]
            if len(finite) and np.mean(finite)<=threshold:
                candidates.append((float(np.mean(finite)),i,j))
        if not candidates:break
        _,i,j=min(candidates)
        clusters[i]=sorted(clusters[i]+clusters[j]);del clusters[j]
    labels=np.empty(len(D),int)
    for k,c in enumerate(clusters):labels[c]=k
    return labels


def revision_clusters(trials, groups, mode="augmented", trim=.02, thresholds=(.15,.3,.6),
                      linkage="average"):
    D,info=evidence_distances(trials,groups,trim,mode)
    cluster=average_evidence_clusters if linkage=="average" else observed_complete_clusters
    return {str(t):cluster(D,t).tolist() for t in thresholds},info
