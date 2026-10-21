"""Bounded source-pair evidence mass for the frozen global-mixture selector.

The weight-concentration count below is NOT a number of independent samples.
Rows from one/two curves are dependent. Keep the original 21-state sampling
protocol fixed; exact duplicate rows add no information. This is a transparent
reliability heuristic, not a calibrated hypothesis test or uncertainty bound.
The original global_consensus.py remains byte-for-byte frozen.
"""
from itertools import combinations
import numpy as np
from scipy.optimize import minimize
from scipy.special import ndtr
from .global_consensus import mixture_parts, correlation_partition


def pair_mass(rows, saturation_count=21.):
    """Positive within-pair weights -> bounded between-pair mass.

    n_balanced = (sum w)^2 / sum(w^2) <= retained row count.
    mass = n_balanced / (saturation_count + n_balanced).
    The constant 21 is the existing number of sampled states per branch overlap,
    chosen before evaluating this revision, not selected on the old failures.
    """
    if not np.isfinite(saturation_count) or saturation_count <= 0:
        raise ValueError('saturation_count must be positive and finite')
    if not rows:
        return np.empty(0), dict(count=0, balanced_count=0., mass=0.)
    weights = np.array([r['weight'] for r in rows], float)
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError('Retained row weights must be finite and positive')
    weights /= weights.max()  # avoid overflowing squares for large precision
    local = weights / weights.sum()
    count = float(1. / np.sum(local**2))
    return local, dict(count=len(rows), balanced_count=count,
                      mass=float(count/(saturation_count+count)))


def unique_cross_rows(rows):
    """Remove exact records duplicated by replay; preserve distinct match points.

    This does not turn neighbouring grid points into independent observations.
    Actual production rows have curve, branch, state and time identifiers.
    """
    pairs, seen = {}, set()
    fields = ('g', 'h', 'trial_p', 'trial_q', 'branch_p', 'branch_q',
              'state', 'time_p', 'time_q', 'statistic', 'weight', 'variance_z')
    for row in rows:
        g, h = row['g'], row['h']
        if g == h:
            continue
        if g > h:
            raise ValueError('Source pairs must use canonical g < h order')
        if not np.isfinite(row['statistic']):
            raise ValueError('Nonfinite scalar statistic')
        key = tuple(row.get(k) for k in fields)
        if key in seen:
            continue
        seen.add(key)
        pairs.setdefault((g, h), []).append(row)
    return pairs


def select_reliable_mixture(rows, groups, saturation_count=21.,
                            weight_density=True, weight_edges=True):
    """Apply the SAME reliability mass to the global density and graph evidence.

    Two switches permit prespecified density-only / graph-only ablations.
    Peak family, bounds, nine starts, partition objective and single-core rule
    are unchanged from V3. Neither true labels nor validation/test data enter.
    """
    pairs = unique_cross_rows(rows)
    if not pairs:
        return dict(status='no_cross_source_evidence', core=[], rows=[],
                    clusters=[[g] for g in range(groups)], pair_reliability=[])
    if any(g < 0 or h >= groups for g, h in pairs):
        raise ValueError('Source label outside specified group range')
    local, info = {}, {}
    for pair, rr in sorted(pairs.items()):
        local[pair], info[pair] = pair_mass(rr, saturation_count)
    density_mass = {p: info[p]['mass'] if weight_density else 1. for p in pairs}
    total_mass = sum(density_mass.values())
    exported, pair_indices = [], {}
    for pair, rr in sorted(pairs.items()):
        start = len(exported)
        for r, w in zip(rr, local[pair]):
            exported.append(dict(r, z=float(np.arctan(r['statistic'])),
                pair_weight=float(w), reliability=info[pair]['mass'],
                global_weight=float(w*density_mass[pair]/total_mass)))
        pair_indices[pair] = np.arange(start, len(exported))
    z = np.array([r['z'] for r in exported])
    w = np.array([r['global_weight'] for r in exported])
    fits = [minimize(lambda v: -float(w@mixture_parts(z, v)[0]),
                     [initial_mu, .25, .6], method='L-BFGS-B',
                     bounds=[(-1.5, 1.5), (.04, .7), (.05, .95)])
            for initial_mu in np.linspace(-1.2, 1.2, 9)]
    fit = min(fits, key=lambda r: r.fun)
    mu, sigma, proportion = fit.x
    log_mix, gamma, _ = mixture_parts(z, fit.x)
    normalizer = ndtr((np.pi/2-mu)/sigma)-ndtr((-np.pi/2-mu)/sigma)
    radius = sigma*np.sqrt(max(0., 2*np.log(np.pi/(sigma*np.sqrt(2*np.pi)*normalizer))))
    interval = [max(-np.pi/2, float(mu-radius)), min(np.pi/2, float(mu+radius))]
    support = np.full((groups, groups), np.nan)
    evidence = np.full_like(support, np.nan)
    np.fill_diagonal(support, proportion)
    np.fill_diagonal(evidence, 0.)
    audits = []
    for pair, ii in pair_indices.items():
        mean = float(local[pair]@gamma[ii])
        strength = info[pair]['mass'] if weight_edges else 1.
        edge = strength*(mean-proportion)
        support[pair] = support[pair[::-1]] = mean
        evidence[pair] = evidence[pair[::-1]] = edge
        audits.append(dict(g=pair[0], h=pair[1], **info[pair],
            global_mass=float(density_mass[pair]/total_mass),
            graph_mass=float(strength), support=mean,
            unweighted_enrichment=float(mean-proportion), edge_evidence=float(edge)))
    partition, cluster_fit = correlation_partition(evidence, threshold=0.)
    def score(cluster):
        return sum(evidence[g, h] for g, h in combinations(cluster, 2)
                   if np.isfinite(evidence[g, h]))
    eligible = [c for c in partition if len(c) >= 3]
    core = max(eligible, key=lambda c: (score(c), len(c))) if eligible else []
    clusters = ([core] if core else [])+[[g] for g in range(groups) if g not in core]
    for row, q in zip(exported, gamma):
        row.update(responsibility=float(q), enrichment=float(q-proportion),
                   inside=bool(q >= proportion))
    return dict(status='ok', core=core, clusters=clusters, partition=partition,
        support=support, enrichment=support-proportion, edge_evidence=evidence,
        rows=exported, interval=interval, pair_reliability=audits,
        peak=dict(mu=float(mu), sigma=float(sigma), proportion=float(proportion),
                  weighted_log_likelihood=float(w@log_mix),
                  uniform_log_likelihood=float(-np.log(np.pi))),
        mixture_starts=[dict(success=bool(r.success), objective=float(r.fun),
                            parameters=r.x.tolist()) for r in fits],
        available_pairs=len(pairs), clustering=cluster_fit,
        weighting=dict(saturation_count=float(saturation_count),
                       density=bool(weight_density), edges=bool(weight_edges)),
        criterion='sum within-partition reliability*(mean responsibility-global proportion); main block only')
