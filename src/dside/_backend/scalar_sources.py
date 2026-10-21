"""One scalar for EVERY equal-state pair, retaining its two source labels.

Cross-source ratios are features, not estimates of a shared true damping.
No common-RHS fit or two-dimensional operator statistic is used here.
"""
from itertools import combinations, combinations_with_replacement
from dataclasses import replace
import numpy as np
from .core import green_matrices
from .state_matching import monotone_branches, complete_clusters


def selector_smoothing(raw_trials, smooth_trials, kind="cubic"):
    """Alternative TRAINING-only pre-smoothing, not a parametric RHS model."""
    if kind == "cubic":
        return smooth_trials
    if kind != "poly5":
        raise ValueError("Unknown selector smoother")
    output = []
    for raw, sm in zip(raw_trials, smooth_trials):
        polynomial = np.polynomial.Chebyshev.fit(raw.t-raw.t[0], raw.y, 5)
        derivatives = np.column_stack([polynomial.deriv(k)(sm.t) for k in range(4)])
        output.append(replace(sm, x=derivatives[:,0], derivatives=derivatives))
    return output


def observed_complete_clusters(D, threshold):
    """Complete-link over OBSERVED comparisons, missing comparisons stay unknown.

    A merge needs a measured edge; every measured cross-edge must pass.
    This can link indirectly connected sources. It is a heuristic, not proof
    that dynamics agree on states not jointly observed.
    """
    clusters = [[g] for g in range(len(D))]
    while True:
        candidates = []
        for i, j in combinations(range(len(clusters)), 2):
            edges = D[np.ix_(clusters[i], clusters[j])].ravel()
            finite = edges[np.isfinite(edges)]
            if len(finite) and np.max(finite) <= threshold:
                candidates.append((float(np.max(finite)), i, j))
        if not candidates:
            break
        _, i, j = min(candidates)
        clusters[i] = sorted(clusters[i]+clusters[j])
        del clusters[j]
    labels = np.empty(len(D), int)
    for label, members in enumerate(clusters):
        labels[members] = label
    return labels


def inverse_coordinates(tr, regularization=1e-7):
    """Plugin initial position/velocity, with the correct polynomial correction.

    q=x-x0-v0*t; m=V*x-x0*t; q=G*f-a*m.
    b=R*q, A=R*m. Thus b+a*A=R*G*f, NOT f at positive lambda.
    All initial quantities are obtained from this training spline.
    """
    t = tr.t-tr.t[0]
    _, G = green_matrices(t, [0.], 2)
    _, V = green_matrices(t, [], 1)
    U, s, Vt = np.linalg.svd(G, full_matrices=False)
    lam = regularization*s[0]**2
    R = (Vt.T*(s/(s*s+lam)))@U.T
    q = tr.x-tr.x[0]-tr.derivatives[0, 1]*t
    m = V@tr.x-tr.x[0]*t
    return R@q, R@m, dict(lambda_absolute=float(lam),
        effective_rank=float(np.sum(s*s/(s*s+lam))),
        initial_x=float(tr.x[0]), initial_v=float(tr.derivatives[0, 1]),
        resolution_bias_frobenius=float(np.linalg.norm(R@G-np.eye(len(t)))))


def scalar_pairs(trials, regularization=1e-7, trim=.12, denominator_fraction=.15,
                 states_per_overlap=21, precision_weights=False):
    """Interpolate each monotone branch at exactly matched estimated states.

    Every branch pair contributes equal total mass after rejection; every source
    pair later contributes equal mass. Repeated states on one curve are allowed,
    but times must differ by at least one grid step. No independent-sample claim.
    """
    branches, diagnostics = [], {}
    for tr in trials:
        b, A, diag = inverse_coordinates(tr, regularization)
        diagnostics[tr.name] = diag
        for br in monotone_branches(tr, trim):
            ix = br["ix"]
            branches.append(dict(br, b=b[ix], A=A[ix], time=tr.t[ix],
                                 dt=float(tr.t[1]-tr.t[0])))
    velocity_scale = np.median([np.sqrt(np.mean(
        tr.derivatives[:, 1]**2)) for tr in trials])
    floor = max(1e-6, denominator_fraction*velocity_scale)
    rows, rejected = [], dict(small_denominator=0, same_time=0, no_overlap=0)
    for left, right in combinations(branches, 2):
        if left["group"] > right["group"]:
            left, right = right, left
        lo, hi = max(left["x"][0], right["x"][0]), min(left["x"][-1], right["x"][-1])
        if hi-lo < .01:
            rejected["no_overlap"] += 1
            continue
        grid = np.linspace(lo, hi, states_per_overlap+2)[1:-1]
        bp, ap, tp = [np.interp(grid, left["x"], left[k]) for k in ("b", "A", "time")]
        bq, aq, tq = [np.interp(grid, right["x"], right[k]) for k in ("b", "A", "time")]
        keep = np.abs(ap-aq) >= floor
        rejected["small_denominator"] += int(np.sum(~keep))
        if left["trial"] == right["trial"]:
            distinct = np.abs(tp-tq) > max(left["dt"], right["dt"])
            rejected["same_time"] += int(np.sum(keep & ~distinct))
            keep &= distinct
        size = int(keep.sum())
        mass = (ap-aq)**2 if precision_weights else np.ones(len(grid))
        total_mass = float(mass[keep].sum())
        for k in np.flatnonzero(keep):
            rows.append(dict(g=int(left["group"]), h=int(right["group"]),
                trial_p=left["trial"], trial_q=right["trial"],
                branch_p=int(left["branch"]), branch_q=int(right["branch"]),
                state=float(grid[k]), time_p=float(tp[k]), time_q=float(tq[k]),
                numerator=float(bq[k]-bp[k]), denominator=float(ap[k]-aq[k]),
                statistic=float((bq[k]-bp[k])/(ap[k]-aq[k])), weight=float(mass[k]/total_mass)))
    return rows, dict(curves=diagnostics, rejection=rejected, denominator_floor=floor,
                      velocity_scale=float(velocity_scale), n_branches=len(branches))


def weighted_quantiles(values, weights, probabilities):
    x, inverse = np.unique(values, return_inverse=True)
    w = np.bincount(inverse, weights=weights)
    cdf = (np.cumsum(w)-.5*w)/np.sum(w)
    return np.interp(probabilities, cdf, x)


def distribution_distances(rows, groups=6, mode="pooled", scale=1.):
    """Compare distributions of scalar ratios, with equal source-pair weight.

    atan(T/scale) bounds near-singular tail influence; it is invertible.
    The 41 equally spaced quantiles approximate W1 of transformed distributions.
    'partners' compares g,k versus h,k for common reference sources k.
    """
    probs = (np.arange(41)+.5)/41
    samples = {}
    for g, h in combinations_with_replacement(range(groups), 2):
        rr = [r for r in rows if r["g"] == g and r["h"] == h]
        if rr:
            val = np.arctan(np.array([r["statistic"] for r in rr])/scale)
            weights = np.array([r["weight"] for r in rr])
            weights /= weights.sum()
            samples[g, h] = val, weights
    quantiles = {key: weighted_quantiles(v, w, probs) for key, (v, w) in samples.items()}
    center = float(np.median(np.concatenate(list(quantiles.values())))) if quantiles else np.nan
    signatures = {}
    for g in range(groups):
        local = [value for key, value in samples.items() if g in key]
        if local:
            signatures[g] = weighted_quantiles(np.concatenate([x for x, w in local]),
                np.concatenate([w/len(local) for x, w in local]), probs)
    D = np.full((groups, groups), np.inf)
    np.fill_diagonal(D, 0)
    for g, h in combinations(range(groups), 2):
        if (g, h) not in samples:
            continue
        if mode == "pooled":
            if g in signatures and h in signatures:
                D[g, h] = D[h, g] = np.mean(np.abs(signatures[g]-signatures[h]))
        elif mode == "consensus":
            # Dominant-cohort working assumption, appropriate only when a large
            # related cohort exists. Equal source-pair weights prevent a group
            # with many trials from defining the global reference by itself.
            D[g,h] = D[h,g] = float(np.mean(np.abs(quantiles[g,h]-center)))
        elif mode == "partners":
            distances = []
            for k in range(groups):
                kg, kh = tuple(sorted((g, k))), tuple(sorted((h, k)))
                if kg in quantiles and kh in quantiles and kg != kh:
                    distances.append(float(np.mean(np.abs(quantiles[kg]-quantiles[kh]))))
            if len(distances) >= 2:
                D[g, h] = D[h, g] = float(np.median(distances))
        else:
            raise ValueError("Unknown distribution comparison: "+mode)
    return D, {str(g)+"_"+str(h): q.tolist() for (g, h), q in quantiles.items()}


def scalar_source_clusters(trials, thresholds=(.15, .3, .6), mode="pooled", **kwargs):
    rows, info = scalar_pairs(trials, **kwargs)
    base_mode = mode.replace("_masked", "")
    distances, quantiles = distribution_distances(rows, mode=base_mode)
    clustering = observed_complete_clusters if mode.endswith("_masked") else complete_clusters
    return {str(t): clustering(distances, t).tolist() for t in thresholds}, dict(
        pairs=rows, diagnostics=info, distances=distances.tolist(), quantiles=quantiles)
