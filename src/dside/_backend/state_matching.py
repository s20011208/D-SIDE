"""Source-labelled equal-state elimination for x'' + a_g x' = f_g(x).

No latent simulation values or validation/test trajectories enter the selector.
Inverse matching is an explicitly biased diagnostic. Energy matching avoids the
second derivative and the false assumption a_g == a_h across sources.
Pair rows are dependent constraints, NOT independent statistical replicates.
"""
from dataclasses import replace
from itertools import combinations
from time import perf_counter
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.linalg import qr
from scipy.optimize import lsq_linear
from scipy.spatial.distance import squareform

from .core import GreenModel, green_matrices
from .baselines import IndependentModel


def inverse_signature(tr, regularization=1e-5):
    """Profile polynomial nuisance before regularized inversion.

    P x = P G f - a P V1 x; b=R P x, A=R P V1 x.
    This implies b+a A=R P G f, not f. Repeated states need NOT
    have identical filtered entries. Exposed as a diagnostic/ablation.
    """
    t = tr.t
    H, G = green_matrices(t, np.zeros(1), 2)
    _, V = green_matrices(t, np.zeros(0), 1)
    Q, _ = qr(H, mode="economic")
    project = lambda z: z-Q@(Q.T@z)
    PG = project(G)
    U, s, Vt = np.linalg.svd(PG, full_matrices=False)
    lam = regularization*s[0]**2
    R = (Vt.T*(s/(s*s+lam)))@U.T
    b, A = R@project(tr.x), R@project(V@tr.x)
    return b, A, {"lambda": float(lam), "effective_rank": float(np.sum(s*s/(s*s+lam))),
                  "filter_identity_error": float(np.linalg.norm(R@PG-np.eye(len(t)))),
                  "initial_nuisance": "profiled, not equated to initial derivatives"}


def monotone_branches(tr, trim=.08):
    """Monotone portions of the training spline; endpoints trimmed in time."""
    n = len(tr.t)
    cut = max(1, int(np.ceil(trim*(n-1))))
    idx = np.arange(cut, n-cut)
    if len(idx) < 4:
        return []
    delta = np.diff(tr.x[idx])
    sign = np.sign(delta)
    for j in range(1, len(sign)):
        if sign[j] == 0:
            sign[j] = sign[j-1]
    boundaries = np.r_[0, np.flatnonzero(sign[1:] != sign[:-1])+1, len(idx)-1]
    branches = []
    for k, (left, right) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        ix = idx[left:right+1]
        if len(ix) < 3:
            continue
        if tr.x[ix[0]] > tr.x[ix[-1]]:
            ix = ix[::-1]
        x = tr.x[ix]
        keep = np.r_[True, np.diff(x) > 1e-10]
        ix = ix[keep]
        if len(ix) >= 3 and tr.x[ix[-1]]-tr.x[ix[0]] > 1e-6:
            branches.append({"x": tr.x[ix], "v": tr.derivatives[ix, 1],
                "ix": ix, "trial": tr.name, "branch": k, "group": tr.group})
    return branches


def interval_row(branch, lo, hi):
    """(B,C) with B+a*C = integral_lo^hi f(x) dx, divided by width.

    B=(v(hi)^2-v(lo)^2)/2, C=integral_lo^hi v(x) dx.
    The orientation is increasing position, even for a descending branch.
    """
    x, v = branch["x"], branch["v"]
    if lo < x[0] or hi > x[-1] or hi <= lo:
        return None
    inner = (x > lo)&(x < hi)
    grid = np.r_[lo, x[inner], hi]
    speeds = np.interp(grid, x, v)
    width = hi-lo
    B = .5*(speeds[-1]**2-speeds[0]**2)/width
    C = np.trapz(speeds, grid)/width
    return float(B), float(C)


def _profile(rows, separate_functions):
    """Bounded two-operator least squares after removing bin means.

    A bin contributes total weight one regardless of branch pair count.
    Separate-function fit is nested in the common-function fit, providing
    a within-source lack-of-fit reference. This is not a calibrated test.
    """
    b = np.asarray([r["b"] for r in rows])
    C = np.zeros((len(rows), 2))
    side = np.asarray([r["side"] for r in rows], int)
    C[np.arange(len(rows)), side] = [r["c"] for r in rows]
    bins = np.asarray([r["bin"] for r in rows], int)
    w = np.asarray([1/np.sum(bins == k) for k in bins])
    keys = [(int(k), int(s) if separate_functions else 0) for k, s in zip(bins, side)]
    rb, RC = b.copy(), C.copy()
    for key in sorted(set(keys)):
        ii = np.asarray([k == key for k in keys])
        rb[ii] -= np.average(b[ii], weights=w[ii])
        RC[ii] -= np.average(C[ii], axis=0, weights=w[ii])
    root = np.sqrt(w)
    design, target = root[:, None]*RC, -root*rb
    fit = lsq_linear(design, target, bounds=(.02, 1.6), tol=1e-10, max_iter=100)
    residual = rb+RC@fit.x
    s = np.linalg.svd(design, compute_uv=False)
    rss = float(np.sum(w*residual**2)/max(w.sum(), 1e-12))
    return {"omega": fit.x.tolist(), "rss": rss, "singular_values": s.tolist(),
            "rank": int(np.linalg.matrix_rank(design)), "success": bool(fit.success),
            "fitted_forces": (b+C@fit.x).tolist()}


def pair_signature(trials_g, trials_h, backend="energy", bins=10, trim=.08,
                   inverse_regularization=1e-5):
    """Pairwise force compatibility allowing a different damping per source."""
    if backend not in ("energy", "inverse"):
        raise ValueError("backend must be energy or inverse")
    groups = [trials_g, trials_h]
    branches = [[br for tr in local for br in monotone_branches(tr, trim)] for local in groups]
    if any(not b for b in branches):
        return {"distance": float("inf"), "reason": "no_monotone_training_branch", "rows": []}
    supports = [[min(br["x"][0] for br in bb), max(br["x"][-1] for br in bb)] for bb in branches]
    lo, hi = max(s[0] for s in supports), min(s[1] for s in supports)
    if hi-lo <= .05*min(s[1]-s[0] for s in supports):
        return {"distance": float("inf"), "reason": "insufficient_training_overlap", "rows": []}
    knots = np.linspace(lo, hi, bins+1)
    inv = {}
    inverse_diagnostics = {}
    if backend == "inverse":
        for local in groups:
            for tr in local:
                b, a, diag = inverse_signature(tr, inverse_regularization)
                inv[tr.name] = (b, a)
                inverse_diagnostics[tr.name] = diag
    rows = []
    for k, (left, right) in enumerate(zip(knots[:-1], knots[1:])):
        local_rows = []
        for side, bb in enumerate(branches):
            for branch in bb:
                if backend == "energy":
                    result = interval_row(branch, left, right)
                else:
                    q = .5*(left+right)
                    if branch["x"][0] <= q <= branch["x"][-1]:
                        bb0, aa0 = inv[branch["trial"]]
                        ix = branch["ix"]
                        result = (np.interp(q, branch["x"], bb0[ix]),
                                  np.interp(q, branch["x"], aa0[ix]))
                    else:
                        result = None
                if result is not None:
                    local_rows.append({"bin": k, "side": side, "group": branch["group"],
                        "trial": branch["trial"], "branch": branch["branch"],
                        "lo": float(left), "hi": float(right), "b": float(result[0]), "c": float(result[1])})
        if len(set(r["side"] for r in local_rows)) == 2:
            rows.extend(local_rows)
    if len(set(r["bin"] for r in rows)) < 3:
        return {"distance": float("inf"), "reason": "fewer_than_three_matched_intervals", "rows": rows}
    own, common = _profile(rows, True), _profile(rows, False)
    # Normalize by force magnitude, with a fixed force-unit floor. This assumes
    # the unchanged simulation units; it is not automatically unit invariant.
    scale = max(float(np.sqrt(np.mean(np.square(own["fitted_forces"])))), .2)
    excess = max(common["rss"]-own["rss"], 0.)
    distance = np.sqrt(excess)/scale
    return {"distance": float(distance), "reason": "computed", "rows": rows,
            "own": own, "common": common, "force_scale": scale, "excess": excess,
            "overlap": [float(lo), float(hi)], "inverse_diagnostics": inverse_diagnostics}


def training_distances(trials, backend="energy", **kwargs):
    groups = max(t.group for t in trials)+1
    D = np.full((groups, groups), np.inf)
    np.fill_diagonal(D, 0.)
    pairs = []
    for g, h in combinations(range(groups), 2):
        result = pair_signature([t for t in trials if t.group == g],
                                [t for t in trials if t.group == h], backend, **kwargs)
        D[g, h] = D[h, g] = result["distance"]
        pairs.append(dict(result, g=g, h=h))
    return D, pairs


def complete_clusters(distances, threshold):
    """Complete linkage prevents one compatible bridge merging two bad ends.

    Infinity means lack of evidence/overlap, not demonstrated heterogeneity.
    No known number of clusters, simulator labels, or majority size is used.
    """
    D = np.asarray(distances)
    if D.ndim != 2 or D.shape[0] != D.shape[1] or not np.allclose(D, D.T):
        raise ValueError("symmetric square distance matrix required")
    if len(D) == 1:
        return np.zeros(1, dtype=int)
    safe = np.where(np.isfinite(D), D, 1e8)
    tree = linkage(squareform(safe, checks=True), method="complete")
    labels = fcluster(tree, t=threshold, criterion="distance")-1
    return labels.astype(int)


class MatchedSourceGreen:
    """Cluster first using training data, fit partial-sharing dynamics inside."""
    def __init__(self, labels, lambda_u=.002, operator_shrinkage=0., **kwargs):
        self.labels = np.asarray(labels, int)
        self.lambda_u = float(lambda_u)
        self.operator_shrinkage = float(operator_shrinkage)
        self.kwargs = kwargs

    def fit(self, trials, independent=None):
        tic = perf_counter()
        if independent is None:
            independent = IndependentModel(**self.kwargs,
                lambda_u=self.kwargs.get("lambda0", .0002)).fit(trials)
        self.models, self.target_indices = [], []
        self.selected_sources, self.fallback, self.outer_log = [], [], []
        self.inner_diagnostics = []
        cache = {}
        for g in range(len(self.labels)):
            members = np.flatnonzero(self.labels == self.labels[g]).tolist()
            self.selected_sources.append([h for h in members if h != g])
            if len(members) == 1:
                model, index = independent.models[g], 0
                self.fallback.append(True)
            else:
                key = tuple(members)
                if key not in cache:
                    mapping = {h: k for k, h in enumerate(members)}
                    local = [replace(tr, group=mapping[tr.group]) for tr in trials if tr.group in mapping]
                    # Similar forcing is not an assumption of similar damping.
                    options = dict(self.kwargs, omega_shrinkage=self.operator_shrinkage)
                    cache[key] = GreenModel(**options, lambda_u=self.lambda_u).fit(local)
                    self.outer_log.extend(cache[key].outer_log)
                    self.inner_diagnostics.append(cache[key].inner_diagnostics)
                model, index = cache[key], members.index(g)
                self.fallback.append(False)
            self.models.append(model)
            self.target_indices.append(index)
        self.omega = np.vstack([m.omega[i] for m, i in zip(self.models, self.target_indices)])
        self.elapsed = perf_counter()-tic
        return self

    def rhs(self, group, x):
        return self.models[group].rhs(self.target_indices[group], x)

    def predict(self, group, t, initial):
        return self.models[group].predict(self.target_indices[group], t, initial)


class GroupChoice:
    """Group-wise validation choice; exact independent fallback when alpha=0."""
    def __init__(self, independent, models, decisions):
        self.independent, self.models, self.decisions = independent, models, decisions
        self.omega = np.vstack([(1-decisions[g]["alpha"])*independent.omega[g]+
            decisions[g]["alpha"]*models[decisions[g]["chosen"]].omega[g]
            for g in range(len(independent.omega))])

    def rhs(self, group, x):
        d = self.decisions[group]
        if d["alpha"] == 0:
            return self.independent.rhs(group, x)
        return (1-d["alpha"])*self.independent.rhs(group, x)+d["alpha"]*self.models[d["chosen"]].rhs(group, x)

    def predict(self, group, t, initial):
        from .graph_green import simulate_field
        d = self.decisions[group]
        if d["alpha"] == 0:
            return self.independent.predict(group, t, initial)
        if d["alpha"] == 1:
            return self.models[d["chosen"]].predict(group, t, initial)
        return simulate_field(self, group, t, initial)
