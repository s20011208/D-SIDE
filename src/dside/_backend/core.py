"""Autonomous scalar RHS, normalized integer-order operator with no D^0 term.

State derivative inputs are excluded from the RHS to prevent absorption of
operator coefficients. This restriction is substantive, not ridge identification.
"""
from dataclasses import dataclass, field
from time import perf_counter
import numpy as np
from scipy.linalg import cho_factor, cho_solve, expm, qr
from math import factorial
from scipy.interpolate import UnivariateSpline
from scipy.integrate import solve_ivp
from scipy.optimize import minimize


@dataclass
class Trial:
    group: int
    name: str
    t: np.ndarray
    y: np.ndarray
    initial: np.ndarray
    truth_t: np.ndarray = None
    truth_x: np.ndarray = None


@dataclass
class SmoothTrial:
    group: int
    name: str
    t: np.ndarray
    x: np.ndarray
    derivatives: np.ndarray
    diagnostics: dict = field(default_factory=dict)


def smooth_trial(trial, grid=61, noise_sd=0.035):
    """Spline on this training trial only; never supplied held-out samples.

    FITPACK cubic residual budget n*sigma^2 is a declared noise-based choice,
    not an automatically estimated bandwidth or a claim of optimal smoothing.
    """
    t, idx = np.unique(np.asarray(trial.t, float), return_index=True)
    y = np.asarray(trial.y, float)[idx]
    if len(t) < 5 or np.any(~np.isfinite(t)) or np.any(~np.isfinite(y)):
        raise ValueError("need five distinct finite training observations")
    spl = UnivariateSpline(t, y, k=3, s=len(t)*noise_sd**2, ext=2)
    q = np.linspace(t[0], t[-1], grid)
    derivatives = np.column_stack([spl(q, nu=k) for k in range(4)])
    return SmoothTrial(trial.group, trial.name, q-q[0], derivatives[:, 0], derivatives,
                       {"n": len(t), "residual_budget": len(t)*noise_sd**2,
                        "residual_mse": float(np.mean((spl(t)-y)**2)),
                        "knots": len(spl.get_knots()), "noise_sd": noise_sd,
                        "boundary": "FITPACK boundary behaviour; no explicit natural boundary constraint; no future data"})


def companion(omega, order):
    omega = np.asarray(omega, float)
    if omega.shape != (order-1,):
        raise ValueError("omega contains coefficients of D^1,...,D^(K-1)")
    A = np.zeros((order, order))
    if order > 1:
        A[:-1, 1:] = np.eye(order-1)
        A[-1, 1:] = -omega
    return A


def green_matrices(t, omega, order):
    """Causal impulse response and unknown initial-state design; trapezoids.

    Uniform grid required. Complexity O(J*K^3 + J^2), independent of data values.
    """
    t = np.asarray(t, float)
    if t.ndim != 1 or len(t) < 2 or np.any(~np.isfinite(t)) or np.any(np.diff(t) <= 0):
        raise ValueError("integration times must be finite and strictly increasing")
    dt = np.diff(t)
    if len(t) < order+1 or not np.allclose(dt, dt[0]):
        raise ValueError("uniform integration grid required")
    A = companion(omega, order)
    E = expm(A*dt[0])
    H = np.empty((len(t), order))
    power = np.eye(order)
    for j in range(len(t)):
        H[j] = power[0]
        power = power @ E
    lag = np.subtract.outer(np.arange(len(t)), np.arange(len(t)))
    W = H[np.maximum(lag, 0), -1]*dt[0]
    W[lag < 0] = 0
    W[:, 0] *= 0.5
    W[np.diag_indices(len(t))] *= 0.5
    W[0] = 0
    return H, W


def green_apply(t, omega, order, phi):
    """Apply the SAME trapezoidal Green operator without a J-by-J matrix.

    Recurrence Z_j=E Z_(j-1)+dt/2*(E b phi_(j-1)+b phi_j).
    O(J*K^2*r) work and O(J*r+J*K) storage rather than O(J^2).
    """
    t, phi = np.asarray(t, float), np.asarray(phi, float)
    dt = np.diff(t)
    if len(t) < order+1 or not np.isfinite(t).all() or np.any(dt <= 0) or not np.allclose(dt, dt[0]):
        raise ValueError("strictly increasing finite uniform grid required")
    if phi.ndim != 2 or phi.shape[0] != len(t):
        raise ValueError("feature rows must match quadrature times")
    E = expm(companion(omega, order)*dt[0])
    H, design = np.empty((len(t), order)), np.zeros_like(phi)
    power, Z = np.eye(order), np.zeros((order, phi.shape[1]))
    b = np.zeros(order); b[-1] = 1
    for j in range(len(t)):
        H[j] = power[0]
        power = power @ E
        if j:
            Z = E@Z + .5*dt[0]*(np.outer(E@b, phi[j-1])+np.outer(b, phi[j]))
            design[j] = Z[0]
    return H, design


def matern52(x, z, length=0.9):
    d = np.abs(np.asarray(x).reshape(-1, 1)-np.asarray(z).reshape(1, -1))/length
    r = np.sqrt(5)*d
    return (1+r+r*r/3)*np.exp(-r)


class Nystrom:
    def __init__(self, x, rank=16):
        x = np.asarray(x, float).ravel()
        self.center = float(np.median(x))
        self.scale = max(float(np.std(x)), 0.15)
        xx = (x-self.center)/self.scale
        self.rank = min(rank, int(np.ceil(2*np.sqrt(len(x)))), len(x))
        self.z = np.quantile(xx, np.linspace(0.01, 0.99, self.rank))
        K = matern52(self.z, self.z)
        eig, vec = np.linalg.eigh(K)
        self.whiten = (vec/np.sqrt(np.maximum(eig, 1e-9))) @ vec.T

    def __call__(self, x):
        return matern52((np.asarray(x).ravel()-self.center)/self.scale, self.z) @ self.whiten


def group_penalty(groups, lambda0, lambda_u, pi=None):
    pi = np.ones(groups)/groups if pi is None else np.asarray(pi, float)
    if not np.isclose(pi.sum(), 1) or np.any(pi <= 0):
        raise ValueError("positive group weights must sum to one")
    # lambda0 ||sum pi B||^2 + lambda_u sum pi ||B_g-sum pi B||^2
    return lambda_u*np.diag(pi)+(lambda0-lambda_u)*np.outer(pi, pi)


def profile_trial(tr, phi, omega, order, matching="green"):
    J = len(tr.t)
    w = np.ones(J)/(J-1)
    w[[0, -1]] *= 0.5
    if matching == "gradient":
        # Comparator directly estimates order-K derivative; no profiled IC.
        target = tr.derivatives[:, order].copy()
        for j, val in enumerate(omega, 1):
            target += val*tr.derivatives[:, j]
        return phi*np.sqrt(w[:, None]), target*np.sqrt(w), None
    if matching == "integral":
        # Polynomial Green identity (original GSFM constant-coefficient form).
        H = np.column_stack([tr.t**k/factorial(k) for k in range(order)])
        _, W = green_matrices(tr.t, np.zeros(order-1), order)
        target = tr.x.copy()
        for j, a in enumerate(omega, 1):
            _, V = green_matrices(tr.t, np.zeros(order-j-1), order-j)
            target += a*(V@tr.x)
        root = np.sqrt(w)
        Q, _ = qr(root[:, None]*H, mode="economic")
        AA, yy = root[:, None]*(W@phi), root*target
        return AA-Q@(Q.T@AA), yy-Q@(Q.T@yy), None
    if matching == "weak":
        from numpy.polynomial import Polynomial
        T = tr.t[-1]
        rows, targets = [], []
        p = Polynomial([1, 0, -1])**(order+1)
        for width in [0.2*T, 0.35*T]:
            for center in np.linspace(width, T-width, 6):
                z = (tr.t-center)/width
                mask = np.abs(z) <= 1
                test = [np.where(mask, p.deriv(j)(z)/width**j, 0) for j in range(order+1)]
                y = (-1)**order*np.trapz(test[order]*tr.x, tr.t)
                for j, a in enumerate(omega, 1):
                    y += a*(-1)**j*np.trapz(test[j]*tr.x, tr.t)
                rows.append(np.trapz(test[0][:, None]*phi, tr.t, axis=0))
                targets.append(y)
        return np.asarray(rows)/np.sqrt(len(rows)), np.asarray(targets)/np.sqrt(len(rows)), None
    if matching == "green_recursive":
        H, design = green_apply(tr.t, omega, order, phi)
        W = None
    else:
        H, W = green_matrices(tr.t, omega, order)
        design = W @ phi
    root = np.sqrt(w)
    Q, _ = qr(root[:, None]*H, mode="economic")
    yy = root*tr.x
    AA = root[:, None]*design
    # Orthogonal projection removes all K unknown initial-state coordinates.
    return AA-Q@(Q.T@AA), yy-Q@(Q.T@yy), (H, W)


def solve_inner(blocks, groups, rank, lambda0, lambda_u, pool=False, diagnostics=True):
    """Equivalent convex quadratic solve in total group coefficients B.

    Mean/deviations extracted afterwards satisfy exact functional centering.
    Positive lambda0 and lambda_u make the coefficient Hessian SPD.
    """
    if lambda0 <= 0 or lambda_u <= 0:
        raise ValueError("positive penalties required for unique coefficients")
    count = np.bincount([b[0] for b in blocks], minlength=groups)
    if np.any(count == 0):
        raise ValueError("every fitted group needs training data")
    dim = rank if pool else groups*rank
    gram = np.zeros((dim, dim))
    rhs = np.zeros(dim)
    for g, D, y in blocks:
        sl = slice(0, rank) if pool else slice(g*rank, (g+1)*rank)
        weight = 1/(groups*count[g])
        gram[sl, sl] += weight*(D.T@D)
        rhs[sl] += weight*(D.T@y)
    penalty = lambda0*np.eye(rank) if pool else np.kron(group_penalty(groups, lambda0, lambda_u), np.eye(rank))
    hessian = gram+penalty
    beta = cho_solve(cho_factor(hessian, lower=True), rhs)
    objective = float(beta @ penalty @ beta)
    rss = 0.0
    for g, D, y in blocks:
        b = beta if pool else beta[g*rank:(g+1)*rank]
        rss += float(np.sum((D@b-y)**2)/(groups*count[g]))
    B = np.tile(beta, (groups, 1)) if pool else beta.reshape(groups, rank)
    mean = B.mean(axis=0)
    return B, rss+objective, {"normal_equation_residual": float(np.linalg.norm(hessian@beta-rhs)),
                            "centering_residual": float(np.linalg.norm((B-mean).mean(axis=0))),
                            "hessian_min_eigenvalue": float(np.linalg.eigvalsh(hessian)[0]) if diagnostics else None,
                            "shape": list(hessian.shape), "matching_rss": rss}


class GreenModel:
    def __init__(self, order=2, lambda0=0.0002, lambda_u=0.002, rank=16,
                 pool=False, shared_operator=False, omega_shrinkage=0.0005,
                 starts=2, maxiter=35, matching="green", known_omega=None,
                 operator_bounds=None):
        self.order, self.lambda0, self.lambda_u, self.rank = order, lambda0, lambda_u, rank
        self.pool, self.shared_operator = pool, shared_operator
        self.omega_shrinkage, self.starts, self.maxiter = omega_shrinkage, starts, maxiter
        self.matching, self.known_omega = matching, known_omega
        self.operator_bounds = operator_bounds
        if operator_bounds is not None:
            b = np.asarray(operator_bounds, float)
            if b.shape != (order-1, 2) or not np.isfinite(b).all() or np.any(b[:, 0] >= b[:, 1]):
                raise ValueError("operator_bounds must give one finite increasing interval per coefficient")

    def fit(self, trials):
        tic = perf_counter()
        self.groups = max(t.group for t in trials)+1
        observed_n = sum(t.diagnostics.get("n", len(t.x)) for t in trials)
        growth_rank = int(np.ceil(2*np.sqrt(observed_n)))
        effective_rank = growth_rank if self.rank == 0 else min(self.rank, growth_rank)
        self.features = Nystrom(np.concatenate([t.x for t in trials]), effective_rank)
        phi = [self.features(t.x) for t in trials]
        self.outer_log = []
        nc = self.order-1
        nops = 1 if self.shared_operator else self.groups

        def unpack(v):
            a = np.asarray(v).reshape(nops, nc)
            return np.tile(a, (self.groups, 1)) if nops == 1 else a

        def inner(omega, diagnostics=False):
            blocks = []
            for tr, ph in zip(trials, phi):
                D, y, _ = profile_trial(tr, ph, omega[tr.group], self.order, self.matching)
                blocks.append((tr.group, D, y))
            return solve_inner(blocks, self.groups, self.features.rank, self.lambda0, self.lambda_u, self.pool, diagnostics)

        def fun(v):
            omega = unpack(v)
            # A ridge penalty selects stable sharing; it does not prove identification.
            return inner(omega)[1]+self.omega_shrinkage*float(np.mean((omega-omega.mean(axis=0))**2))

        if nc == 0:
            self.omega = np.empty((self.groups, 0))
        elif self.known_omega is not None:
            self.omega = np.asarray(self.known_omega).reshape(self.groups, nc)
        else:
            bounds_one = [(0.02, 1.6)] if self.order == 2 else [(0.5, 4.0), (1.0, 4.0)]
            if self.operator_bounds is not None:
                bounds_one = [tuple(b) for b in self.operator_bounds]
            bounds = bounds_one*nops
            best = None
            for start in range(self.starts):
                frac = [0.25, 0.7, 0.5][start % 3]
                x0 = np.array([lo+frac*(hi-lo) for lo, hi in bounds])
                res = minimize(fun, x0, method="L-BFGS-B", bounds=bounds,
                               options={"maxiter": self.maxiter, "ftol": 1e-10, "gtol": 1e-7})
                self.outer_log.append({"start": start, "objective": float(res.fun), "success": bool(res.success),
                                       "message": str(res.message), "nit": int(res.nit), "nfev": int(res.nfev),
                                       "omega": unpack(res.x).tolist()})
                if np.isfinite(res.fun) and (best is None or res.fun < best.fun):
                    best = res
            if best is None:
                raise RuntimeError("all outer starts failed numerically")
            self.omega = unpack(best.x)
        self.B, self.objective, self.inner_diagnostics = inner(self.omega, True)
        self.mean_beta = self.B.mean(axis=0)
        self.deviation_beta = self.B-self.mean_beta
        self.elapsed = perf_counter()-tic
        self.initial_states = {}
        if self.matching in ("green", "green_recursive"):
            for tr, ph in zip(trials, phi):
                if self.matching == "green_recursive":
                    H, response = green_apply(tr.t, self.omega[tr.group], self.order, ph)
                    forcing = response@self.B[tr.group]
                else:
                    H, W = green_matrices(tr.t, self.omega[tr.group], self.order)
                    forcing = W@ph@self.B[tr.group]
                w = np.ones(len(tr.t)); w[[0, -1]] = 0.5
                alpha = np.linalg.lstsq(H*np.sqrt(w[:, None]),
                                        (tr.x-forcing)*np.sqrt(w), rcond=None)[0]
                self.initial_states[tr.name] = alpha.tolist()
        return self

    def rhs(self, group, x):
        return self.features(x) @ self.B[group]

    def predict(self, group, t, initial):
        t = np.asarray(t, float)
        initial = np.asarray(initial, float)
        if initial.shape != (self.order,):
            raise ValueError("prediction requires K independently supplied initial coordinates")
        def ode(_, z):
            return np.r_[z[1:], self.rhs(group, [z[0]])[0]-np.dot(self.omega[group], z[1:])]
        sol = solve_ivp(ode, (float(t[0]), float(t[-1])), initial, t_eval=t,
                        rtol=2e-8, atol=2e-10, method="DOP853")
        if not sol.success or sol.y.shape[1] != len(t) or np.any(~np.isfinite(sol.y)):
            raise RuntimeError("free simulation failed: "+sol.message)
        return sol.y[0]
