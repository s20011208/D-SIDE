"""Runnable comparators. Lifted GP is a simple posterior-mean baseline, not BN-MEODE reproduction."""
from dataclasses import replace
from time import perf_counter
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import cho_factor, cho_solve
from .core import GreenModel


class IndependentModel:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, trials):
        tic = perf_counter()
        self.models = []
        for g in range(max(t.group for t in trials)+1):
            local = [replace(t, group=0) for t in trials if t.group == g]
            self.models.append(GreenModel(**self.kwargs).fit(local))
        self.omega = np.vstack([m.omega for m in self.models])
        self.outer_log = [dict(entry, group=g) for g, m in enumerate(self.models) for entry in m.outer_log]
        self.elapsed = perf_counter()-tic
        self.inner_diagnostics = [m.inner_diagnostics for m in self.models]
        return self

    def rhs(self, group, x):
        return self.models[group].rhs(0, x)

    def predict(self, group, t, initial):
        return self.models[group].predict(0, t, initial)


class LiftedGP:
    """K=2 first-order phase-space GP mean: x'=v, v'=h(x,v).

    Independent groups, Matérn kernel, finite training subset; gradient-based
    acceleration targets. No fitted separate operator, hence no omega metric.
    """
    def fit(self, trials):
        tic = perf_counter()
        self.models = []
        for g in range(max(t.group for t in trials)+1):
            local = [t for t in trials if t.group == g]
            z = np.concatenate([t.derivatives[:, :2] for t in local])
            y = np.concatenate([t.derivatives[:, 2] for t in local])
            idx = np.linspace(0, len(z)-1, min(len(z), 100), dtype=int)
            z, y = z[idx], y[idx]
            scale = np.maximum(z.std(axis=0), .2)
            zz = z/scale
            K = self.kernel(zz, zz)
            beta = cho_solve(cho_factor(K+.03*np.eye(len(K))), y)
            self.models.append((zz, scale, beta))
        self.elapsed = perf_counter()-tic
        self.outer_log = []
        self.inner_diagnostics = []
        return self

    @staticmethod
    def kernel(z, zz):
        d = np.sqrt(np.sum((np.asarray(z)[:, None]-zz[None])**2, axis=2))
        r = np.sqrt(5)*d
        return (1+r+r*r/3)*np.exp(-r)

    def predict(self, group, t, initial):
        zz, scale, beta = self.models[group]
        def ode(_, z):
            return [z[1], (self.kernel((z/scale)[None], zz)@beta)[0]]
        sol = solve_ivp(ode, (t[0], t[-1]), initial, t_eval=t, rtol=2e-8, atol=2e-10, method="DOP853")
        if not sol.success or len(sol.t) != len(t):
            raise RuntimeError("lifted GP integration failed")
        return sol.y[0]
