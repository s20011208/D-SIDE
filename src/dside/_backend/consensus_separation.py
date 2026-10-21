"""Versioned simulation intervention; no change to scalar selection.

Calibrated launch speeds are a deliberately controlled simulation design.
They use the generating ODE to make trajectories cover comparable position
ranges, not true labels or downstream clustering/prediction outcomes. Actual
initial states are removed before estimation. Old generators remain intact.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from .core import Trial
from .consensus_design import generate as previous_generate
from .global_consensus import rhs_truth, damping_truth


def solve_truth(group, damping, initial, duration):
    sol = solve_ivp(lambda t, z: [z[1], rhs_truth(group, z[0])-damping*z[1]],
                    (0., duration), initial, method='DOP853', dense_output=True,
                    rtol=1e-10, atol=1e-12)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol


def calibrated_initials(config, seed):
    assignment = np.random.default_rng(np.random.SeedSequence([seed, 8299])).permutation(8)
    targets = np.asarray(config['target_maxima'])[assignment]
    x0, duration = config['common_initial_position'], config['duration']
    grid = np.linspace(0., duration, 601)
    initials = []
    for g, a in enumerate(damping_truth(config)):
        def discrepancy(v):
            return float(solve_truth(g, a, [x0, v], duration).sol(grid)[0].max()-targets[g])
        lo, hi = 0., 2.
        if discrepancy(lo) >= 0:
            raise ValueError('Requested maximum is already exceeded with zero launch speed')
        while discrepancy(hi) < 0:
            hi *= 2
            if hi > 32:
                raise RuntimeError('Could not bracket calibrated launch speed')
        v = brentq(discrepancy, lo, hi, xtol=1e-10)
        initials.append([x0, float(v)])
    return np.asarray(initials), targets


def generate(config, seed, split='train'):
    if split != 'train' or config['initial_design'] != 'calibrated_state_range':
        return previous_generate(config, seed, split)
    initials, _ = calibrated_initials(config, seed)
    duration = config['duration']
    answer = []
    for g, a in enumerate(damping_truth(config)):
        rng = np.random.default_rng(np.random.SeedSequence([seed, 8201, g, 0, 0]))
        rng.uniform(-1.25, 1.25, 2)  # retain matched old time/noise RNG streams
        sol = solve_truth(g, a, initials[g], duration)
        times = np.r_[0., np.sort(rng.uniform(0., duration, config['observations']-2)), duration]
        y = sol.sol(times)[0]+rng.normal(0., config['noise_sd'], len(times))
        tt = np.linspace(0., duration, 101)
        answer.append(Trial(g, f'train-g{g}-t0', times, y, initials[g], tt, sol.sol(tt)[0]))
    return answer
