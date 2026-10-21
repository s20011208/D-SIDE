"""Controlled multi-speed initial conditions; the selector is unchanged.

All eight sources receive the same starting position and one of eight distinct
positive launch speeds. Assignment is a seeded random permutation independent
of mechanism labels. Observation times/noise match the old random-initial-state
design at the same seed; held-out data are exactly unchanged.
"""
import numpy as np
from scipy.integrate import solve_ivp
from .core import Trial
from .global_consensus import generate as generate_random, rhs_truth, damping_truth


def generate(config,seed,split='train'):
    if split!='train' or config.get('initial_design')!='common_position_distinct_speeds':
        return generate_random(config,seed,split)
    assignment=np.random.default_rng(np.random.SeedSequence([seed,8299])).permutation(8)
    speeds=np.array(config['launch_speeds'])[assignment]
    damping=damping_truth(config);duration=config['duration'];answer=[]
    for g in range(8):
        rng=np.random.default_rng(np.random.SeedSequence([seed,8201,g,0,0]))
        rng.uniform(-1.25,1.25,2)  # consume the old initial-state draw, preserving times/noise
        initial=np.array([config['common_initial_position'],speeds[g]])
        sol=solve_ivp(lambda t,z:[z[1],rhs_truth(g,z[0])-damping[g]*z[1]],(0,duration),initial,
                     method='DOP853',dense_output=True,rtol=1e-10,atol=1e-12)
        if not sol.success:raise RuntimeError(sol.message)
        times=np.r_[0.,np.sort(rng.uniform(0,duration,config['observations']-2)),duration]
        y=sol.sol(times)[0]+rng.normal(0,config['noise_sd'],len(times))
        tt=np.linspace(0,duration,101)
        answer.append(Trial(g,f'train-g{g}-t0',times,y,initial,tt,sol.sol(tt)[0]))
    return answer
