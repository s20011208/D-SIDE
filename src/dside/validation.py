"""Independent arithmetic checks on the compact published experiment records."""
from pathlib import Path
import numpy as np
import pandas as pd
from .io import read

# Absolute tolerances for the fixed, dimensionless illustration's full refit.
# CPU-dispatched BLAS kernels can perturb the nonconvex fit. The controlled
# measurements and the separate, stricter checkpoint check are documented in
# docs/numerical_portability.md. These are validation rules, not fit settings.
REFIT_PREDICTION_ATOL = 1e-6
REFIT_TRIAL_MSE_ATOL = 1e-7
REFIT_MEAN_MSE_ATOL = 1e-8


def audit_results(root):
    root = Path(root)
    if not (root/'original/group_comparison.csv').exists():
        raise FileNotFoundError('Published records live in the repository checkout. Use dside audit-results --root /path/to/D-SIDE/results/published')
    for name in ['original', 'adapted_retrospective', 'adapted_fresh']:
        path = root/name
        group = pd.read_csv(path/'group_comparison.csv')
        selections = pd.read_csv(path/'source_selection.csv')
        summary = read(path/'summary.json')['methods']
        for method, ref in summary.items():
            g = group[group.method == method]
            if not np.isclose(g.mse.mean(), ref['all_mse'], rtol=1e-12, atol=1e-14):
                raise AssertionError(name+'/'+method+' MSE differs')
            nt = g.mse > 1.01*g.independent_mse
            if int(nt.sum()) != ref['nt']: raise AssertionError(name+'/'+method+' negative-transfer count differs')
            s = selections[selections.method == method]
            if len(s) and int(s.exact.sum()) != ref['exact']: raise AssertionError(name+'/'+method+' exact grouping differs')
        print('Verified %s: %d source-level method records' % (name, len(group)))
    return True
