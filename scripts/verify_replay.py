"""Compare a complete CLI refit to the historical prediction fixture."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from dside.io import read
from dside.validation import REFIT_PREDICTION_ATOL, REFIT_TRIAL_MSE_ATOL, REFIT_MEAN_MSE_ATOL


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--run', required=True); args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    case = Path(args.run)/'illustration/seed_2026103000'
    ref = root/'tests/fixtures/seed_2026103000'
    count = 0; max_error = 0.
    with np.load(case/'predictions.npz', allow_pickle=False) as actual, np.load(ref/'predictions.npz', allow_pickle=False) as expected:
        assert set(actual.files) == set(expected.files)
        for name in actual.files:
            error = float(np.max(abs(actual[name]-expected[name]))); max_error = max(max_error, error)
            np.testing.assert_allclose(actual[name], expected[name], atol=REFIT_PREDICTION_ATOL, rtol=0); count += 1
    assert read(case/'density_only_selection.json')['core'] == read(ref/'selection_reference.json')['core']
    a = pd.read_csv(case/'test.csv'); b = pd.read_csv(ref/'test.csv')
    joined = a.merge(b, on=['method', 'group', 'trial'], suffixes=('_actual', '_reference'))
    assert len(joined) == len(a) == len(b)
    np.testing.assert_allclose(joined.mse_actual, joined.mse_reference, rtol=0, atol=REFIT_TRIAL_MSE_ATOL)
    means = joined.groupby('method')[['mse_actual', 'mse_reference']].mean()
    np.testing.assert_allclose(means.mse_actual, means.mse_reference, rtol=0, atol=REFIT_MEAN_MSE_ATOL)
    # Negative-transfer decisions must agree exactly, despite tolerated roundoff.
    source_a = a.groupby(['method', 'group']).mse.mean().unstack('method')
    source_b = b.groupby(['method', 'group']).mse.mean().unstack('method')
    nt_a = source_a.gt(1.01*source_a.independent, axis=0)
    nt_b = source_b.gt(1.01*source_b.independent, axis=0)
    np.testing.assert_array_equal(nt_a.to_numpy(), nt_b.to_numpy())
    assert read(Path(args.run)/'all_models_locked.json')['test_created'] is False
    assert read(Path(args.run)/'completed.json')['failed_predictions'] == 0
    max_score_error = float(np.max(abs(joined.mse_actual-joined.mse_reference)))
    print('Verified %d arrays and %d trial scores; maximum absolute prediction difference %.3g; maximum trial-MSE difference %.3g; source selection and negative-transfer labels unchanged' % (count, len(joined), max_error, max_score_error))


if __name__ == '__main__': main()
