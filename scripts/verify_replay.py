"""Compare a complete CLI example to the historical prediction fixture."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from dside.io import read


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
            np.testing.assert_allclose(actual[name], expected[name], atol=2e-8, rtol=2e-8); count += 1
    assert read(case/'density_only_selection.json')['core'] == read(ref/'selection_reference.json')['core']
    a = pd.read_csv(case/'test.csv'); b = pd.read_csv(ref/'test.csv')
    joined = a.merge(b, on=['method', 'group', 'trial'], suffixes=('_actual', '_reference'))
    assert len(joined) == len(a) == len(b)
    np.testing.assert_allclose(joined.mse_actual, joined.mse_reference, rtol=1e-6, atol=2e-9)
    assert read(Path(args.run)/'all_models_locked.json')['test_created'] is False
    assert read(Path(args.run)/'completed.json')['failed_predictions'] == 0
    print('Verified %d arrays and %d trial scores; maximum absolute prediction difference %.3g' % (count, len(joined), max_error))


if __name__ == '__main__': main()
