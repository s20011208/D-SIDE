"""Public-release regression against an actual frozen, historical experiment."""
import json
from pathlib import Path
import numpy as np
import pytest
from dside import DSIDE, Trajectory
from dside.validation import audit_results, REFIT_PREDICTION_ATOL, REFIT_MEAN_MSE_ATOL
from dside._backend.serialization import load_model
from dside.io import read, sha

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT/'tests/fixtures/seed_2026103000'


@pytest.fixture(scope='module')
def fitted():
    records = read(FIXTURE/'training.json')
    class PositionOnly:
        def __init__(self, r):
            self.source = r['group']; self.time = r['t']; self.position = r['y']; self.name = r['name']
        @property
        def initial(self): raise AssertionError('Training truth must not be accessed')
        @property
        def truth_x(self): raise AssertionError('Training truth must not be accessed')
    return DSIDE().fit([PositionOnly(r) for r in records])


def test_backend_source_hashes():
    for row in read(ROOT/'source_provenance.json')['files']:
        assert sha(ROOT/row['release_path']) == row['sha256']


def test_default_is_density_only_and_matches_frozen_selection(fitted):
    ref = read(FIXTURE/'selection_reference.json')
    assert fitted.selection_['weighting']['density'] is True
    assert fitted.selection_['weighting']['edges'] is False
    assert fitted.selected_sources_ == ref['core']
    np.testing.assert_allclose(fitted.selection_['support'], ref['support'], atol=2e-7, rtol=0)
    for key in ['mu', 'sigma', 'proportion']:
        assert fitted.selection_['peak'][key] == pytest.approx(ref['peak'][key], abs=2e-7)


def test_refitted_predictions_against_archived_reference(fitted):
    tests = read(FIXTURE/'test_design.json')
    errors = []
    with np.load(FIXTURE/'predictions.npz', allow_pickle=False) as reference:
        for tr in tests:
            prediction = fitted.predict(tr['group'], tr['t'], tr['initial'])
            np.testing.assert_allclose(prediction, reference['density_only__'+tr['name']], atol=REFIT_PREDICTION_ATOL, rtol=0)
            errors.append(np.mean((prediction-reference['truth__'+tr['name']])**2))
    assert np.mean(errors) == pytest.approx(0.008423258902134238, abs=REFIT_MEAN_MSE_ATOL, rel=0)


def test_fixed_checkpoint_predictions_against_archived_reference():
    """Isolate free-simulation arithmetic from floating-point refit variation."""
    checkpoints = FIXTURE/'frozen_fit'
    manifest = read(checkpoints/'manifest.json')
    for relative_path, expected_hash in manifest['sha256'].items():
        assert sha(checkpoints/relative_path) == expected_hash
    models = {key: load_model(checkpoints/key) for key in ['shared', 'source_6', 'source_7']}
    with np.load(FIXTURE/'predictions.npz', allow_pickle=False) as reference:
        for tr in read(FIXTURE/'test_design.json'):
            group = tr['group']
            key, local = ('shared', group) if group < 6 else ('source_%d' % group, 0)
            prediction = models[key].predict(local, tr['t'], tr['initial'])
            np.testing.assert_allclose(prediction, reference['density_only__'+tr['name']], atol=2e-8, rtol=0)


def test_checkpoint_roundtrip_shared_and_independent_sources(fitted, tmp_path):
    checkpoint = tmp_path/'model'; fitted.save(checkpoint); restored = DSIDE.load(checkpoint)
    assert restored.selected_sources_ == fitted.selected_sources_
    tests = read(FIXTURE/'test_design.json')
    for source in [0, 5, 6, 7]:
        tr = next(t for t in tests if t['group'] == source)
        np.testing.assert_array_equal(restored.predict(source, tr['t'], tr['initial']), fitted.predict(source, tr['t'], tr['initial']))


def test_input_scope_and_invalid_prediction_states(fitted):
    t = np.linspace(0., 1., 8)
    with pytest.raises(ValueError, match='one training trajectory'):
        DSIDE().fit([Trajectory('same', t, t), Trajectory('same', t, t)])
    with pytest.raises(ValueError, match='nine sources'):
        DSIDE().fit([Trajectory(i, t, t) for i in range(10)])
    with pytest.raises(ValueError, match='strictly increasing'):
        DSIDE().fit([Trajectory(0, t[::-1], t)])
    with pytest.raises(ValueError, match='position and velocity'):
        fitted.predict(0, t, [0.])


def test_published_tables_and_packaged_demo():
    assert audit_results(ROOT/'results/published')
    assert read(ROOT/'configs/demo.json') == read(ROOT/'src/dside/configs/demo.json')
