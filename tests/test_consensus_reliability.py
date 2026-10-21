import numpy as np
import pytest
from dside._backend.consensus_reliability import pair_mass, select_reliable_mixture
from dside._backend.global_consensus import select_global_mixture


def example_rows():
    rng = np.random.default_rng(915)
    rows = []
    for g in range(5):
        for h in range(g+1, 5):
            z = rng.normal(.3, .08, 25) if h < 4 else rng.uniform(-1.3, 1.3, 9)
            rows += [dict(g=g, h=h, state=float(i), statistic=float(np.tan(t)),
                          weight=float(1+.2*i)) for i, t in enumerate(z)]
    return rows


def test_few_rows_and_concentrated_weights_have_less_influence():
    def info(w): return pair_mass([dict(weight=x) for x in w])[1]
    small, large = info([1.]), info([1.]*21)
    assert small['mass'] == pytest.approx(1/22)
    assert large['mass'] == pytest.approx(.5)
    assert info([10000.]+[1.]*20)['mass'] < .05
    assert info([1.]*10000)['mass'] < 1.
    assert pair_mass([])[1]['mass'] == 0.


def test_weight_scale_does_not_change_pair_mass():
    rows = [dict(weight=v) for v in [1., 3., 7.]]
    a, da = pair_mass(rows)
    b, db = pair_mass([dict(weight=r['weight']*1e150) for r in rows])
    assert np.allclose(a, b)
    assert da['mass'] == pytest.approx(db['mass'])
    with pytest.raises(ValueError): pair_mass([dict(weight=0.)])


def test_disabled_weighting_reproduces_frozen_algorithm():
    rows = example_rows()
    a = select_global_mixture(rows, 5)
    b = select_reliable_mixture(rows, 5, weight_density=False, weight_edges=False)
    assert a['partition'] == b['partition']
    assert a['core'] == b['core']
    assert np.allclose(a['support'], b['support'], atol=1e-7)


def test_weak_pair_affects_both_distribution_and_graph_less():
    rows = [r for r in example_rows() if (r['g'], r['h']) != (0, 4)]
    rows.append(dict(g=0, h=4, statistic=-8., weight=1.))
    result = select_reliable_mixture(rows, 5)
    pairs = {(r['g'], r['h']): r for r in result['pair_reliability']}
    assert pairs[0, 4]['global_mass'] < pairs[0, 1]['global_mass']/5
    assert pairs[0, 4]['graph_mass'] == pytest.approx(1/22)
    r = pairs[0, 4]
    assert abs(r['edge_evidence']) <= 1/22
    assert r['edge_evidence'] == pytest.approx(r['mass']*r['unweighted_enrichment'])
    assert sum(r['global_weight'] for r in result['rows']) == pytest.approx(1.)


def test_duplicate_replay_and_within_source_rows_add_no_evidence():
    rows = example_rows()
    a = select_reliable_mixture(rows, 5)
    b = select_reliable_mixture(rows+rows+[dict(g=0, h=0, statistic=100, weight=1e6)], 5)
    assert a['core'] == b['core']
    assert a['pair_reliability'] == b['pair_reliability']
    assert len(a['rows']) == len(b['rows'])


def test_no_evidence_falls_back_to_independent():
    result = select_reliable_mixture([], 8)
    assert result['clusters'] == [[g] for g in range(8)]
    assert result['core'] == []
