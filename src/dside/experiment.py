"""Versioned simulation replay; fits are locked before any test creation."""
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
import platform
import sys
import numpy as np
import pandas as pd
import scipy
from .estimator import DSIDE, Trajectory
from ._backend.consensus_separation import generate
from ._backend.consensus_reliability import select_reliable_mixture
from ._backend.global_consensus import select_global_mixture, ClusterGreen, rhs_truth, damping_truth
from ._backend.serialization import save_model, load_model
from .io import dump, read, sha


def now(): return datetime.now(timezone.utc).isoformat()


def snapshot(model, path):
    path.mkdir()
    mapping = {}
    for g in range(8):
        save_model(model.models[g], path/("source_%d" % g))
        mapping[g] = model.mapping[g] if hasattr(model, 'mapping') else 0
    dump(path/'mapping.json', mapping)


def fit_case(seed, cohort, config, output):
    tic = perf_counter(); c = config['simulation']
    out = Path(output)/cohort/("seed_%d" % seed); out.mkdir(parents=True)
    generated = generate(c, seed, 'train')
    observations = [Trajectory(t.group, t.t, t.y, t.name) for t in generated]
    options = dict(c['fit']); options['operator_bounds'] = options['operator_bounds'][0]
    estimator = DSIDE(noise_sd=c['noise_sd'], grid=c['grid'], **options).fit(observations)
    dump(out/'training.json', [dict(source=t.group, name=t.name, time=t.t, position=t.y) for t in generated])
    # Initial states/truth are for post-hoc plots only and are never estimator inputs.
    dump(out/'generation.json', [dict(source=t.group, initial=t.initial, time=t.truth_t, position=t.truth_x) for t in generated])
    estimator.save(out/'dside_checkpoint')
    rows = estimator.selection_['rows']
    original_rows = [{k: v for k, v in r.items() if k not in ['global_weight', 'pair_weight', 'reliability', 'responsibility', 'enrichment', 'inside', 'z']} for r in rows]
    selections = {'density_only': estimator.selection_}
    if 'old' in config['methods']:
        selections['old'] = select_global_mixture(original_rows, 8)
    for name, wd, we in [('both', True, True), ('edges_only', False, True)]:
        if name in config['methods']:
            selections[name] = select_reliable_mixture(original_rows, 8, 21., wd, we)
    for name, value in selections.items(): dump(out/(name+'_selection.json'), value)
    models = {'independent': estimator.independent_, 'density_only': estimator.model_}
    cache = {tuple(tuple(g) for g in estimator.selection_['clusters']): estimator.model_}
    plans = {name: sel['clusters'] for name, sel in selections.items() if name != 'density_only'}
    plans.update(all_pool=[list(range(8))], oracle=[list(range(6)), [6], [7]])
    diagnostics = {}
    for name, plan in plans.items():
        if name not in config['methods']: continue
        key = tuple(tuple(g) for g in plan); reused = key in cache
        if not reused:
            cache[key] = ClusterGreen(plan, independent=estimator.independent_, order=2, **c['fit']).fit(estimator.smooth_)
        models[name] = cache[key]
        diagnostics[name] = dict(reused=reused, outer=models[name].outer_log, inner=models[name].inner_diagnostics)
    for name in ['independent', 'density_only']:
        m = models[name]; diagnostics[name] = dict(reused=False, outer=m.outer_log, inner=m.inner_diagnostics)
    for name, model in models.items(): snapshot(model, out/name)
    dump(out/'fit_diagnostics.json', diagnostics)
    dump(out/'fit_completed.json', dict(seed=seed, cohort=cohort, seconds=perf_counter()-tic,
        selected_sources=estimator.selected_sources_, selection_seconds=estimator.selection_seconds_))
    dump(out/'model_lock.json', dict(utc=now(), test_created=False,
        hashes={str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()}))


def evaluate_case(seed, cohort, config, output):
    out = Path(output)/cohort/("seed_%d" % seed); c = config['simulation']
    for name, digest in read(out/'model_lock.json')['hashes'].items():
        if sha(out/name) != digest: raise RuntimeError('Locked fit changed: '+name)
    test = generate(c, seed, 'test')
    dump(out/'test_design.json', [dict(name=t.name, source=t.group, time=t.truth_t, initial=t.initial) for t in test])
    scores, mechanisms, source_selection, predictions = [], [], [], {}
    for name in config['methods']:
        children = [load_model(out/name/("source_%d" % g)) for g in range(8)]
        mapping = {int(k): v for k, v in read(out/name/'mapping.json').items()}
        selection_path = out/(name+'_selection.json')
        if selection_path.exists():
            core = read(selection_path)['core']
            source_selection.append(dict(cohort=cohort, seed=seed, method=name, core=core,
                exact=core == list(range(6)), good_selected=len(set(core)&set(range(6))), bad_selected=len(set(core)&{6, 7})))
        for t in test:
            failure = None
            try:
                prediction = children[t.group].predict(mapping[t.group], t.truth_t, t.initial)
                mse = float(np.mean((prediction-t.truth_x)**2))
            except Exception as error:
                prediction = np.full_like(t.truth_x, np.nan); mse = float('inf'); failure = str(error)
            predictions[name+'__'+t.name] = prediction
            scores.append(dict(cohort=cohort, seed=seed, group=t.group, trial=t.name, method=name, mse=mse, failure=failure))
        x = np.linspace(-1, 1, 201); damping = damping_truth(c)
        for g in range(8):
            a = float(children[g].omega[mapping[g], 0])
            mechanisms.append(dict(cohort=cohort, seed=seed, group=g, method=name, estimated_damping=a,
                true_damping=float(damping[g]), damping_sq_error=float((a-damping[g])**2),
                rhs_mse=float(np.mean((children[g].rhs(mapping[g], x)-rhs_truth(g, x))**2))))
    predictions.update({'truth__'+t.name: t.truth_x for t in test})
    np.savez_compressed(out/'predictions.npz', **predictions)
    pd.DataFrame(scores).to_csv(out/'test.csv', index=False)
    pd.DataFrame(mechanisms).to_csv(out/'mechanism.csv', index=False)
    dump(out/'source_selection.json', source_selection)
    dump(out/'evaluation_audit.json', dict(utc=now(), model_lock_sha=sha(out/'model_lock.json'),
        failed_predictions=sum(bool(r['failure']) for r in scores)))


def run(config, output):
    algorithm = config['algorithm']
    if algorithm['variant'] != 'density_only' or algorithm['weight_density'] is not True or algorithm['weight_edges'] is not False:
        raise ValueError('The default D-SIDE release is the density-only variant')
    if algorithm.get('saturation_count') != 21. or algorithm.get('minimum_core_size') != 3:
        raise ValueError('Recorded protocols require saturation_count=21 and minimum_core_size=3')
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    backend = Path(__file__).parent/'_backend'
    dump(output/'protocol.json', dict(utc=now(), config=config, python=sys.version,
        numpy=np.__version__, scipy=scipy.__version__, pandas=pd.__version__, platform=platform.platform(),
        source_hashes={p.name: sha(p) for p in backend.glob('*.py')}))
    cases = [(seed, c['name']) for c in config['cohorts'] for seed in c['seeds']]
    tic = perf_counter()
    for seed, cohort in cases:
        fit_case(seed, cohort, config, output)
        print('Fitted %s / %s' % (cohort, seed), flush=True)
    dump(output/'all_models_locked.json', dict(utc=now(), test_created=False,
        hashes={cohort+'/'+str(seed): sha(output/cohort/("seed_%d" % seed)/'model_lock.json') for seed, cohort in cases}))
    for seed, cohort in cases:
        evaluate_case(seed, cohort, config, output)
        print('Evaluated %s / %s' % (cohort, seed), flush=True)
    tables = [pd.read_csv(output/cohort/("seed_%d" % seed)/'test.csv') for seed, cohort in cases]
    scores = pd.concat(tables, ignore_index=True)
    group = scores.groupby(['cohort', 'seed', 'method', 'group'], as_index=False).mse.mean()
    baseline = group[group.method == 'independent'].drop(columns='method').rename(columns={'mse': 'independent_mse'})
    group = group.merge(baseline, on=['cohort', 'seed', 'group'])
    group['negative_transfer'] = group.mse > 1.01*group.independent_mse
    group.to_csv(output/'group_metrics.csv', index=False)
    selection = [row for seed, cohort in cases for row in read(output/cohort/("seed_%d" % seed)/'source_selection.json')]
    pd.DataFrame(selection).to_csv(output/'source_selection.csv', index=False)
    summary = []
    for (cohort, method), g in group.groupby(['cohort', 'method']):
        s = [r for r in selection if r['cohort'] == cohort and r['method'] == method]
        summary.append(dict(cohort=cohort, method=method, seeds=int(g.seed.nunique()), mse=float(g.mse.mean()),
            related_mse=float(g[g.group < 6].mse.mean()), negative_transfer=int(g.negative_transfer.sum()),
            group_runs=len(g), exact_grouping=sum(r['exact'] for r in s) if s else None))
    pd.DataFrame(summary).to_csv(output/'summary.csv', index=False)
    dump(output/'completed.json', dict(utc=now(), cases=len(cases), seconds=perf_counter()-tic,
        failed_predictions=sum(bool(r) for r in scores.failure.fillna(False))))
    return output
