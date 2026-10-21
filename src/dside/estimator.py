"""Small public API around the frozen density-only research implementation."""
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
import numpy as np
from ._backend.core import Trial, smooth_trial
from ._backend.baselines import IndependentModel
from ._backend.global_consensus import ClusterGreen
from ._backend.consensus_reliability import select_reliable_mixture
from ._backend.scalar_sources import selector_smoothing
from ._backend.scalar_precision import precision_pairs
from ._backend.serialization import save_model, load_model
from .io import dump, read


@dataclass
class Trajectory:
    """One position-only training trajectory; source is a string or integer ID."""
    source: object
    time: object
    position: object
    name: str = ""


class _PredictionEnsemble:
    def predict(self, group, times, initial):
        return self.models[group].predict(self.mapping[group], times, initial)

    def rhs(self, group, position):
        return self.models[group].rhs(self.mapping[group], position)


class DSIDE:
    """Scalar K=2, one training curve per source, at most nine sources.

    Density reliability weighting is always enabled and graph reliability
    weighting is always disabled. No truth labels, training initial states,
    validation observations or future states are accepted by this API.
    """
    def __init__(self, noise_sd=0.035, grid=61, rank=16, lambda0=0.0002,
                 omega_shrinkage=0.0005, starts=2, maxiter=100,
                 operator_bounds=(0.02, 2.5), saturation_count=21.):
        if not np.isfinite(noise_sd) or noise_sd < 0:
            raise ValueError("noise_sd must be finite and nonnegative")
        if grid < 7:
            raise ValueError("grid must have at least seven points")
        self.noise_sd, self.grid = float(noise_sd), int(grid)
        self.saturation_count = float(saturation_count)
        self.fit_options = dict(order=2, rank=rank, lambda0=lambda0,
            omega_shrinkage=omega_shrinkage, starts=starts, maxiter=maxiter,
            operator_bounds=[list(operator_bounds)])

    def fit(self, trajectories):
        trajectories = list(trajectories)
        if not 1 <= len(trajectories) <= 9:
            raise ValueError("This release supports one to nine sources")
        self.sources_ = [t.source for t in trajectories]
        if len(set(self.sources_)) != len(self.sources_):
            raise ValueError("This release requires exactly one training trajectory per source")
        if not all(isinstance(s, (str, int)) for s in self.sources_):
            raise ValueError("source IDs must be strings or integers")
        self.source_index_ = {source: i for i, source in enumerate(self.sources_)}
        raw = []
        for i, tr in enumerate(trajectories):
            time = np.asarray(tr.time, float); position = np.asarray(tr.position, float)
            if time.ndim != 1 or position.shape != time.shape or len(time) < 6:
                raise ValueError("Each trajectory needs at least six paired time/position values")
            if not np.isfinite(time).all() or not np.isfinite(position).all() or np.any(np.diff(time) <= 0):
                raise ValueError("Use finite observations with strictly increasing times")
            raw.append(Trial(i, tr.name or "source_%d" % i, time.copy(), position.copy(), np.full(2, np.nan)))
        if len(set(t.name for t in raw)) != len(raw):
            raise ValueError("Trajectory names must be unique")
        tic = perf_counter()
        smooth = [smooth_trial(t, self.grid, self.noise_sd) for t in raw]
        rows, diagnostics = precision_pairs(raw, selector_smoothing(raw, smooth, 'poly5'), self.noise_sd, trim=.02)
        self.selection_ = select_reliable_mixture(rows, len(raw), self.saturation_count,
                                                weight_density=True, weight_edges=False)
        self.selection_["precision_diagnostics"] = diagnostics
        self.selection_seconds_ = perf_counter()-tic
        self.selected_sources_ = [self.sources_[g] for g in self.selection_["core"]]
        self.independent_ = IndependentModel(**self.fit_options).fit(smooth)
        self.model_ = ClusterGreen(self.selection_["clusters"], independent=self.independent_,
                                  **self.fit_options).fit(smooth)
        self.raw_, self.smooth_ = raw, smooth
        self.elapsed_ = perf_counter()-tic
        self.estimated_damping_ = {s: float(self.model_.omega[g, 0]) for g, s in enumerate(self.sources_)}
        return self

    def predict(self, source, time, initial_state):
        """Free simulation from supplied position/velocity; no future inputs."""
        times = np.asarray(time, float)
        if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError("Prediction times must be finite and strictly increasing")
        initial = np.asarray(initial_state, float)
        if initial.shape != (2,) or not np.isfinite(initial).all():
            raise ValueError("Supply a finite initial position and velocity")
        return self.model_.predict(self.source_index_[source], times, initial)

    def rhs(self, source, position):
        return self.model_.rhs(self.source_index_[source], np.asarray(position, float))

    def save(self, directory):
        path = Path(directory); path.mkdir(parents=True, exist_ok=False)
        for g in range(len(self.sources_)):
            save_model(self.model_.models[g], path/("source_%d" % g))
        dump(path/'metadata.json', dict(sources=self.sources_, selected_sources=self.selected_sources_,
            mapping=self.model_.mapping, selection=self.selection_, noise_sd=self.noise_sd,
            grid=self.grid, fit_options=self.fit_options, saturation_count=self.saturation_count))

    @classmethod
    def load(cls, directory):
        path = Path(directory); meta = read(path/'metadata.json')
        options = dict(meta['fit_options']); options.pop('order'); options['operator_bounds'] = options['operator_bounds'][0]
        obj = cls(noise_sd=meta['noise_sd'], grid=meta['grid'], saturation_count=meta['saturation_count'], **options)
        obj.sources_ = meta['sources']; obj.source_index_ = {s: g for g, s in enumerate(obj.sources_)}
        obj.selected_sources_ = meta['selected_sources']; obj.selection_ = meta['selection']
        ensemble = _PredictionEnsemble()
        ensemble.models = {g: load_model(path/("source_%d" % g)) for g in range(len(obj.sources_))}
        ensemble.mapping = {int(g): local for g, local in meta['mapping'].items()}
        obj.model_ = ensemble
        obj.estimated_damping_ = {s: float(ensemble.models[g].omega[ensemble.mapping[g], 0]) for g, s in enumerate(obj.sources_)}
        return obj
