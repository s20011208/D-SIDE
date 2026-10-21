# Reproduction guide

The release separates an executable example, the recorded multi-seed benchmark, and a historical weighting ablation. Replaying a recorded seed checks reproducibility; it does not create a new independent confirmation result.

## Install in an independent environment

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade "pip==24.3.1" "setuptools==75.3.2" "wheel==0.45.1"
python -m pip install -r requirements.lock.txt
python -m pip install -e . --no-deps --no-build-isolation
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
```

The package metadata declares the supported Python versions. Use the supplied dependency lock for exact package-version matching. A lockfile and single-thread settings do not fix CPU-dispatched BLAS kernels, so they do not guarantee bitwise-identical refitting across machines.

The release separates exact provenance checks, numerical regression of a complete refit, and prediction from a fixed fitted checkpoint. Source and fixture hashes and the illustration's source-group labels are checked exactly; refitted floating-point predictions and scores use explicit tolerances. See [numerical portability](numerical_portability.md) for the controlled CPU-kernel diagnostic, the original CI failure, and the comparison thresholds.

The README uses an editable installation from a repository checkout. If installing the built wheel elsewhere, the demo configuration is included in the package, while benchmark tables remain in the checkout: use `dside audit-results --root /path/to/D-SIDE/results/published` to inspect them.

## Run the complete example

```bash
dside demo --output runs/demo
```

The example should generate one eight-source training dataset, construct labeled scalar statistics, select sources with the released density-only rule, fit independent and selected models, and score genuinely free test trajectories from supplied initial states. It writes the observed data, selection diagnostics, model-fit information, result tables, and figures to the requested output directory. Use a new directory when replaying so previous scientific records remain intact.

The fixed example seed is a recorded illustration, not a success selected after each rerun. Inspect all included benchmark seeds before drawing performance conclusions.

## Reproduce the recorded benchmark settings

```bash
dside reproduce --config configs/benchmark_original.json --output runs/original
dside reproduce --config configs/benchmark_adapted.json --output runs/adapted
```

The first configuration preserves the original training starts with revised damping. Its ten recorded seeds are retrospective. The second reproduces the fixed adapted-initial-condition design, retaining the distinction between the historical ten and fresh twenty seeds specified in its configuration.

The adapted design was selected before its evaluation from seven candidates using development seeds 2026105200–2026105204. Reproducing its fixed configuration does not reselect initial conditions, optimize a new setting, or search for a better result. The launch-speed calibration uses known generating equations solely to construct this controlled simulation.

For comparable CPU accounting and to avoid unnecessary nested linear-algebra threads, the original scientific runs used:

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
```

Thread count changes are not scientific parameter changes, but they can affect wall time and small floating-point details.

## Audit the published evidence

```bash
dside audit-results
python -m pytest -q
python scripts/verify_replay.py --run runs/demo
```

The result audit checks the compact recorded evidence supplied with the repository. It is distinct from rerunning the full simulation. `verify_replay.py` compares a newly fitted complete example with the archived reference under the full-refit numerical contract; the tests also exercise fixed-checkpoint prediction separately. A successful audit confirms numerical/table consistency under its implemented checks; it does not establish that the statistical procedure is unbiased, universally beneficial, or theoretically identified.

The recorded benchmark compares independent fitting, the unweighted selector, D-SIDE, complete pooling, and a true-group diagnostic reference. Keep all methods and every declared seed, including failed selections. Preserve the method's hard single-core rule, the density-only reliability switch, smoothing and feature settings, and test initial states.

## Check the scientific contract

A valid replay should satisfy all of the following:

1. Only the eight noisy training curves and declared noise standard deviation enter source selection.
2. True training initial states and true source labels do not enter the estimator. The true-group reference is separately and explicitly labeled.
3. Global density uses source-pair reliability mass; graph evidence does not receive that extra factor.
4. Functions are shared only inside the selected main block; excluded sources stay independent.
5. Test decisions are not used to select hyperparameters, pooling blocks, or initial-condition designs.
6. Prediction starts from independently supplied position and velocity, then uses only predicted states.
7. Per-source MSE is averaged over held-out trajectories before equal-source aggregation. Negative transfer uses the same 1% threshold.

The full mathematical contract is in [method.md](method.md). Exact generating equations, seeds, result tables, and the noise-to-signal definition are in [experiments.md](experiments.md).

## Expected evidence, not a promised universal outcome

For original starts and revised damping, the recorded ten-seed D-SIDE mean MSE is 0.0078693186443 versus 0.0090159413356 for independent fitting, with exact source recovery 7/10 and negative transfer 4/80. The unweighted selector gives identical results in that setting.

For adapted starts and fresh twenty seeds, D-SIDE mean MSE is 0.0069352917766 versus 0.0065815050556 independently, with exact recovery 6/20 and negative transfer 59/160. That unfavorable result is part of the expected record, not an error to repair by changing seeds or thresholds.

## Release validation scope

Packaging validation independently installed the locked environment, ran the complete fixed illustration, and compared every prediction with the archived reference. The first local replay had zero observed prediction difference in that environment; this is a historical local result, not a cross-platform guarantee. A subsequent controlled CPU-kernel replay quantified small refit differences while preserving the illustrated source group and negative-transfer labels; see [the portability record](numerical_portability.md).

The complete multi-seed benchmark commands are provided for replay; the full multi-seed study was not rerun as part of packaging. A replay of an existing seed is not a new confirmation experiment.
