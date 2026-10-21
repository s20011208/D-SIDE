# Published simulation records

These compact tables are exports from completed research experiments, not new experiments run during repository packaging. Predictions and numerical identities are independently checked using the included fixed example.

| Directory | Existing protocol | Cohort | D-SIDE MSE | Exact group | Negative transfer |
|---|---|---|---:|---:|---:|
| [`original/`](original/) | Updated damping, original initial conditions | 10 historical seeds | 0.0078693186443 | 7/10 | 4/80 |
| [`adapted_retrospective/`](adapted_retrospective/) | Calibrated coverage | 10 historical seeds | 0.0079410476040 | 4/10 | 33/80 |
| [`adapted_fresh/`](adapted_fresh/) | Calibrated coverage | 20 fresh seeds | 0.0069352917766 | 6/20 | 59/160 |
| [`historical_weighting/`](historical_weighting/) | Original damping, reliability-weighting ablation | 10 historical seeds | 0.0072980815549 | 6/10 | 3/80 |

Each final-setting directory includes source-level `group_comparison.csv`, exact selections, mechanism scores, pair evidence, and `summary.json`. Method key `density_only` is the released D-SIDE variant. `old`, `all_pool`, and `oracle` are comparators; oracle uses true simulation labels only as a reference.

Negative transfer is source MSE > 1.01 × independent MSE. MSE is averaged over 10 held-out trajectories/source, 101 time points/trajectory, and then equally over sources and seeds. RHS scores are mean squared errors on 201 equally spaced positions in `[-1,1]`, not square-root L2 norms. No interval coverage is claimed.

The calibrated-coverage failures are intentionally retained. More matching rows are dependent evidence, not more independently observed trajectories.

Run `dside audit-results` to recompute the final-setting summary metrics from the source-level tables. See [`docs/reproduce.md`](../../docs/reproduce.md) for complete benchmark replay and [`result_provenance.json`](result_provenance.json) for the source experiment identity.
