# Numerical portability and regression checks

D-SIDE keeps the scientific algorithm, archived results, and numerical comparison policy separate. The frozen backend and reference fixtures remain unchanged. Full refitting is compared numerically; it is not required to reproduce every floating-point bit on a different CPU.

## Three distinct checks

| Check | What is held fixed | Comparison |
|---|---|---|
| Provenance and protocol | Committed backend and fixed-checkpoint files; declared source selection and transfer decisions | Exact file hashes for the backend/checkpoints and categorical equality for decisions |
| Full-refit regression | The recorded data/seed and algorithm; smoothing, selection, fitting, and simulation are rerun | Bounded absolute prediction and score differences |
| Fixed-checkpoint prediction | The fitted function representation and operator coefficients; no optimization is rerun | A separate, stricter prediction comparison |

The source selection and negative-transfer decisions for the fixed release illustration remain categorical checks. Their observed equality in this diagnostic is not a guarantee that arbitrary data near a decision boundary will be insensitive to floating-point changes.

## Why the initial CI comparison failed

The first regression test compared **freshly refitted** predictions with archived predictions using `atol=2e-8, rtol=2e-8`. The first GitHub Actions runs exposed differences larger than that limit. This test combined several numerical stages: trajectory smoothing, matrix decompositions, numerical optimization of operator coefficients, and ODE simulation. It was not a test of the ODE solver alone.

Pinned Python and package versions do not pin the machine's CPU instruction set or the BLAS implementation's runtime choice of numerical kernel. One thread also does not make those kernels identical. Small rounding changes can propagate through the fitting procedure, including its finite-difference operator optimization, while the source code and mathematical objective remain the same.

To investigate that failure, a controlled replay used the same Python 3.8 environment and changed the OpenBLAS dispatch from its Cooperlake kernel to its Haswell kernel. No model, seed, data-generating setting, estimator parameter, or archived reference was changed. The comparison below concerns the fixed historical illustration, seed **2026103000**, rather than a new scientific experiment.

## Controlled measurements

The complete replay compared 480 arrays: 400 predicted trajectories from five methods, plus 80 generated truth arrays. It also compared all 400 trial MSEs. The methods were independent fitting, the original unweighted selector, D-SIDE (`density_only`), complete pooling, and the true-group reference.

| Quantity | Observed difference |
|---|---:|
| Maximum absolute prediction difference across the complete replay | $2.75024383045\times10^{-7}$ |
| Maximum absolute difference among 400 trial MSEs | $2.89385984897\times10^{-8}$ |
| D-SIDE mean-MSE difference, new minus archived | $-3.65524445654\times10^{-9}$ |
| Selected D-SIDE sources | Exactly $[0,1,2,3,4,5]$ in both runs |
| Changed negative-transfer decisions | 0 |

D-SIDE's mean MSE changed from approximately 0.0084232589021 to 0.0084232552469 in this one-seed replay. These figures refer to the illustration, not the ten-seed mean reported in the experiment tables.

An earlier, separate diagnostic held the fitted checkpoint fixed and reevaluated all 80 D-SIDE test trajectories through the public API. Its maximum absolute prediction difference was approximately $5.0148\times10^{-12}$. That much smaller discrepancy distinguishes fixed-model prediction from the complete refit; it does not establish a universal error bound for either procedure.

The initial local packaging receipt, which recorded zero prediction difference, is preserved as a historical environment-specific result. It is not replaced by a claim that every machine must reproduce that zero. The compact release-validation record is stored in [release_validation.json](release_validation.json).

## Released numerical contract

The full-refit comparisons use **absolute tolerances**, with relative tolerance set to zero. These values apply to the fixed illustration in its dimensionless simulation units:

| Compared output | Absolute tolerance |
|---|---:|
| Each refitted trajectory prediction | $10^{-6}$ |
| Each trial MSE in the complete replay | $10^{-7}$ |
| Each method’s mean MSE of the fixed illustration | $10^{-8}$ |
| Prediction from the unchanged frozen checkpoint | $2\times10^{-8}$ |

These thresholds distinguish a numerically equivalent replay of this fixed benchmark from an unintended implementation change. They are release regression thresholds, not statistical confidence limits or guarantees over all hardware, trajectories, or model fits. Rescaling the units or evaluating another dataset requires reassessing the numerical comparison. The absolute prediction tolerance is small relative to the benchmark's position-noise standard deviation of 0.035, while remaining larger than the measured CPU-kernel refit discrepancy.

The diagnosis led to a change in the **refit regression comparison policy** and separate fixed-checkpoint tests. It did not change the algorithm, optimizer objective, smoothing rules, source-clustering criterion, saved reference predictions, benchmark scores, or random seeds. Predictions and scores are neither rounded nor clamped to force agreement.

## Reading future validation results

Use [the reproduction guide](reproduce.md) to run the example and its comparisons. Keep raw deviations when a check fails; do not alter scientific inputs or silently replace the fixture to obtain a pass. A passing numerical regression certifies the implemented release checks for that environment. It does not establish new statistical performance, guarantee source recovery, or remove the limitations documented in [limitations.md](limitations.md).
