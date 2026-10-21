# Experiments and recorded evidence

This release presents the scalar-distribution source-selection method as **D-SIDE**, with reliability weighting applied to the global density only. The experiments below were actually run before repository packaging. They are simulation studies, not applications to clinical, motion-capture, or air-pollution data. Negative outcomes are retained.

## Model, observations, and common settings

Each source follows $X_g''+a_gX_g'=f_g(X_g)$. Sources 0–5 have the same nonlinear function

$$
f(x)=-x+0.45\sin(1.7x)-0.1x^3/(1+x^2).
$$

Sources 6 and 7 instead use

$$
f_6(x)=-1.9x-0.9\sin(1.7x)-0.1x^3/(1+x^2),
$$

$$
f_7(x)=-0.65x-0.65\sin(1.7x)-0.1x^3/(1+x^2).
$$

The estimator is not given these formulas or class labels. It fits a nonparametric kernel function, not a regression in this generating-function dictionary.

| Quantity | Setting |
|---|---|
| Sources | 8; six related and two heterogeneous |
| Training trajectories | Exactly one per source; no designated sparse target |
| Position observations per trajectory | 45, including both endpoints and 43 independently uniform interior times |
| Training interval | $[0,3]$ |
| Observation noise | Independent Gaussian, standard deviation 0.035 |
| Direct velocity/acceleration observations | None |
| Quadrature grid | 61 uniform nodes |
| Additional validation trajectories | None |
| Held-out trajectories | 10 per source, 80 per seed |
| Test initial position and velocity | Independent uniform draws from $[-1.25,1.25]$ |
| Prediction interval | $[0,0.8]$ from each new test initial state |
| Scoring grid | 101 equally spaced times per trajectory |
| Test initial states | Exact position and velocity supplied equally to every method |
| Prediction inputs | Initial state only; no future truth or observations |

Test trajectories are new trajectories within the same sources, not continuations of the training trajectory and not unseen-source generalization. The generating ODEs are deterministic, autonomous, and correctly specified by the estimator's structural form. There are no unobserved inputs, process noise, or within-source mechanism changes in this benchmark.

## Revised damping benchmark

The latest simulation intervention sets

$$
(a_0,\ldots,a_7)=(0.3875,0.3925,0.3975,0.4025,0.4075,0.4050,1.2,1.75).
$$

All methods use operator bounds $[0.02,2.5]$: the previous upper bound 1.6 would exclude the new source-7 truth. All other fitting parameters remain fixed as described in [the method](method.md).

Two initial-condition designs are kept separate:

- **Original initial conditions.** Every training trajectory starts at position $-1.15$. The eight positive velocities $(0.35,0.55,0.75,0.95,1.15,1.35,1.55,1.75)$ are randomly permuted among sources for each seed.
- **Adapted initial conditions.** All training positions start at $-1.5$. Eight target maximum positions, equally spaced from 1.0 to 2.2, are randomly permuted among sources. A root solve using each source's **known generating ODE** chooses a positive launch velocity so its maximum over a 601-point training grid matches the assigned target. This is a controlled simulation design, not an estimator available from real observations.

The adapted design was chosen from seven declared candidates using five development seeds, **2026105200–2026105204**. The criterion considered retained pair counts across all 28 source pairs, not correct source labels, clustering outcomes, or test errors. No replacement design was chosen after the unfavorable evaluation.

The ten historical seeds, **2026103000–2026103009**, had already been examined during method development and are labeled **retrospective**. The twenty seeds **2026105300–2026105319** were evaluated after freezing this intervention and are labeled **fresh**. The original-initial-condition control was run on the historical ten only; there is no unreported fresh-20 result for that control.

## Main prediction and selection outcomes

Position MSE is averaged over each source's 10 test trajectories and 101 scoring times, then averaged equally across eight sources. “Exact selection” means selecting precisely sources 0–5. Negative transfer is counted per seed–source pair when

$$
\operatorname{MSE}_{g}^{\mathrm{method}}>1.01\operatorname{MSE}_{g}^{\mathrm{independent}}.
$$

| Initial conditions and evaluation | Independent MSE | D-SIDE MSE | Change versus independent | Exact selection | Negative transfer |
|---|---:|---:|---:|---:|---:|
| Original, retrospective 10 seeds | 0.00901594 | 0.00786932 | −12.7% | 7/10 | 4/80 |
| Adapted, retrospective 10 seeds | 0.00749845 | 0.00794105 | +5.9% | 4/10 | 33/80 |
| Adapted, fresh 20 seeds | 0.00658151 | 0.00693529 | +5.4% | 6/20 | 59/160 |

The favorable original-initial-condition result supports a prediction gain in one controlled, retrospective setting. It does not establish uniform improvement or reliable prevention of negative transfer. All three rows must remain visible when interpreting the final intervention.

### Comparators in the same experiments

**Unweighted selector** uses the original equal-source-pair global distribution, with all other clustering and fitting steps unchanged. **Complete pooling** shares one nonlinear function across all eight sources while retaining separate, shrunk damping coefficients. **True-group reference** supplies the true collection 0–5 and leaves 6 and 7 independent. The latter is a diagnostic reference with unavailable true labels, not a deployable comparator or a guaranteed prediction lower bound.

| Design / evaluation | Method | All-source MSE | Related-source MSE | Negative transfer |
|---|---|---:|---:|---:|
| Original / retrospective 10 | Independent | 0.00901594 | 0.00352585 | — |
| Original / retrospective 10 | Unweighted selector | 0.00786932 | 0.00199702 | 4/80 |
| Original / retrospective 10 | **D-SIDE** | **0.00786932** | **0.00199702** | **4/80** |
| Original / retrospective 10 | Complete pooling | 0.00928114 | 0.00449140 | 46/80 |
| Original / retrospective 10 | True-group reference | 0.00781430 | 0.00192366 | 3/80 |
| Adapted / retrospective 10 | Independent | 0.00749845 | 0.00472664 | — |
| Adapted / retrospective 10 | Unweighted selector | 0.00686517 | 0.00365956 | 27/80 |
| Adapted / retrospective 10 | **D-SIDE** | **0.00794105** | **0.00434723** | **33/80** |
| Adapted / retrospective 10 | Complete pooling | 0.01042660 | 0.00658665 | 61/80 |
| Adapted / retrospective 10 | True-group reference | 0.00640171 | 0.00326432 | 26/80 |
| Adapted / fresh 20 | Independent | 0.00658151 | 0.00410522 | — |
| Adapted / fresh 20 | Unweighted selector | 0.00691629 | 0.00383264 | 62/160 |
| Adapted / fresh 20 | **D-SIDE** | **0.00693529** | **0.00376422** | **59/160** |
| Adapted / fresh 20 | Complete pooling | 0.00982073 | 0.00638817 | 129/160 |
| Adapted / fresh 20 | True-group reference | 0.00574785 | 0.00299368 | 45/160 |

In the original-initial-condition ten-seed experiment, the unweighted and density-weighted selectors make the same decisions and predictions. The gain over independent fitting therefore cannot be attributed specifically to reliability weighting. Even the true-group reference has individual prediction harms, so imperfect selection is not the only source of negative transfer.

The adapted fresh evaluation selected 105/120 possible related-source memberships and admitted 14/40 possible heterogeneous-source memberships. A missed related source stays independent; a selection error is therefore distinct from a negative-transfer event.

For adapted fresh seeds, the seed-paired D-SIDE-minus-independent mean MSE difference is $0.00035379$, with a descriptive 95% t interval $[-0.00077379,0.00148136]$. The corresponding difference from the unweighted selector is $0.00001900$, with interval $[-0.00065985,0.00069786]$. These descriptive intervals are not evidence of a significant improvement and are not corrected for selection across experiments.

## Why more pairs did not solve the problem

On the historical seeds, the median retained count per source pair increased from 21 under original initial conditions to 73 under adapted initial conditions. These are correlated evaluations of the same few curves, not new independent trajectories.

A post-hoc training-only diagnostic removed measurement noise and set the declared noise standard deviation to zero. Exact selection then became:

| Design / evaluation | Noisy observations | Noise-removed diagnostic |
|---|---:|---:|
| Original / retrospective 10 | 7/10 | 10/10 |
| Adapted / retrospective 10 | 4/10 | 10/10 |
| Adapted / fresh 20 | 6/20 | 19/20 |

Because setting the declared standard deviation to zero also changes precision filtering and weights, this is a pipeline ablation rather than a clean estimate of one causal noise effect.

For the related sources in historical seeds, the noise-removed coordinate discrepancy $b+aA-f(\widetilde X)$ had mean square 0.0027464 under original initial conditions and 0.0228540 under adapted initial conditions, an increase of about 8.3-fold. Larger motion ranges improved pairing availability while worsening the polynomial/inverse approximation. The initial-condition selection criterion did not control that bias.

## Noise standard deviation versus noise-to-signal ratio

The experiments fix the absolute observation-noise standard deviation, not a common noise-to-signal ratio. For a particular trajectory, define

$$
\rho_g=\frac{0.035}{\left[m^{-1}\sum_{j=1}^{m}\{X_g(t_{gj})-\overline X_g\}^2\right]^{1/2}},\qquad m=45.
$$

The denominator uses the **noise-free trajectory at its actual observation times**, with divisor $m$ rather than $m-1$. The table reports the arithmetic mean of individual ratios, not a ratio of pooled standard deviations.

| Design / evaluation | Mean $\rho_g$ | Individual range |
|---|---:|---:|
| Original / retrospective 10 | 0.0640 | 0.0419–0.0998 |
| Adapted / retrospective 10 | 0.0391 | 0.0300–0.0584 |
| Adapted / fresh 20 | 0.0396 | 0.0270–0.0658 |

The adapted design's lower relative position noise did not prevent worse selection. Position-level signal-to-noise ratio does not measure amplification in inverse coordinates or ratios.

## Earlier weighting ablation: separate setting

An earlier experiment used damping $(0.3875,0.3925,0.3975,0.4025,0.4075,0.4125,1.05,0.75)$, original initial conditions, and operator bounds $[0.02,1.6]$. All ten historical seeds were already seen. It compared where to apply the same pair reliability factor:

| Variant | All-source MSE | Exact selection | Negative transfer |
|---|---:|---:|---:|
| Equal pair mass | 0.00735439 | 4/10 | 3/80 |
| Reliability in density and graph | 0.00786345 | 4/10 | 9/80 |
| **Reliability in density only** | **0.00729808** | **6/10** | **3/80** |
| Reliability in graph only | 0.00731584 | 4/10 | 3/80 |

This ablation explains the released variant's identity. Its small density-only improvement was retrospective and did not establish independent confirmation. It must not be combined with the revised-damping results as if all rows came from one unchanged data-generating setting.

## Metrics, isolation, and numerical records

- Right-hand-side error is the mean squared function discrepancy on 201 equally spaced positions in $[-1,1]$. It can include states poorly covered in training; it is not a universal mechanism-recovery measure.
- Operator error is squared damping-coefficient error. Prediction improvements need not imply better operator recovery.
- All 40 experiment instances in the revised-damping intervention were fitted and locked before held-out scoring. Ground-truth training initial states were removed from estimator inputs. True grouping was used only by the explicitly named reference fit.
- Original and adapted historical comparisons use matched observation-time/noise streams and exactly the same 800 held-out true trajectories across ten seeds.
- The recorded run had zero failed mixture starts, zero failed actual outer starts, and zero failed predicted trajectories. Fit and evaluation wall times were approximately 40.7 and 17.3 seconds in the original two-worker environment; these are run receipts, not portable speed benchmarks.
- Main tables summarize already recorded results. Repository packaging replays are reproducibility checks, not additional independent scientific confirmation.

The public release retains compact result tables, protocols, and reproduction code rather than all redundant model checkpoints. See the [published result index](../results/published/README.md) for the shipped tables and the root [Quick Start](../README.md#quick-start) for the supported command-line entry point.
