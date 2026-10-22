# Figure provenance

These figures use the saved **density-only reliability weighting** variant of D-SIDE. They are not an additional experiment or a new parameter search.

| Asset | Content | Evidence |
|---|---|---|
| `hero.svg` | D-SIDE identity and decorative trajectories | Explicitly schematic; no numerical claim |
| `workflow.svg` | Five stages, from limited noisy trajectories to selective dynamics estimation | Algorithm schematic; singleton fallback is shown |
| `source_selection.svg` | Retained equal-state pairs, weighted global distribution, signed source evidence, selected group | Actual saved run, seed **2026103000**, new damping and **original** initial conditions |
| `prediction.svg` | A fixed first held-out trajectory and per-source prediction errors | First source/first test trial from seed 2026103000; aggregate panel uses **all 10 retrospective seeds** 2026103000–2026103009 |
| `figure_data.json` | Self-contained numerical input and original source-file hashes | No checkpoints, external workspace, private absolute paths, or future inputs are needed |

SVG is the main README format; numerical plots also have PNG versions. Hero/workflow PNG previews are available when the native librsvg renderer is installed. The build script does not require it for the primary SVG figures.

## Rebuild

From the repository root, after installing NumPy and Matplotlib:

```bash
python scripts/figures/build_figures.py
```

The plotting script uses only NumPy, Matplotlib, and Python's standard library. It does not fit models or regenerate the simulation.

To rebuild only the workflow and source-clustering panels:

```bash
python scripts/figures/build_figures.py --figures workflow source_selection
```

## Interpretation

- The workflow uses the generic integer-order equation `D^K_(omega_g) X = f_g(X)`, with `K >= 2`. The saved numerical example is second order (`K = 2`). Its equal-state schematic markers are computed on the drawn cubic Bezier curves at the same vertical coordinate, rather than positioned by eye.
- The source-selection example was fixed as the first historical seed. The displayed source pair `(0, 5)` makes equal-state matching easy to see; the global histogram and evidence matrix use **every retained cross-source pair**.
- The displayed curves use the actual **61-node matching grid**, extracted as every fifth node of the stored 301-point polynomial evaluation. Matching markers therefore lie on the same linearly interpolated segments used to form the statistics. The original noisy-observation scatter is unchanged.
- Distribution colors use known simulation truth **only after selection**, for diagnostic display. The algorithm is not supplied with these labels or the number of related sources.
- Histogram heights use the saved global weights, divided by bin width. Each labeled pair contributes according to its saved reliability mass. The fitted line uses the stored truncated-Gaussian-plus-uniform parameters.
- The shaded distribution interval marks model enrichment. It is **not a confidence interval** and does not prove that a source is related.
- The selected block in the matrix is the algorithm's actual output, `[0, 1, 2, 3, 4, 5]`. Sources `6` and `7` retain independent fits in this run.
- The prediction illustration is the first source and first test trajectory, with identical initial position and velocity for all methods. It was not selected for improvement; D-SIDE has a larger error on this particular trajectory. The right panel reports mean MSE across all 10 new test trajectories per source and all 10 retrospective seeds.
- Aggregate improvement in this controlled setting does not imply that every source, seed, or trajectory benefits. The broader repository includes failure settings and negative-transfer counts.
