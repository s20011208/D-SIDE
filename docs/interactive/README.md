# Interactive algorithm figure

Open [the live panel](https://s20011208.github.io/D-SIDE/interactive/) to explore the saved illustration, seed **2026103000**. The [repository README](../../README.md#inside-the-algorithm) contains a linked static preview because GitHub Markdown does not run embedded JavaScript.

## Controls

- Choose one of the 28 source pairs, or click its cell in the source-evidence matrix.
- Move the retained-statistic slider, use its arrow keys, or select **Previous** / **Next** to inspect an equal-state match. The matched points and the corresponding statistic update together.
- Click the distribution plot to select the nearest retained statistic from the current source pair.

The histogram highlights the selected pair's actual weighted contribution to the **fixed global distribution**. Its total mass can differ across source pairs. The fitted mixture, evidence matrix and selected group are saved outputs; these interactions do not re-estimate the model, change a statistical threshold, or alter source selection.

All 793 retained statistics and all 28 source pairs come from [the committed plotting bundle](../../assets/figure_data.json). Reliability weighting is applied to the global distribution only. The shaded central range is descriptive, not a confidence interval.

## Run offline or rebuild

The committed `index.html` contains its plotting data, CSS and JavaScript. Download that file and open it in a browser, or open it from a local repository checkout. It requires no server, external JavaScript libraries, estimator fitting or network connection.

To regenerate it from the plotting bundle and template, run from the repository root:

```bash
python scripts/figures/build_interactive.py
```

The generator requires only Python's standard library. The generated page records the plotting bundle's SHA-256; `python scripts/check_release.py` checks that its statistics, peak, evidence and selected group still match the bundle.

## GitHub Pages setup

This repository's Pages files are in `docs/`. For its first publication, the repository owner must open **Settings → Pages**, select **Deploy from a branch**, choose **main** and **/docs**, and save. GitHub then serves the panel at the live URL above and updates it after later pushes. The site root redirects to the interactive panel.

The repository's SSH deploy key authorizes code pushes; it does not authorize changing GitHub repository settings. Enabling Pages does not require granting access to an organization or sharing an access token.
