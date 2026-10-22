#!/usr/bin/env python3
"""Build the offline interactive figure from the committed saved-data bundle.

Only the standard library is required. No estimator is fitted, statistical
decision is recomputed, or scientific parameter is exposed as a UI control.
"""
import hashlib
import json
import bisect
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    source = ROOT / 'assets/figure_data.json'
    data = json.loads(source.read_text())
    # The saved figure bundle samples the quintic smoother at 301 times. The
    # selector actually matches piecewise-linear states on the 61-node grid.
    # Plot those same knots so each saved equal-state endpoint lies on its curve.
    training = []
    for row in data['training']:
        if len(row['smooth_time']) != 301:
            raise ValueError('The frozen illustration must have 301 saved curve samples')
        training.append({
            'source': row['source'], 'time': row['time'],
            'observed_position': row['observed_position'],
            'smooth_time': row['smooth_time'][::5],
            'selector_position': row['selector_position'][::5],
        })
    residuals = []
    for row in data['scalar_rows']:
        for source_key, time_key in [('g', 'time_p'), ('h', 'time_q')]:
            curve = training[row[source_key]]
            times, positions, time = curve['smooth_time'], curve['selector_position'], row[time_key]
            i = max(0, min(len(times)-2, bisect.bisect_right(times, time)-1))
            position = positions[i]+(positions[i+1]-positions[i])*(time-times[i])/(times[i+1]-times[i])
            residuals.append(abs(position-row['state']))
    if max(residuals) > 1e-12:
        raise ValueError('Saved match endpoints do not lie on the selector grid curves')
    bundle = {
        'seed': data['provenance']['example_seed'],
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'training': training,
        'selector_grid_size': 61,
        'matched_endpoint_maximum_interpolation_residual': max(residuals),
        'rows': data['scalar_rows'], 'peak': data['peak'],
        'center': data['center_interval'], 'evidence': data['evidence'],
        'core': data['selected_core'],
    }
    template = (ROOT / 'scripts/figures/interactive_template.html').read_text()
    payload = json.dumps(bundle, separators=(',', ':'), ensure_ascii=True).replace('<', '\\u003c')
    output = ROOT / 'docs/interactive/index.html'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(template.replace('/*__DSIDE_SAVED_DATA__*/', payload))
    print('Built docs/interactive/index.html from %d saved statistics (%d bytes).' %
          (len(bundle['rows']), output.stat().st_size))


if __name__ == '__main__':
    main()
