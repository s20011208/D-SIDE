"""CPU reproduction commands for the D-SIDE research release."""
import argparse
from pathlib import Path
from .io import read


def main():
    parser = argparse.ArgumentParser(prog='dside', description='Distributional Selective Integration for Dynamics Estimation')
    sub = parser.add_subparsers(dest='command', required=True)
    demo = sub.add_parser('demo', help='Replay the fixed seed-2026103000 illustration and plot all stages')
    demo.add_argument('--output', required=True)
    reproduce = sub.add_parser('reproduce', help='Replay an existing fixed simulation protocol')
    reproduce.add_argument('--config', required=True); reproduce.add_argument('--output', required=True)
    audit = sub.add_parser('audit-results', help='Recompute published summary metrics from source-level rows')
    audit.add_argument('--root', default=str(Path(__file__).resolve().parents[2]/'results/published'))
    args = parser.parse_args()
    if args.command == 'audit-results':
        from .validation import audit_results
        try:
            audit_results(Path(args.root))
        except FileNotFoundError as error:
            parser.error(str(error))
        return
    from .experiment import run
    from .plotting import plot_run
    config = read(Path(__file__).parent/'configs/demo.json') if args.command == 'demo' else read(args.config)
    out = run(config, args.output)
    image = plot_run(out)
    print('Completed. Summary: %s; figure: %s' % (out/'summary.csv', image))
