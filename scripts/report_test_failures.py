"""Expose real JUnit failures in check annotations without requiring log access."""
import argparse
import os
from pathlib import Path
from xml.etree import ElementTree


def escape(value):
    return value.replace('%', '%25').replace('\r', '%0D').replace('\n', '%0A')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--report', required=True); args = ap.parse_args()
    path = Path(args.report)
    if not path.exists():
        print('No JUnit report was produced. Inspect the earlier installation or test step.')
        return
    root = ElementTree.parse(path).getroot(); failures = []
    for case in root.iter('testcase'):
        failure = case.find('failure')
        if failure is None: failure = case.find('error')
        if failure is None: continue
        name = case.get('classname', '')+'.'+case.get('name', '')
        message = (failure.text or failure.get('message', ''))[:14000]
        workspace = os.environ.get('GITHUB_WORKSPACE')
        if workspace: message = message.replace(workspace+'/', '')
        failures.append((name, message))
        print('::error title='+escape(name)+'::'+escape(message))
    summary = os.environ.get('GITHUB_STEP_SUMMARY')
    if summary:
        with open(summary, 'a') as out:
            out.write('## Numerical and release tests\n\n')
            out.write('JUnit failures: %d\n\n' % len(failures))
            for name, message in failures:
                out.write('### '+name+'\n\n```text\n'+message+'\n```\n\n')
    print('Reported %d actual test failures.' % len(failures))


if __name__ == '__main__': main()
