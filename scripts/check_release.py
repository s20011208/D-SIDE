"""Validate local links, SVGs, source hashes and public-release boundaries."""
import argparse
import json
import re
from pathlib import Path
from urllib.parse import unquote
from xml.etree import ElementTree
from dside.io import read, sha


def main():
    root = Path(__file__).resolve().parents[1]
    issues = []; links = 0
    for p in [root/'README.md', root/'NOTICE.md']+list((root/'docs').rglob('*.md'))+list((root/'docs').rglob('*.html'))+list((root/'assets').glob('*.md'))+list((root/'results/published').glob('*.md')):
        content = p.read_text()
        targets = re.findall(r'\]\(([^)]+)\)', content)+re.findall(r'(?:src|href)="([^"]+)"', content)
        for target in targets:
            if target.startswith(('https://', 'http://', '#', 'mailto:')): continue
            target = unquote(target.split('#')[0].split('?')[0])
            if not target: continue
            links += 1
            if not (p.parent/target).exists(): issues.append(str(p.relative_to(root))+': missing '+target)
        if '/home/' in content or 'file://' in content: issues.append(str(p.relative_to(root))+': local private path')
    for p in (root/'assets').glob('*.svg'): ElementTree.parse(p)
    for row in read(root/'source_provenance.json')['files']:
        if sha(root/row['release_path']) != row['sha256']: issues.append('Changed frozen backend: '+row['release_path'])
    fixture = root/'tests/fixtures/seed_2026103000'
    for p in list((root/'configs').glob('*.json'))+list(fixture.glob('*.json')):
        json.loads(p.read_text())
    if read(root/'configs/demo.json') != read(root/'src/dside/configs/demo.json'): issues.append('Packaged demo config differs')
    panel = (root/'docs/interactive/index.html').read_text()
    payload = re.search(r'<script id="saved-data" type="application/json">([\s\S]*?)</script>', panel)
    if payload is None:
        issues.append('Interactive panel has no saved-data payload')
    else:
        data = json.loads(payload.group(1)); figure = read(root/'assets/figure_data.json')
        if data['source_sha256'] != sha(root/'assets/figure_data.json'):
            issues.append('Interactive panel plotting-input hash differs')
        for panel_key, source_key in [('rows', 'scalar_rows'), ('peak', 'peak'), ('center', 'center_interval'), ('evidence', 'evidence'), ('core', 'selected_core')]:
            if data[panel_key] != figure[source_key]: issues.append('Interactive panel differs: '+panel_key)
        template = (root/'scripts/figures/interactive_template.html').read_text()
        if panel != template.replace('/*__DSIDE_SAVED_DATA__*/', payload.group(1)):
            issues.append('Interactive HTML is stale; rebuild with scripts/figures/build_interactive.py')
    if issues: raise SystemExit('\n'.join(issues))
    print('Verified %d local documentation links, SVG XML, configs, frozen backend hashes, and interactive-panel data.' % links)


if __name__ == '__main__': main()
