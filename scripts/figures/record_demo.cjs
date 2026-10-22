/* Capture real controls from the committed offline panel; never refit models.
 * Optional tooling: Playwright 1.48.2 with Chromium. See assets/README.md.
 */
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const {chromium} = require('playwright');
const root = path.resolve(__dirname, '../..');
const output = path.join(root, 'runs/animation-frames');
const hash = p => crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');

(async () => {
  if (fs.existsSync(path.join(output, 'frames.json'))) {
    throw new Error('Use a fresh runs/animation-frames directory to preserve an existing recording.');
  }
  fs.mkdirSync(output, {recursive:true});
  const browser = await chromium.launch({headless:true, args:['--no-sandbox']});
  try {
    const page = await browser.newPage({viewport:{width:1240, height:1500}, deviceScaleFactor:1});
    const errors = [];
    page.on('pageerror', error => errors.push(String(error)));
    await page.goto('file://' + path.join(root, 'docs/interactive/index.html'));
    await page.waitForFunction(() => window.DSIDE_PANEL_STATE);
    // Crop the website's long introduction/footer. Keep all controls and panels.
    await page.evaluate(() => {
      const wrapper = document.createElement('section');
      wrapper.id = 'recording';
      const heading = document.createElement('div');
      heading.className = 'demo-heading';
      heading.innerHTML = '<strong>D-SIDE · Interactive algorithm</strong><span>Automatic demonstration</span>';
      const action = document.createElement('p'); action.id = 'demo-action';
      const controls = document.querySelector('.controls');
      controls.before(wrapper);
      wrapper.append(heading, action, controls, document.querySelector('.panels'));
    });
    await page.addStyleTag({content:`
      #recording{background:#f5f8fb;padding:16px 0 8px}
      .demo-heading{display:flex;justify-content:space-between;align-items:baseline;gap:20px;margin:0 0 7px;color:#102a43}
      .demo-heading strong{font-size:23px}.demo-heading span{font-size:13px;color:#60768b}
      #demo-action{font-size:15px;height:26px;margin:0 0 14px;color:#087f8c}
      .matrix button{font-size:12px}.controls{margin-bottom:16px}
    `});
    const frames = [];
    async function capture(action, duration) {
      await page.locator('#demo-action').evaluate((el, text) => el.textContent=text, action);
      await page.waitForTimeout(100);
      const state = await page.evaluate(() => window.DSIDE_PANEL_STATE);
      if (JSON.stringify(state.selected_sources)!=='[0,1,2,3,4,5]') throw new Error('Frozen clustering changed');
      const file = String(frames.length).padStart(3, '0')+'.png';
      await page.locator('#recording').screenshot({path:path.join(output, file)});
      frames.push({file, duration_ms:duration, action, pair:state.pair,
        index:state.index, statistic:state.statistic, selected_sources:state.selected_sources});
    }
    async function sweep(action, count) {
      const max = Number(await page.locator('#sample').getAttribute('max'));
      const indices = [...new Set(Array.from({length:count}, (_, i) => Math.round(i*max/(count-1))))];
      for (const index of indices) {
        await page.locator('#sample').fill(String(index));
        await capture(action, 350);
      }
    }
    await capture('1 · Choose a source pair: sources 0 and 5', 1400);
    await sweep('2 · Move the slider: follow matched points and their statistic', 11);
    await page.locator('#evidence button[data-pair="0_6"]').first().click();
    await capture('3 · Click an evidence cell: inspect sources 0 and 6', 1200);
    await sweep('The selected pair changes; the global fit and clustering stay fixed', 8);
    await page.selectOption('#source-pair', '6_7');
    await capture('4 · Switch sources: inspect sources 6 and 7', 1200);
    await sweep('Compare saved statistics across source pairs', 8);
    await page.selectOption('#source-pair', '0_5');
    await page.locator('#sample').fill('10');
    await capture('Click this preview to explore the interactive panel yourself', 2200);
    if (errors.length) throw new Error(JSON.stringify(errors));
    fs.writeFileSync(path.join(output, 'frames.json'), JSON.stringify({
      illustration_seed:2026103000, panel_sha256:hash(path.join(root, 'docs/interactive/index.html')),
      plotting_bundle_sha256:hash(path.join(root, 'assets/figure_data.json')),
      viewport:{width:1240,height:1500}, browser:await browser.version(),
      frames, javascript_errors:errors,
      scope:'Real offline panel screenshots. Only displayed pair/sample changes; no model is refitted.'
    }, null, 2)+'\n');
    console.log('Captured '+frames.length+' real panel frames.');
  } finally { await browser.close(); }
})().catch(error => {console.error(error); process.exit(1);});
