const fs = require('fs');
const path = require('path');
const {chromium} = require('playwright');

const root = path.resolve(__dirname, '..');
const outDir = path.join(root, 'exports', 'annotator_ui_diagnostics', 'frame_screens');
fs.mkdirSync(outDir, {recursive: true});

const url = process.argv[2] || 'http://127.0.0.1:8791';
const chromePath = fs.existsSync('C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe')
  ? 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe'
  : undefined;

(async () => {
  const browser = await chromium.launch({headless: true, executablePath: chromePath});
  const page = await browser.newPage({viewport: {width: 1366, height: 768}});
  await page.goto(url, {waitUntil: 'networkidle'});
  await page.waitForSelector('#canvas', {timeout: 15000});
  for (let index = 0; index < 4; index++) {
    if (index > 0) {
      await page.click('#nextBtn');
      await page.waitForTimeout(700);
    }
    await page.screenshot({path: path.join(outDir, `frame_${index + 1}.png`), fullPage: true});
  }
  await browser.close();
})();
