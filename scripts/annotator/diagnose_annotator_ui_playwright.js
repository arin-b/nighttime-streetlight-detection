const fs = require('fs');
const path = require('path');
const {chromium} = require('playwright');

const root = path.resolve(__dirname, '..');
const outDir = path.join(root, 'exports', 'annotator_ui_diagnostics');
fs.mkdirSync(outDir, {recursive: true});

const url = process.argv[2] || 'http://127.0.0.1:8791';
const chromePath = fs.existsSync('C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe')
  ? 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe'
  : undefined;

const report = {
  url,
  started_at: new Date().toISOString(),
  console: [],
  page_errors: [],
  actions: [],
  screenshots: [],
};

function logAction(text) {
  report.actions.push({at: new Date().toISOString(), text});
  console.log(text);
}

async function drawBox(page, x1f, y1f, x2f, y2f) {
  const box = await page.locator('#canvas').boundingBox();
  if (!box) throw new Error('Canvas was not visible.');
  const x1 = box.x + box.width * x1f;
  const y1 = box.y + box.height * y1f;
  const x2 = box.x + box.width * x2f;
  const y2 = box.y + box.height * y2f;
  await page.mouse.move(x1, y1);
  await page.mouse.down();
  await page.mouse.move(x2, y2, {steps: 8});
  await page.mouse.up();
}

async function clickCanvas(page, xf, yf) {
  const box = await page.locator('#canvas').boundingBox();
  if (!box) throw new Error('Canvas was not visible.');
  await page.mouse.click(box.x + box.width * xf, box.y + box.height * yf);
}

async function screenshot(page, name) {
  const file = path.join(outDir, `${name}.png`);
  await page.screenshot({path: file, fullPage: true});
  report.screenshots.push(file);
}

async function setStep(page, label) {
  await page.locator('.step-button', {hasText: label}).click();
  await page.waitForTimeout(250);
}

async function clickIfVisible(page, selector) {
  const locator = page.locator(selector);
  if (await locator.count()) {
    await locator.first().click();
    await page.waitForTimeout(150);
  }
}

async function clearCurrentFrame(page) {
  await setStep(page, 'Mark Lamps');
  await clickIfVisible(page, '#deleteAllBoxesBtn');
  await setStep(page, 'Mark Other Lights');
  await clickIfVisible(page, '#deleteAllOtherBoxesBtn');
  await clickIfVisible(page, '#deleteAllShapesBtn');
  await setStep(page, 'Mark Road/Footpath');
  await clickIfVisible(page, '#deleteAllPublicRegionsBtn');
  await setStep(page, 'Mark Lit Area');
  await clickIfVisible(page, '#deleteAllAffectedRegionsBtn');
  await setStep(page, 'Rate Visibility');
  await clickIfVisible(page, '#deleteAllVisibilityLabelsBtn');
  await setStep(page, 'Field Lux / Notes');
  await clickIfVisible(page, '#deleteAllLuxPointsBtn');
}

async function selectFirstSurfaceShape(page) {
  await setStep(page, 'Mark Other Lights');
  await page.locator('#polygonList .row:not(.empty-row)').first().click();
  await page.waitForTimeout(150);
}

const VISUAL_PLANS = [
  {
    pole: [0.52, 0.17, 0.57, 0.58],
    head: [0.51, 0.09, 0.58, 0.20],
    otherBox: [0.08, 0.31, 0.27, 0.46],
    polygon: [[0.06, 0.82], [0.38, 0.66], [0.91, 0.70], [0.98, 0.91], [0.15, 0.91]],
    luxPoint: [0.46, 0.83],
    surfaceType: 'shopfront',
  },
  {
    pole: [0.49, 0.05, 0.53, 0.62],
    head: [0.47, 0.00, 0.55, 0.09],
    otherBox: [0.42, 0.47, 0.53, 0.60],
    polygon: [[0.05, 0.93], [0.35, 0.73], [0.62, 0.63], [0.94, 0.78], [0.95, 0.99], [0.18, 0.98]],
    luxPoint: [0.49, 0.82],
    surfaceType: 'sign_lightbox',
  },
  {
    pole: [0.49, 0.05, 0.53, 0.55],
    head: [0.47, 0.00, 0.55, 0.10],
    otherBox: [0.42, 0.46, 0.53, 0.60],
    polygon: [[0.05, 0.96], [0.38, 0.72], [0.63, 0.63], [0.94, 0.80], [0.95, 0.99], [0.18, 0.99]],
    luxPoint: [0.51, 0.82],
    surfaceType: 'sign_lightbox',
  },
  {
    pole: [0.33, 0.00, 0.38, 0.51],
    head: [0.31, 0.02, 0.39, 0.14],
    otherBox: [0.41, 0.44, 0.52, 0.57],
    polygon: [[0.04, 0.96], [0.34, 0.75], [0.62, 0.61], [0.95, 0.78], [0.97, 0.99], [0.20, 0.99]],
    luxPoint: [0.50, 0.82],
    surfaceType: 'sign_lightbox',
  },
];

async function annotateFrame(page, index) {
  const plan = VISUAL_PLANS[index] || VISUAL_PLANS[VISUAL_PLANS.length - 1];
  logAction(`Annotating frame ${index + 1}`);
  await clearCurrentFrame(page);

  await setStep(page, 'Mark Lamps');
  await page.selectOption('#lampDrawClass', 'streetlight_lamp_head');
  await drawBox(page, 0.05, 0.05, 0.16, 0.16);
  await clickIfVisible(page, '#deleteAllBoxesBtn');
  await page.selectOption('#lampDrawClass', 'streetlight_pole');
  await drawBox(page, ...plan.pole);
  await page.selectOption('#lampDrawClass', 'streetlight_lamp_head');
  await drawBox(page, ...plan.head);

  await setStep(page, 'Mark Other Lights');
  await page.selectOption('#surfaceType', plan.surfaceType);
  await page.click('#otherBoxToolBtn');
  await drawBox(page, 0.01, 0.01, 0.12, 0.12);
  await clickIfVisible(page, '#deleteAllOtherBoxesBtn');
  await page.click('#otherBoxToolBtn');
  await drawBox(page, ...plan.otherBox);
  await page.selectOption('#surfaceType', 'wet_road_reflection');
  await page.click('#polygonToolBtn');
  await clickCanvas(page, 0.10, 0.10);
  await clickCanvas(page, 0.18, 0.10);
  await clickCanvas(page, 0.18, 0.18);
  await page.click('#finishPolygonBtn');
  await clickIfVisible(page, '#deleteAllShapesBtn');
  await page.click('#polygonToolBtn');
  for (const point of plan.polygon) await clickCanvas(page, point[0], point[1]);
  await page.click('#finishPolygonBtn');

  await setStep(page, 'Mark Road/Footpath');
  await page.selectOption('#regionType', 'road');
  await page.click('#addPublicRegionBtn');
  await clickIfVisible(page, '#deleteAllPublicRegionsBtn');
  await selectFirstSurfaceShape(page);
  await setStep(page, 'Mark Road/Footpath');
  await page.click('#addPublicRegionBtn');

  await selectFirstSurfaceShape(page);
  await setStep(page, 'Mark Lit Area');
  await clickIfVisible(page, '#markAffectedNotVisibleBtn');
  await clickIfVisible(page, '#deleteAllAffectedRegionsBtn');
  await selectFirstSurfaceShape(page);
  await setStep(page, 'Mark Lit Area');
  await page.selectOption('#affectedVisibility', 'visible');
  await page.click('#addAffectedRegionBtn');
  await clickIfVisible(page, '#deleteAllAffectedRegionsBtn');
  await selectFirstSurfaceShape(page);
  await setStep(page, 'Mark Lit Area');
  await page.selectOption('#affectedVisibility', 'visible');
  await page.click('#addAffectedRegionBtn');

  await setStep(page, 'Rate Visibility');
  await page.selectOption('#lampStatus', 'on');
  await page.click('#addLampStatusBtn');
  await page.selectOption('#visibilityClass', 'adequate');
  await page.click('#addVisibilityBtn');
  await page.selectOption('#attributionClass', 'mixed');
  await page.fill('#attributionEvidence', 'diagnostic per-lamp rating');
  await page.click('#addAttributionBtn');

  await setStep(page, 'Field Lux / Notes');
  await page.click('#pointToolBtn');
  await clickCanvas(page, ...plan.luxPoint);
  await page.selectOption('#luxType', 'P5');
  await page.click('#addLuxBtn');
  await clickIfVisible(page, '#deleteAllLuxPointsBtn');
  await clickCanvas(page, ...plan.luxPoint);
  await page.click('#addLuxBtn');
  await page.fill('#qaFlag', `ui_diagnostic_frame_${index + 1}`);
  await page.click('#addQaBtn');

  await screenshot(page, `frame_${String(index + 1).padStart(2, '0')}_before_accept`);
  await page.click('#acceptedBtn');
  await page.waitForTimeout(1400);
  await screenshot(page, `frame_${String(index + 1).padStart(2, '0')}_after_accept`);
}

(async () => {
  const browser = await chromium.launch({
    headless: true,
    executablePath: chromePath,
  });
  const page = await browser.newPage({viewport: {width: 1366, height: 768}});
  page.on('console', message => report.console.push({type: message.type(), text: message.text()}));
  page.on('pageerror', error => report.page_errors.push(String(error.stack || error)));

  try {
    await page.goto(url, {waitUntil: 'networkidle'});
    await page.waitForSelector('#canvas', {timeout: 15000});
    await screenshot(page, 'initial');
    for (let index = 0; index < 4; index++) {
      await annotateFrame(page, index);
    }
    await screenshot(page, 'final');
  } finally {
    report.finished_at = new Date().toISOString();
    fs.writeFileSync(path.join(outDir, 'report.json'), JSON.stringify(report, null, 2));
    await browser.close();
  }
})();
