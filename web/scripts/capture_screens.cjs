const fs = require('fs');
const path = require('path');
const puppeteer = require('puppeteer-core');

const OUT_DIR = path.resolve(__dirname, '../../results/ui_screens');
if (!fs.existsSync(OUT_DIR)) {
  fs.mkdirSync(OUT_DIR, { recursive: true });
}

async function run() {
  console.log('Launching browser to capture screenshots...');
  const browser = await puppeteer.launch({
    executablePath: '/usr/bin/chromium-browser',
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--hide-scrollbars'],
    defaultViewport: {
      width: 1440,
      height: 900,
      deviceScaleFactor: 2,
    },
  });

  const page = await browser.newPage();

  // 1. Console Empty State at 1440x900
  console.log('1. Capturing console empty state...');
  await page.goto('http://localhost:8000', { waitUntil: 'networkidle0' });
  await new Promise((r) => setTimeout(r, 600));
  await page.screenshot({
    path: path.join(OUT_DIR, 'console_empty_1440.png'),
    fullPage: false,
  });

  // 2. Load demo cases from API
  const demoCases = await page.evaluate(async () => {
    const res = await fetch('/api/demo-cases');
    return await res.json();
  });

  console.log(`Found ${demoCases.length} demo cases.`);

  // 3. Capture each demo case at 1440x900
  for (let i = 0; i < demoCases.length; i++) {
    const dc = demoCases[i];
    console.log(`Capturing [${i + 1}/${demoCases.length}]: ${dc.id}...`);

    // Reset viewport to desktop
    await page.setViewport({ width: 1440, height: 900, deviceScaleFactor: 2 });

    // Open popover
    await page.evaluate(() => {
      const btns = Array.from(document.querySelectorAll('button'));
      const demoBtn = btns.find((b) => b.textContent.includes('Demo cases'));
      if (demoBtn) demoBtn.click();
    });

    await new Promise((r) => setTimeout(r, 300));

    // Turn on annotations if first case
    if (i === 0) {
      await page.evaluate(() => {
        const checkbox = document.querySelector('input[type="checkbox"]');
        if (checkbox && !checkbox.checked) {
          checkbox.click();
        }
      });
      await new Promise((r) => setTimeout(r, 150));
    }

    // Click the case by data-case-id
    await page.evaluate((caseId) => {
      const btn = document.querySelector(`button[data-case-id="${caseId}"]`);
      if (btn) {
        btn.click();
      } else {
        console.error('Button not found for case:', caseId);
      }
    }, dc.id);

    // Wait for verdict and animations
    await new Promise((r) => setTimeout(r, 800));

    // Desktop screenshot
    await page.screenshot({
      path: path.join(OUT_DIR, `${dc.id}_1440.png`),
      fullPage: false,
    });

    // Mobile screenshot at 390x844 for case 1 and case 3
    if (dc.id.includes('case1_') || dc.id.includes('case3_gaming_attempt_defended')) {
      console.log(`  Capturing mobile 390x844 for ${dc.id}...`);
      await page.setViewport({ width: 390, height: 844, deviceScaleFactor: 2 });
      await new Promise((r) => setTimeout(r, 400));
      await page.screenshot({
        path: path.join(OUT_DIR, `${dc.id}_mobile_390.png`),
        fullPage: true,
      });
      // Restore desktop viewport
      await page.setViewport({ width: 1440, height: 900, deviceScaleFactor: 2 });
    }
  }

  await browser.close();
  console.log(`All screenshots saved to ${OUT_DIR}`);
}

run().catch((err) => {
  console.error('Screenshot capture failed:', err);
  process.exit(1);
});
