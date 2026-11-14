// G3.6 screenshots with Microsoft Edge headless driven over the DevTools protocol (Node's built-in WebSocket, no
// dependency). Device-metrics emulation gives a real 390 px layout (a plain --window-size below ~500 px is not honoured
// by new-headless on Windows). For every page x size x viewport: a viewport capture, a full-page capture, and an
// overflow measurement (document scrollWidth vs viewport width, and elements whose right edge passes the viewport).
// Usage: node shots.mjs <outdir> [base url] [page filter]
import { spawn, spawnSync } from "node:child_process";
import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

const [out, base = "http://127.0.0.1:5000", only = ""] = process.argv.slice(2);
mkdirSync(out, { recursive: true });
const EDGE = "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe";
const PAGES = { overview: "/", pipeline: "/pipeline", servers: "/servers", compare: "/compare", quantum: "/quantum", about: "/about" };
const SIZES = ["small_v2", "large_v2"];
const VIEWS = [{ n: "desktop", w: 1440, h: 900, mobile: false }, { n: "phone", w: 390, h: 844, mobile: true }];
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const port = 9333;
const edge = spawn(EDGE, ["--headless=new", ...(process.env.GPU ? ["--enable-unsafe-swiftshader", "--use-angle=swiftshader", "--enable-webgl", "--ignore-gpu-blocklist"] : ["--disable-gpu"]), "--hide-scrollbars", `--remote-debugging-port=${port}`,
  `--user-data-dir=${join(tmpdir(), "gq-edge-cdp")}`, "about:blank"], { stdio: "ignore" });
let target;
for (let i = 0; i < 50 && !target; i++) {
  await sleep(200);
  try { target = (await (await fetch(`http://127.0.0.1:${port}/json/list`)).json()).find((t) => t.type === "page"); } catch { /* not up */ }
}
const ws = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((r) => ws.addEventListener("open", r, { once: true }));
let id = 0;
const waiting = new Map();
ws.addEventListener("message", (m) => {
  const msg = JSON.parse(m.data);
  if (msg.id && waiting.has(msg.id)) { waiting.get(msg.id)(msg); waiting.delete(msg.id); }
});
const send = (method, params = {}) => new Promise((resolve, reject) => {
  const n = ++id;
  waiting.set(n, (msg) => (msg.error ? reject(new Error(`${method}: ${msg.error.message}`)) : resolve(msg.result)));
  ws.send(JSON.stringify({ id: n, method, params }));
});
const evaluate = async (expr) => (await send("Runtime.evaluate", { expression: expr, returnByValue: true, awaitPromise: true })).result.value;

await send("Page.enable");
await send("Network.enable");
const outside = new Set();
ws.addEventListener("message", (m) => { const msg = JSON.parse(m.data); if (msg.method === "Runtime.exceptionThrown") console.log("EXCEPTION", JSON.stringify(msg.params.exceptionDetails).slice(0, 600)); if (msg.method === "Runtime.consoleAPICalled" && msg.params.type === "error") console.log("CONSOLE.ERROR", msg.params.args.map((a) => a.value ?? a.description).join(" ").slice(0, 600)); if (msg.method === "Network.requestWillBeSent") { const u = msg.params.request.url; if (!u.startsWith(base) && !u.startsWith("data:") && !u.startsWith("blob:") && !u.startsWith("chrome-extension:")) outside.add(u); } });
await send("Runtime.enable");
const report = [];
for (const [page, path] of Object.entries(PAGES)) {
  if (only && page !== only) continue;
  for (const size of SIZES) {
    for (const v of VIEWS) {
      await send("Emulation.setDeviceMetricsOverride", { width: v.w, height: v.h, deviceScaleFactor: 1, mobile: v.mobile });
      await send("Page.navigate", { url: `${base}${path}?size=${size}` });
      // wait until the API data has rendered: no skeleton, no pending fetch, and a settle delay
      for (let i = 0; i < 120; i++) {
        await sleep(250);
        const ready = await evaluate(`document.readyState === "complete" && !!document.querySelector("main")
          && !document.querySelector('[aria-busy="true"]')`).catch(() => false);
        if (ready) break;
      }
      await sleep(process.env.GPU ? (Number(process.env.WAIT) || 4000) : 900);
      const m = await evaluate(`(() => {
        const W = document.documentElement.clientWidth, over = [];
        document.querySelectorAll("body *").forEach((e) => {
          if (e.closest(".blob-frame, .table-wrap, .recharts-wrapper")) return;
          const r = e.getBoundingClientRect();
          if (r.width && r.right > W + 1) over.push(e.tagName.toLowerCase() + "." + String(e.className).split(" ")[0] + "@" + Math.round(r.right));
        });
        return { W, scrollW: document.documentElement.scrollWidth, H: document.documentElement.scrollHeight,
          errors: document.querySelectorAll('[role="alert"]').length, over: over.slice(0, 8) };
      })()`);
      const name = `${page}-${size}-${v.n}`;
      const shot = await send("Page.captureScreenshot", { format: "png" });
      writeFileSync(join(out, `${name}.png`), Buffer.from(shot.data, "base64"));
      const full = await send("Page.captureScreenshot", { format: "png", captureBeyondViewport: true,
        clip: { x: 0, y: 0, width: v.w, height: Math.min(m.H, 8000), scale: 1 } });
      writeFileSync(join(out, `${name}-full.png`), Buffer.from(full.data, "base64"));
      const ok = m.scrollW <= m.W && m.over.length === 0 && m.errors === 0;
      report.push(`${ok ? "OK  " : "FAIL"} ${name}: viewport ${m.W} px, scrollWidth ${m.scrollW}, page height ${m.H}, error panels ${m.errors}${m.over.length ? ", overflow: " + m.over.join(" ") : ""}`);
      console.log(report.at(-1));
    }
  }
}
if (!only || only === "menu") {
  await send("Emulation.setDeviceMetricsOverride", { width: 390, height: 844, deviceScaleFactor: 1, mobile: true });
  await send("Page.navigate", { url: `${base}/compare?size=large_v2` });
  await sleep(3000);
  await evaluate(`document.querySelector(".hamburger").click()`);
  await sleep(600);
  const open = await evaluate(`!!document.querySelector("#phone-menu")`);
  const shot = await send("Page.captureScreenshot", { format: "png" });
  writeFileSync(join(out, "menu-large_v2-phone.png"), Buffer.from(shot.data, "base64"));
  report.push(`${open ? "OK  " : "FAIL"} menu-large_v2-phone: hamburger opens the menu: ${open}`);
  console.log(report.at(-1));
}
if (!only || only === "drawer") {
  await send("Emulation.setDeviceMetricsOverride", { width: 1440, height: 900, deviceScaleFactor: 1, mobile: false });
  await send("Page.navigate", { url: `${base}/servers?size=large_v2` });
  for (let i = 0; i < 80 && !(await evaluate(`!!document.querySelector(".tile.on")`)); i++) await sleep(250);
  await evaluate(`document.querySelector(".tile.on").click()`);
  await sleep(700);
  const top = await evaluate(`(() => { const d = document.querySelector(".drawer").getBoundingClientRect();
    const el = document.elementFromPoint(d.left + d.width - 30, 30); return el ? el.closest(".drawer") !== null : false; })()`);
  const shot = await send("Page.captureScreenshot", { format: "png" });
  writeFileSync(join(out, "drawer-large_v2-desktop.png"), Buffer.from(shot.data, "base64"));
  report.push(`${top ? "OK  " : "FAIL"} drawer-large_v2-desktop: drawer is on top at its top-right corner: ${top}`);
  console.log(report.at(-1));
}
if (only === "result") {
  await send("Emulation.setDeviceMetricsOverride", { width: 1440, height: 900, deviceScaleFactor: 1, mobile: false });
  await send("Page.navigate", { url: `${base}/pipeline?size=small_v2` });
  await sleep(3000);
  await evaluate(`[...document.querySelectorAll("button")].find((b) => b.textContent === "Run live").click()`);
  for (let i = 0; i < 240 && !(await evaluate(`!![...document.querySelectorAll("button")].find((b) => b.textContent === "View results")`)); i++) await sleep(500);
  await evaluate(`[...document.querySelectorAll("button")].find((b) => b.textContent === "View results").click()`);
  await sleep(800);
  const open = await evaluate(`!!document.querySelector("[role=dialog][aria-label=\\"Run result\\"]")`);
  const shot = await send("Page.captureScreenshot", { format: "png" });
  writeFileSync(join(out, "result-small_v2-desktop.png"), Buffer.from(shot.data, "base64"));
  report.push(`${open ? "OK  " : "FAIL"} result-small_v2-desktop: result window open: ${open}`);
  console.log(report.at(-1));
}
report.push(`outside requests: ${outside.size ? [...outside].join(" ") : "none"}`);
console.log(report.at(-1));
writeFileSync(join(out, "overflow_report.txt"), report.join("\n") + "\n");
ws.close();
// Edge's launcher exits early and its GPU/renderer children are re-parented, so kill by our private profile folder
spawnSync("powershell", ["-NoProfile", "-Command", "Get-CimInstance Win32_Process -Filter \"Name='msedge.exe'\" | "
  + "Where-Object { $_.CommandLine -match 'gq-edge-cdp' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"]);
process.exit(report.some((r) => r.startsWith("FAIL")) ? 1 : 0);
