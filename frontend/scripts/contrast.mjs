// WCAG 2.1 contrast of every text/background pair the theme uses; tokens are read from src/theme.css.
// Usage: node scripts/contrast.mjs   (exit 1 if a pair is below its threshold)
import { readFileSync } from "node:fs";

const css = readFileSync(new URL("../src/theme.css", import.meta.url), "utf8");
const root = css.slice(css.indexOf(":root"), css.indexOf("}", css.indexOf(":root")));
const tokens = Object.fromEntries([...root.matchAll(/--([\w-]+):\s*(#[0-9a-fA-F]{6})/g)].map((m) => [m[1], m[2]]));

const lin = (c) => (c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
const lum = (hex) => {
  const [r, g, b] = [1, 3, 5].map((i) => lin(parseInt(hex.slice(i, i + 2), 16) / 255));
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
};
const ratio = (a, b) => {
  const [x, y] = [lum(a), lum(b)].sort((p, q) => q - p);
  return (x + 0.05) / (y + 0.05);
};

// [foreground, background, use, need]: need 4.5 for normal text; 3.0 for large text (bold >= 18.66 px or >= 24 px)
// and for non-text UI parts (focus ring, borders, rail dashes; WCAG 1.4.11). Only pairs the UI actually uses.
const PAIRS = [
  ["text", "bg", "body text, hero QUANTA, headline second word, unpressed pill labels", 4.5],
  ["muted", "bg", "secondary text and field labels on the page", 4.5],
  ["periwinkle", "bg", "headline first word (Bebas Neue >= 32 px)", 3.0],
  ["periwinkle", "bg", "focus ring on the page (non-text)", 3.0],
  ["text", "bg", "active nav underline (non-text)", 3.0],
  ["text", "bg-deep", "rail active label, phone menu active link, footer", 4.5],
  ["muted", "bg-deep", "rail labels, phone menu links, footer text", 4.5],
  ["coral", "bg-deep", "rail dashes, menu close button (non-text)", 3.0],
  ["periwinkle", "bg-deep", "focus ring on the rail and menu (non-text)", 3.0],
  ["text", "panel", "card, table, chart and drawer text", 4.5],
  ["muted", "panel", "secondary text in cards, axis ticks, muted valid_quantized cross", 4.5],
  ["periwinkle", "panel", "references, chart legend, result value: quantum queries", 4.5],
  ["peach", "panel", "result value: above the optimum", 4.5],
  ["coral", "panel", "round arrow buttons (non-text)", 3.0],
  ["ok", "panel", "valid check, result value: less energy than the baseline", 4.5],
  ["bad", "panel", "invalid cross, error title, result value: more energy than the baseline", 4.5],
  ["text", "panel-2", "raised surfaces: counters, notices, tooltips, highlighted row", 4.5],
  ["muted", "panel-2", "counter labels on raised surfaces", 4.5],
  ["ok", "panel-2", "matches-cached check in the notice", 4.5],
  ["periwinkle", "panel-2", "active stepper state", 4.5],
  ["ink", "coral", "coral button and pressed pill labels (white on coral is never used: 2.88)", 4.5],
  ["text", "navy", "navy button label", 4.5],
  ["text", "blob-blue", "hero headline where it touches the blue blob (phone)", 3.0],
];

let fail = 0;
console.log(`tokens: ${JSON.stringify(tokens)}`);
console.log("fg / bg | hex | ratio | need | result | use");
for (const [fg, bg, use, need] of PAIRS) {
  const r = ratio(tokens[fg], tokens[bg]);
  const ok = r >= need;
  fail += !ok;
  console.log(`${fg} / ${bg} | ${tokens[fg]} on ${tokens[bg]} | ${r.toFixed(2)} | ${need.toFixed(1)} | ${ok ? "PASS" : "FAIL"} | ${use}`);
}
console.log(fail ? `CONTRAST FAIL (${fail} pairs)` : `CONTRAST PASS (${PAIRS.length} pairs)`);
process.exit(fail ? 1 : 0);
