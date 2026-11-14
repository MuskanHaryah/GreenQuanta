// G3.4 live run through the Vite dev server's /api proxy, exactly as the Pipeline page does it:
// POST /api/runs, then GET /api/runs/<id> every 1 s until done/failed; also a second POST to see the 409 attach body.
const base = process.argv[2] ?? "http://localhost:5173";
const size = process.argv[3] ?? "small_v2";
const stage2 = process.argv[4] ?? "csqa";
const t0 = Date.now();
const log = (...a) => console.log(`[${((Date.now() - t0) / 1000).toFixed(1)} s]`, ...a);

const page = await fetch(base + "/");
log("GET / (dev server page)", page.status, (await page.text()).includes('<div id="root">') ? "has #root" : "NO #root");
const post = await fetch(base + "/api/runs", {
  method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ size, stage2 }),
});
const { job_id } = await post.json();
log("POST /api/runs", post.status, JSON.stringify({ size, stage2 }), "-> job_id", job_id);
const again = await fetch(base + "/api/runs", {
  method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ size, stage2 }),
});
log("second POST /api/runs", again.status, JSON.stringify(await again.json()));

let job, seen = 0;
for (;;) {
  await new Promise((r) => setTimeout(r, 1000));
  job = await (await fetch(`${base}/api/runs/${job_id}`)).json();
  for (const e of job.events.slice(seen)) log(`event ${e.kind} at t=${e.t} s`);
  seen = job.events.length;
  if (job.status === "done" || job.status === "failed") break;
}
log("status", job.status, "elapsed_s", job.elapsed_s, "error", job.error);
if (job.result) {
  const o = job.result.out;
  log("result", JSON.stringify({ chosen_ids: o.chosen_ids, real_valid: o.real_valid, quantized_valid: o.quantized_valid,
    linear_joules: o.linear_joules, diff_r1_pct: o.diff_r1_pct, gap_to_h_min_pct: o.gap_to_h_min_pct }));
}
const cached = await (await fetch(`${base}/api/results/${size}`)).json();
const m = cached.methods[stage2 === "csqa" ? "quantum" : "control"];
log("cached", JSON.stringify({ chosen_ids: cached.stage1.chosen_ids, valid_real: m.valid_real,
  valid_quantized: m.valid_quantized, linear_joules: m.linear_joules, diff_r1_pct: m.diff_r1_pct }));
log("matches_cached", job.matches_cached);
console.log(job.status === "done" && job.matches_cached === true ? "LIVE RUN PASS" : "LIVE RUN FAIL");
process.exit(job.matches_cached === true ? 0 : 1);
