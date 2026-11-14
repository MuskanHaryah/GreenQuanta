import { useCallback, useEffect, useState } from "react";
import { api, ApiError, type Job, type Results, type RunEvent } from "../api";
import { useApi, useSize } from "../hooks";
import { int, joules, kg, kwh, seconds, signedPct, sizeName, two } from "../format";
import { bestBaseline, relDiffPct } from "../selectors";
import { methodName } from "../methods";
import { Async, Modal, Section } from "../components/ui";

export const POLL_MS = 1000;
const TERMINAL = new Set(["done", "failed", "cancelled"]);

const STEPS = [
  { title: "Choose servers", text: "Grover search looks for the cheapest set of servers to switch on. Each round it "
    + "boosts the sets that beat the best one found so far." },
  { title: "Place VMs", text: "Quantum annealing places every VM on the chosen servers. Its moves always keep each "
    + "VM on exactly one server." },
  { title: "Check", text: "Every placement is checked the same way. Each VM is placed once and no server goes over "
    + "its CPU or memory." },
  { title: "Compare", text: "The result is compared with the saved run, whose placement was also replayed in the "
    + "CloudSim Plus simulator to confirm the energy." },
];

type StepState = "idle" | "active" | "done";
const event = (job: Job, kind: RunEvent["kind"]) => job.events.find((e) => e.kind === kind)?.data;
const num = (x: unknown) => (typeof x === "number" ? x : 0);

/** Stepper state from the run's events: PROPOSALS → Stage 1 done; STAGE2 → Stage 2 progress; HYBRID → validated;
 *  RESULT → done (the CloudSim replay is the cached one). */
export function stepStates(job: Job | null): StepState[] {
  if (!job) return ["idle", "idle", "idle", "idle"];
  const has = (k: RunEvent["kind"]) => job.events.some((e) => e.kind === k);
  const stage2Solves = job.events.filter((e) => e.kind === "STAGE2").length;
  const running = !TERMINAL.has(job.status);
  const s1: StepState = has("PROPOSALS") || has("SKIP") ? "done" : running ? "active" : "idle";
  const s3: StepState = has("HYBRID") ? "done" : s1 === "done" && stage2Solves > 0 && running ? "active" : "idle";
  const s2: StepState = has("HYBRID") ? "done" : s1 === "done" && running ? "active" : stage2Solves > 0 ? "done" : "idle";
  const s4: StepState = has("RESULT") && job.status === "done" ? "done" : "idle";
  return [s1, s2, s3, s4];
}

/** What each step actually did in this run, from its event data. */
export function stepDetails(job: Job | null): Array<string | null> {
  if (!job) return [null, null, null, null];
  const p = event(job, "PROPOSALS"), s = event(job, "STAGE2"), h = event(job, "HYBRID");
  const ids = (x: unknown) => (Array.isArray(x) ? x.join(", ") : "");
  return [
    p ? `${num(p.gas_runs)} search runs, ${int(num(p.gas_oracle_calls))} oracle calls, `
      + `${int(num(p.gas_measurements))} measurements.` : null,
    s ? `Servers ${ids(s.active_ids)} switched on. Placed in ${num(s.time_s).toFixed(1)} s.` : null,
    h ? `${h.real_valid ? "Every VM fits." : "Some VMs do not fit."} ${signedPct(num(h.diff_r1_pct))} above the exact optimum.`
      : null,
    job.status === "done" ? (job.matches_cached ? "Same as the saved result." : "Differs from the saved result.") : null,
  ];
}

const STATE_TEXT: Record<StepState, string> = { idle: "waiting", active: "running…", done: "done" };

function Stepper({ job }: { job: Job | null }) {
  const states = stepStates(job), details = stepDetails(job);
  return (
    <ol className="stepper" aria-label="pipeline stages">
      {STEPS.map((s, i) => (
        <li key={s.title} className="step" data-state={states[i]} data-testid={`step-${i + 1}`}>
          <div className="numeral" aria-hidden="true">{two(i + 1)}</div>
          <h3>{s.title}</h3>
          <p>{s.text}</p>
          {job && <span className="state">{STATE_TEXT[states[i]]}</span>}
          {details[i] && <p className="step-live" data-testid={`detail-${i + 1}`}>{details[i]}</p>}
        </li>
      ))}
    </ol>
  );
}

/** The run's input: how many VMs, and the servers it can use, grouped by type. */
export function InputData({ r }: { r: Results }) {
  const types = [...new Set(r.problem.servers.map((s) => s.type))];
  const range = (xs: number[], f: (x: number) => string) => {
    const lo = Math.min(...xs), hi = Math.max(...xs);
    return lo === hi ? f(lo) : `${f(lo)} to ${f(hi)}`;
  };
  return (
    <div className="input-data" data-testid="input-data">
      <dl className="kv">
        <div><dt>Virtual machines</dt><dd>{int(r.problem.vms)}</dd></div>
        <div><dt>Servers</dt><dd>{int(r.problem.servers.length)} of {types.length} types</dd></div>
        <div><dt>Server combinations</dt><dd>{int(r.stage1.states)}</dd></div>
      </dl>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Type</th><th className="num">Servers</th><th className="num">CPU limit</th>
              <th className="num">Memory limit</th><th className="num">Power idle</th><th className="num">Power full</th>
            </tr>
          </thead>
          <tbody>
            {types.map((t) => {
              const of = r.problem.servers.filter((s) => s.type === t);
              return (
                <tr key={t}>
                  <td>Type {t}</td>
                  <td className="num">{of.length}</td>
                  <td className="num">{range(of.map((s) => s.cpu_limit), (x) => x.toFixed(4))}</td>
                  <td className="num">{range(of.map((s) => s.mem_limit), (x) => x.toFixed(4))}</td>
                  <td className="num">{range(of.map((s) => s.idle_w), (x) => `${x.toFixed(0)} W`)}</td>
                  <td className="num">{range(of.map((s) => s.max_w), (x) => `${x.toFixed(0)} W`)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="muted" style={{ margin: 0, fontSize: 13 }}>CPU and memory are in normalised units, as in the data set.</p>
    </div>
  );
}

function Row({ label, children, testId }: { label: string; children: React.ReactNode; testId?: string }) {
  return <div className="result-row" data-testid={testId}><dt>{label}</dt><dd>{children}</dd></div>;
}

/** The finished run, laid out as a short report. Numbers come from the run's RESULT and, for the comparison with the
 *  best baseline, from the saved results of the same size. */
export function ResultWindow({ job, saved, onClose }: { job: Job; saved?: Results; onClose: () => void }) {
  const out = job.result!.out, energy = job.result!.energy;
  const best = saved && bestBaseline(Object.entries(saved.methods));
  const vsBest = best?.[1].linear_joules != null ? relDiffPct(out.linear_joules, best[1].linear_joules) : null;
  const servers = saved?.problem.servers.length;
  return (
    <Modal title="Run result" onClose={onClose}
      footer={<button className="btn btn-coral" onClick={onClose}>Close</button>}>
      <p className="muted result-sub">Size {sizeName(job.size)}, finished in {seconds(job.elapsed_s)}.</p>
      <div className="result-highlights">
        <div><span>Above the exact optimum</span><strong className="tone-peach">{signedPct(out.diff_r1_pct)}</strong></div>
        <div>
          <span>Against {best ? methodName(best[0]) : "the best baseline"}</span>
          <strong className={vsBest != null && vsBest <= 0 ? "tone-ok" : "tone-bad"}>{signedPct(vsBest)}</strong>
        </div>
        <div><span>Servers switched on</span><strong className="tone-periwinkle">
          {out.chosen_ids.length}{servers ? ` of ${servers}` : ""}</strong></div>
      </div>
      <h3 className="result-head">Placement</h3>
      <dl className="result-rows">
        <Row label="Servers used">{out.chosen_ids.join(", ")}</Row>
        <Row label="Every VM fits its server" testId="row-valid">{out.real_valid ? "Yes" : "No"}</Row>
        <Row label="Same as the saved result" testId="row-matches">{job.matches_cached ? "Yes" : "No"}</Row>
      </dl>
      <h3 className="result-head">Energy for one day</h3>
      <dl className="result-rows">
        <Row label="Energy, optimised model">{joules(out.linear_joules)}</Row>
        {energy && <Row label="Server energy">{kwh(energy.it_energy_kwh)}</Row>}
        {energy && <Row label="Whole facility energy">{kwh(energy.facility_energy_kwh)}</Row>}
        {energy && <Row label="CO₂">{kg(energy.co2_kg)}</Row>}
      </dl>
      <h3 className="result-head">Quantum work</h3>
      <dl className="result-rows">
        <Row label="Grover search runs">{int(out.gas_runs)}</Row>
        <Row label="Oracle calls and measurements">{int(out.gas_oracle_calls)} and {int(out.gas_measurements)}</Row>
        <Row label="Quantum annealing time">{seconds(out.stage2_s)}</Row>
      </dl>
    </Modal>
  );
}

const STATUS_TEXT: Record<Job["status"], string> = {
  queued: "Starting", running: "Running", done: "Finished", failed: "Failed", cancelled: "Stopped",
};

export default function Pipeline() {
  const [size] = useSize();
  const results = useApi(api.results, size);
  const [jobId, setJobId] = useState<string | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [starting, setStarting] = useState(false);
  const [stopping, setStopping] = useState(false);
  const [message, setMessage] = useState<{ text: string; bad?: boolean } | null>(null);
  const [showResult, setShowResult] = useState(false);
  const closeResult = useCallback(() => setShowResult(false), []);

  useEffect(() => {
    if (!jobId) return;
    let live = true, timer: ReturnType<typeof setTimeout> | undefined;
    const poll = () => api.run(jobId).then((j) => {
      if (!live) return;
      setJob(j);
      if (TERMINAL.has(j.status)) setStopping(false);
      else timer = setTimeout(poll, POLL_MS);
    }, (e: ApiError) => live && setMessage({ text: `Lost contact with the run. ${e.message}`, bad: true }));
    poll();
    return () => { live = false; clearTimeout(timer); };
  }, [jobId]);

  const running = starting || (job !== null && !TERMINAL.has(job.status));
  const start = async () => {
    setStarting(true);
    setMessage(null);
    setJob(null);
    setShowResult(false);
    try {
      setJobId((await api.startRun(size, "csqa")).job_id);
    } catch (e) {
      const err = e as ApiError;
      if (err.status === 409 && err.jobId) {
        setMessage({ text: "A run is already going, showing that one." });
        setJobId(err.jobId);
      } else {
        setMessage({ text: `Could not start the run. ${err.message}`, bad: true });
      }
    } finally {
      setStarting(false);
    }
  };
  const stop = async () => {
    if (!jobId) return;
    setStopping(true);
    try {
      await api.cancelRun(jobId);
    } catch (e) {
      setStopping(false);
      setMessage({ text: `Could not stop the run. ${(e as ApiError).message}`, bad: true });
    }
  };

  return (
    <>
      <Section id="live" first="Live" second="Run">
        <div className="live">
          <p style={{ margin: 0 }}>
            Runs the quantum pipeline again on the server for size <b>{sizeName(size)}</b>, on the data below. It uses
            the same settings as the saved result, so the answer should match. Small takes about half a minute, Large a
            bit over a minute.
          </p>
          <Async state={results}>{(r) => <InputData r={r} />}</Async>
          <div className="live-controls">
            <button className="btn btn-coral" onClick={start} disabled={running}>{running ? "Running" : "Run live"}</button>
            {running && jobId && (
              <button className="btn btn-ghost" onClick={stop} disabled={stopping} aria-label="stop the run">
                <span aria-hidden="true">■</span> {stopping ? "Stopping" : "Stop"}
              </button>
            )}
            {job?.status === "done" && job.result && (
              <button className="btn btn-navy" onClick={() => setShowResult(true)}>View results</button>
            )}
            {job && <span aria-live="polite">
              {job.status === "queued" ? "Starting" : `${STATUS_TEXT[job.status]} in ${seconds(job.elapsed_s)}`}</span>}
          </div>
          {message && <p className={`notice${message.bad ? " bad" : ""}`} role="status">{message.text}</p>}
          {job?.status === "failed" && <p className="notice bad" role="alert">The run failed. {job.error}</p>}
          {job?.status === "cancelled" && <p className="notice" role="status">The run was stopped. Nothing was saved.</p>}
        </div>
      </Section>
      <Section id="stages" first="Pipeline" second="Steps">
        <Stepper job={job} />
        {job?.status === "done" && job.result && (
          <button className="btn btn-navy" style={{ marginTop: 20 }} onClick={() => setShowResult(true)}>View results</button>
        )}
      </Section>
      {showResult && job?.result && <ResultWindow job={job} saved={results.data} onClose={closeResult} />}
    </>
  );
}
