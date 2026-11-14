import { useEffect, useState } from "react";
import { api, ApiError, type Job, type RunEvent, type Stage2 } from "../api";
import { useSize } from "../hooks";
import { joules, seconds, signedPct, pct, two } from "../format";
import { Check, Section } from "../components/ui";

export const POLL_MS = 1000;
const TERMINAL = new Set(["done", "failed"]);

const STEPS = [
  { title: "Stage 1 · Grover Adaptive Search", text: "Chooses which servers to switch on. A simulated Grover search "
    + "repeatedly marks the server sets cheaper than the best seen so far and amplifies them." },
  { title: "Stage 2 · constraint-preserving SQA", text: "Places every VM on the chosen servers. Simulated quantum "
    + "annealing (path-integral Monte Carlo) whose moves always keep each VM on exactly one server." },
  { title: "Shared validator", text: "The same checker for every method: each VM placed once, no server over its "
    + "CPU or memory limit (real sizes, and the quantum model's rounding grid)." },
  { title: "CloudSim Plus replay", text: "The placement is replayed in the CloudSim Plus simulator to cross-check "
    + "the energy. Done when the results are built; a live run is compared with that cached result." },
];

type StepState = "idle" | "active" | "done";

/** Stepper state from the run's events: PROPOSALS → Stage 1 done; STAGE2 → Stage 2 progress; HYBRID → validated;
 *  RESULT → done (the CloudSim replay is the cached one). */
export function stepStates(job: Job | null): { states: StepState[]; stage2Solves: number } {
  if (!job) return { states: ["idle", "idle", "idle", "idle"], stage2Solves: 0 };
  const has = (k: RunEvent["kind"]) => job.events.some((e) => e.kind === k);
  const stage2Solves = job.events.filter((e) => e.kind === "STAGE2").length;
  const running = !TERMINAL.has(job.status);
  const s1: StepState = has("PROPOSALS") || has("SKIP") ? "done" : running ? "active" : "idle";
  const s3: StepState = has("HYBRID") ? "done" : s1 === "done" && stage2Solves > 0 && running ? "active" : "idle";
  const s2: StepState = has("HYBRID") ? "done" : s1 === "done" && running ? "active" : stage2Solves > 0 ? "done" : "idle";
  const s4: StepState = has("RESULT") && job.status === "done" ? "done" : "idle";
  return { states: [s1, s2, s3, s4], stage2Solves };
}

const STATE_TEXT: Record<StepState, string> = { idle: "waiting", active: "running…", done: "done" };

function Stepper({ job }: { job: Job | null }) {
  const { states, stage2Solves } = stepStates(job);
  return (
    <ol className="stepper" aria-label="pipeline stages">
      {STEPS.map((s, i) => (
        <li key={s.title} className="step" data-state={states[i]} data-testid={`step-${i + 1}`}>
          <div className="numeral" aria-hidden="true">{two(i + 1)}</div>
          <h3>{s.title}</h3>
          <p>{s.text}</p>
          {job && (
            <span className="state">
              {i === 3 && states[i] === "done" ? "cached replay" : STATE_TEXT[states[i]]}
              {i === 1 && stage2Solves > 0 && ` · ${stage2Solves} solve${stage2Solves > 1 ? "s" : ""}`}
            </span>
          )}
        </li>
      ))}
    </ol>
  );
}

function Outcome({ job }: { job: Job }) {
  const out = job.result?.out;
  if (!out) return null;
  return (
    <>
      <dl className="kv">
        <div><dt>Servers on</dt><dd>{out.chosen_ids.join(", ")}</dd></div>
        <div><dt>Valid (real)</dt><dd><Check ok={out.real_valid} /></dd></div>
        <div><dt>Linear joules</dt><dd>{joules(out.linear_joules)}</dd></div>
        <div><dt>vs exact R1</dt><dd>{signedPct(out.diff_r1_pct)}</dd></div>
        <div><dt>Gap to H_min</dt><dd>{pct(out.gap_to_h_min_pct)}</dd></div>
        <div><dt>GAS queries</dt><dd>{out.gas_oracle_calls} + {out.gas_measurements}</dd></div>
      </dl>
      {job.matches_cached === true && <p className="notice" data-testid="matches">matches cached result <span className="ok-text">✓</span></p>}
      {job.matches_cached === false && <p className="notice bad" data-testid="matches">does NOT match the cached result ✗</p>}
    </>
  );
}

export default function Pipeline() {
  const [size] = useSize();
  const [stage2, setStage2] = useState<Stage2>("csqa");
  const [jobId, setJobId] = useState<string | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [starting, setStarting] = useState(false);
  const [message, setMessage] = useState<{ text: string; bad?: boolean } | null>(null);

  useEffect(() => {
    if (!jobId) return;
    let live = true, timer: ReturnType<typeof setTimeout> | undefined;
    const poll = () => api.run(jobId).then((j) => {
      if (!live) return;
      setJob(j);
      if (!TERMINAL.has(j.status)) timer = setTimeout(poll, POLL_MS);
    }, (e: ApiError) => live && setMessage({ text: `lost the run: ${e.message}`, bad: true }));
    poll();
    return () => { live = false; clearTimeout(timer); };
  }, [jobId]);

  const running = starting || (job !== null && !TERMINAL.has(job.status));
  const start = async () => {
    setStarting(true);
    setMessage(null);
    setJob(null);
    try {
      setJobId((await api.startRun(size, stage2)).job_id);
    } catch (e) {
      const err = e as ApiError;
      if (err.status === 409 && err.jobId) {
        setMessage({ text: "A run is already in progress; showing that run." });
        setJobId(err.jobId);
      } else {
        setMessage({ text: `Could not start the run: ${err.message}`, bad: true });
      }
    } finally {
      setStarting(false);
    }
  };

  return (
    <>
      <Section id="stages" first="Quantum" second="Pipeline">
        <Stepper job={job} />
      </Section>
      <Section id="live" first="Live" second="Run">
        <div className="live">
          <p style={{ margin: 0 }}>
            Re-runs the whole pipeline now on the server, for size <b>{size}</b>, with the same seeds as the cached
            result, so it must reproduce it. Small takes about 20–30 s; large took 76.8 s when measured (medium and
            large: a minute or more).
          </p>
          <div className="live-controls">
            <label className="field">Stage 2
              <select value={stage2} onChange={(e) => setStage2(e.target.value as Stage2)} disabled={running}>
                <option value="csqa">Quantum SQA</option>
                <option value="csa">Classical SA control</option>
              </select>
            </label>
            <button className="btn btn-coral" onClick={start} disabled={running}>{running ? "Running…" : "Run live"}</button>
            {job && <span aria-live="polite">{job.status} · {seconds(job.elapsed_s)}</span>}
          </div>
          {message && <p className={`notice${message.bad ? " bad" : ""}`} role="status">{message.text}</p>}
          {job?.status === "failed" && <p className="notice bad" role="alert">Run failed: {job.error}</p>}
          {job && <p className="muted" style={{ margin: 0, fontSize: 12 }}>
            run {job.job_id} · {job.size} · {job.stage2} · events: {job.events.map((e) => e.kind).join(" → ") || "none yet"}</p>}
          {job?.status === "done" && <Outcome job={job} />}
        </div>
      </Section>
    </>
  );
}
