import { useEffect, useState } from "react";
import { Bar, BarChart, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, type GasTrace, type TraceState } from "../api";
import { useApi, useSize } from "../hooks";
import { int, prob } from "../format";
import { Async, CircleButton, Section } from "../components/ui";

/** The pipeline's GAS seeds (RUN_CONFIG["pipe_gas_seeds"]); the API answers 400 bad_seed for any other seed. */
export const GAS_SEEDS = [0, 1, 2];
export const PLAY_MS = 900;
const COLORS = { yStar: "#F2706A", marked: "#8E9BFF", other: "#D6CFEA" };

export const stateColor = (s: TraceState, yStar: number) =>
  s.index === yStar ? COLORS.yStar : s.marked ? COLORS.marked : COLORS.other;

function QueryBar({ trace }: { trace: GasTrace }) {
  const q = trace.totals.oracle_calls + trace.totals.measurements, n = trace.totals.exhaustive;
  const max = Math.max(q, n);
  return (
    <div className="qbar" aria-label="total queries vs exhaustive">
      <span>GAS, this run: <b data-testid="total-queries">{int(q)}</b> queries ({int(trace.totals.oracle_calls)} oracle calls
        + {int(trace.totals.measurements)} measurements)</span>
      <div className="track"><i style={{ width: `${(100 * q) / max}%`, background: COLORS.yStar }} /></div>
      <span>Exhaustive: <b data-testid="exhaustive">{int(n)}</b> evaluations (2<sup>{trace.m}</sup>)</span>
      <div className="track"><i style={{ width: `${(100 * n) / max}%`, background: COLORS.other }} /></div>
    </div>
  );
}

export function Player({ trace }: { trace: GasTrace }) {
  const [i, setI] = useState(0);
  const [playing, setPlaying] = useState(false);
  const last = trace.steps.length - 1;
  const step = trace.steps[i];

  useEffect(() => {
    if (!playing) return;
    const t = setTimeout(() => {
      setI((k) => Math.min(k + 1, last));
      if (i + 1 >= last) setPlaying(false);
    }, PLAY_MS);
    return () => clearTimeout(t);
  }, [playing, i, last]);

  const go = (k: number) => { setPlaying(false); setI(Math.max(0, Math.min(last, k))); };
  return (
    <div className="player">
      <div className="player-controls">
        <CircleButton label="previous round" onClick={() => go(i - 1)} disabled={i === 0}>←</CircleButton>
        <button className="btn btn-coral" onClick={() => (i === last ? (setI(0), setPlaying(true)) : setPlaying((p) => !p))}>
          {playing ? "Pause" : "Play"}</button>
        <CircleButton label="next round" onClick={() => go(i + 1)} disabled={i === last}>→</CircleButton>
        <input type="range" min={0} max={last} value={i} aria-label="step" onChange={(e) => go(Number(e.target.value))} />
        <span>step <b data-testid="step-index">{i}</b> / {last}</span>
      </div>
      <dl className="kv">
        <div><dt>Oracle calls so far</dt><dd data-testid="oracle-calls">{int(step.oracle_calls)}</dd></div>
        <div><dt>Measurements</dt><dd>{step.step + 1}</dd></div>
        <div><dt>Threshold</dt><dd>{step.threshold == null ? "none (uniform)" : step.threshold.toFixed(6)}</dd></div>
        <div><dt>Marked states</dt><dd>{int(step.marked_count)}</dd></div>
        <div><dt>Rotations</dt><dd>{step.rotations}</dd></div>
        <div><dt>P(y*)</dt><dd data-testid="p-ystar">{prob(step.p_y_star)}</dd></div>
        <div><dt>Measured</dt><dd>state {step.measured_index}{step.improved ? " · improved" : ""}</dd></div>
      </dl>
      <div className="legend">
        <span><i className="swatch" style={{ display: "inline-block", background: COLORS.yStar }} /> y* (servers {trace.y_star_ids.join(", ")})</span>
        <span><i className="swatch" style={{ display: "inline-block", background: COLORS.marked }} /> marked (cost below threshold)</span>
        <span><i className="swatch" style={{ display: "inline-block", background: COLORS.other }} /> unmarked</span>
      </div>
      <figure style={{ margin: 0 }} aria-label="top 32 state probabilities">
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={step.top} margin={{ top: 8, right: 8, bottom: 4, left: 0 }}>
            <XAxis dataKey="index" tick={{ fontSize: 10 }} stroke="#D6CFEA" interval="preserveStartEnd" />
            <YAxis tick={{ fontSize: 10 }} stroke="#D6CFEA" width={52} domain={[0, "auto"]}
              tickFormatter={(p: number) => (p === 0 ? "0" : p < 0.01 ? p.toExponential(1) : p.toFixed(2))} />
            <Tooltip cursor={{ fill: "#2A2733" }} isAnimationActive={false}
              content={({ active, payload }) => {
                const s = active && payload?.length ? (payload[0].payload as TraceState) : null;
                return s ? <div className="tooltip">state {s.index}{s.index === trace.y_star_index ? " (y*)" : ""}<br />
                  servers {s.ids.join(", ") || "none"}<br />p = {s.p.toPrecision(6)}<br />cost {s.cost.toFixed(6)}
                  {s.marked ? " · marked" : ""}</div> : null;
              }} />
            <Bar dataKey="p" isAnimationActive={false}>
              {step.top.map((s) => <Cell key={s.index} fill={stateColor(s, trace.y_star_index)} data-testid={`bar-${s.index}`} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
        <figcaption className="muted" style={{ fontSize: 12 }}>
          The 32 most likely server sets (of {int(trace.states)}) before this step's measurement.
        </figcaption>
      </figure>
      <QueryBar trace={trace} />
    </div>
  );
}

export default function Quantum() {
  const [size] = useSize();
  const [seed, setSeed] = useState(GAS_SEEDS[0]);
  const trace = useApi(api.gasTrace, size, seed);
  const claims = useApi(api.claims, size);
  return (
    <>
      <Section id="grover" first="Grover" second="Player">
        <div className="row" style={{ marginBottom: 16, alignItems: "flex-end" }}>
          <label className="field">GAS seed
            <select value={seed} onChange={(e) => setSeed(Number(e.target.value))}>
              {GAS_SEEDS.map((s) => <option key={s} value={s}>{s}</option>)}
            </select>
          </label>
          <p className="muted" style={{ margin: 0, maxWidth: "60ch" }}>One Grover Adaptive Search run of the pipeline
            (Stage 1, size {size}), replayed step by step: each step rotates the state towards the server sets cheaper
            than the best seen, then measures.</p>
        </div>
        <Async state={trace}>{(t) => <Player key={`${t.size}-${t.seed}`} trace={t} />}</Async>
      </Section>
      <Section id="simulated" first="What is" second="Simulated">
        <Async state={claims}>{(c) => (
          <div className="box">
            <h3>What is quantum here, and what is simulated</h3>
            <p>{c.methods.find((m) => m.method_id === "quantum")?.algorithm}</p>
            <ul>{c.notes.map((n) => <li key={n}>{n}</li>)}</ul>
          </div>
        )}</Async>
      </Section>
    </>
  );
}
