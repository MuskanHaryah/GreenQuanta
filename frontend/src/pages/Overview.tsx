import { Link, useLocation } from "react-router-dom";
import { api, type Results } from "../api";
import { useApi, useSize } from "../hooks";
import { bestBaseline, relDiffPct } from "../selectors";
import { int, kg, signed, signedPct } from "../format";
import { Async, Check, Dots, Headline, KpiCard, Section } from "../components/ui";

export function Kpis({ r }: { r: Results }) {
  const q = r.methods.quantum;
  const best = bestBaseline(Object.entries(r.methods));
  const s1 = r.stage1;
  const queries = s1.gas_oracle_calls + s1.gas_measurements;
  return (
    <div className="grid">
      <KpiCard index={1} title="Placement valid" value={<Check ok={q.valid_real} />}
        sub={<>fits the real servers · quantized grid <Check ok={q.valid_quantized} kind="quantized" /></>} />
      <KpiCard index={2} peach title="Energy vs exact optimum" value={signedPct(q.diff_r1_pct)}
        sub={<>linear joules vs R1 (real sizes) · vs R2 (quantized optimum) {signedPct(q.diff_r2_pct)}</>} />
      <KpiCard index={3} title="vs best baseline"
        value={best && q.linear_joules != null && best[1].linear_joules != null
          ? signedPct(relDiffPct(q.linear_joules, best[1].linear_joules)) : "—"}
        sub={best ? <>
          energy (linear joules) vs <b>{best[1].label}</b>; negative = quantum uses less.
          {q.energy && best[1].energy && <> CO₂ {kg(q.energy.co2_kg)} vs {kg(best[1].energy.co2_kg)} (
            {signed(q.energy.co2_kg - best[1].energy.co2_kg, "kg")}).</>}
        </> : "no valid baseline"} />
      <KpiCard index={4} title="Stage-1 quantum queries" value={`${int(queries)} / ${int(s1.exhaustive_evaluations)}`}
        sub={<>GAS oracle calls + measurements ({s1.gas_runs} runs) vs exhaustive 2<sup>{s1.m}</sup> ·{" "}
          {queries < s1.exhaustive_evaluations ? "fewer than exhaustive" : `more than exhaustive at m = ${s1.m}`}</>} />
    </div>
  );
}

export default function Overview() {
  const [size] = useSize();
  const { search } = useLocation();
  const results = useApi(api.results, size);
  return (
    <>
      <section id="intro" className="section hero" aria-label="Green Quanta">
        <Headline as="h1" first="Green" second="Quanta" />
        <p className="lede">
          Energy- and carbon-aware VM placement with quantum algorithms (simulated): Grover Adaptive Search picks the
          servers to switch on, then constraint-preserving simulated quantum annealing places the VMs on them.
        </p>
        <div className="buttons">
          <Link className="btn btn-navy" to={{ pathname: "/compare", search }}>Explore results</Link>
          <Link className="btn btn-coral" to={{ pathname: "/pipeline", search }}>Run live</Link>
        </div>
      </section>
      <Dots />
      <Section id="kpis" first="Quantum" second="Results">
        <p className="muted">Size {size}: the quantum pipeline's cached result (one Stage-2 seed).</p>
        <Async state={results}>{(r) => <Kpis r={r} />}</Async>
      </Section>
    </>
  );
}
