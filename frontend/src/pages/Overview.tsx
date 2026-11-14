import { Link, useLocation } from "react-router-dom";
import { api, type Results } from "../api";
import { useApi, useSize } from "../hooks";
import { bestBaseline, relDiffPct } from "../selectors";
import { methodName } from "../methods";
import { int, kg, signedPct, sizeName } from "../format";
import { Async, Dots, Headline, KpiCard, Section } from "../components/ui";

export function Kpis({ r }: { r: Results }) {
  const q = r.methods.quantum;
  const best = bestBaseline(Object.entries(r.methods));
  const vsBest = best && q.linear_joules != null && best[1].linear_joules != null
    ? relDiffPct(q.linear_joules, best[1].linear_joules) : null;
  const s1 = r.stage1;
  const queries = s1.gas_oracle_calls + s1.gas_measurements;
  return (
    <div className="grid">
      <KpiCard index={1} title="Valid placement" value={q.valid_real ? "Yes" : "No"} tone={q.valid_real ? "ok" : "bad"}
        sub={q.valid_real ? "Every VM fits on its server." : "Some VMs do not fit their server."} />
      <KpiCard index={2} tone="peach" title="Above the exact optimum" value={signedPct(q.diff_r1_pct)}
        sub="Extra energy compared with the best possible placement." />
      <KpiCard index={3} title="Against the best baseline" value={signedPct(vsBest)}
        tone={vsBest == null ? undefined : vsBest <= 0 ? "ok" : "bad"}
        sub={best ? <>
          {vsBest != null && vsBest <= 0 ? "Less" : "More"} energy than {methodName(best[0])}.
          {q.energy && best[1].energy && <> CO₂ {kg(q.energy.co2_kg)} against {kg(best[1].energy.co2_kg)}.</>}
        </> : "No valid baseline."} />
      <KpiCard index={4} tone="periwinkle" title="Quantum search queries" value={`${int(queries)} / ${int(s1.exhaustive_evaluations)}`}
        sub={queries < s1.exhaustive_evaluations
          ? `Grover search needed ${int(queries)} checks instead of trying all ${int(s1.exhaustive_evaluations)} server combinations.`
          : `Grover search used ${int(queries)} checks. At this size trying all ${int(s1.exhaustive_evaluations)} combinations is cheaper.`} />
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
          Energy and carbon aware VM placement with quantum algorithms, simulated on a classical computer. Grover
          search chooses which servers to switch on, then quantum annealing places the VMs on them.
        </p>
        <div className="buttons">
          <Link className="btn btn-navy" to={{ pathname: "/compare", search }}>Explore results</Link>
          <Link className="btn btn-coral" to={{ pathname: "/pipeline", search }}>Run live</Link>
        </div>
      </section>
      <Dots />
      <Section id="kpis" first="Quantum" second="Results">
        <p className="muted">Size {sizeName(size)}</p>
        <Async state={results}>{(r) => <Kpis r={r} />}</Async>
      </Section>
    </>
  );
}
