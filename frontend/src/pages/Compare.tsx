import { useState } from "react";
import { api, type CompareRow, type Results } from "../api";
import { combine, useApi, useSize } from "../hooks";
import { KIND_GROUPS } from "../selectors";
import { methodName, shown } from "../methods";
import { joules, kg, kwh, signedPct } from "../format";
import { Async, MethodBadge, Section } from "../components/ui";
import BarChartCard, { type BarRow } from "../components/BarChartCard";

const bars = (rows: CompareRow[], key: "it_energy_kwh" | "co2_kg" | "diff_r1_pct"): BarRow[] =>
  rows.filter((r) => r[key] != null).map((r) => ({ id: r.method_id, label: methodName(r.method_id), value: r[key] as number }));

function Charts({ rows }: { rows: CompareRow[] }) {
  return (
    <div className="charts">
      <BarChartCard title="Above the exact optimum" rows={bars(rows, "diff_r1_pct")} format={signedPct}
        caption="Extra energy in percent. Methods are ranked by this." />
      <BarChartCard title="Server energy" rows={bars(rows, "it_energy_kwh").sort((a, b) => a.value - b.value)} format={kwh}
        caption="Energy used by the servers over one day." />
      <BarChartCard title="CO₂" rows={bars(rows, "co2_kg")} format={kg} caption="Carbon emitted over one day." />
    </div>
  );
}

function Table({ rows, results }: { rows: CompareRow[]; results: Results }) {
  return (
    <div className="table-wrap" tabIndex={0} aria-label="comparison table">
      <table>
        <thead>
          <tr>
            <th>Method</th><th className="num">Servers on</th><th className="num">Energy</th>
            <th className="num">Above optimum</th><th className="num">CloudSim difference</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => {
            const m = results.methods[r.method_id];
            return (
              <tr key={r.method_id} className={r.kind === "quantum" ? "highlight" : undefined} data-method={r.method_id}>
                <td>
                  <MethodBadge id={r.method_id} label={r.label} />
                  {!r.valid_real && <span className="bad-text"> does not fit</span>}
                </td>
                <td className="num">{r.active_servers}</td>
                <td className="num">{joules(m?.linear_joules)}</td>
                <td className="num">{signedPct(r.diff_r1_pct)}</td>
                <td className="num">{m?.cloudsim ? <span className={r.cloudsim_ok ? undefined : "bad-text"}>
                  {signedPct(m.cloudsim.diff_it_pct_vs_python)}{r.cloudsim_ok ? "" : " mismatch"}</span> : "—"}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export default function Compare() {
  const [size] = useSize();
  const data = combine(useApi(api.compare, size), useApi(api.results, size));
  const [picked, setPicked] = useState(() => new Set(KIND_GROUPS.map((g) => g.id)));
  const kinds = new Set(KIND_GROUPS.filter((g) => picked.has(g.id)).flatMap((g) => g.kinds));
  const visible = (rows: CompareRow[]) => rows.filter((r) => shown(r.method_id) && kinds.has(r.kind));
  const toggle = (id: string) => setPicked((s) => {
    const next = new Set(s);
    if (!next.delete(id)) next.add(id);
    return next;
  });
  return (
    <>
      <Section id="charts" first="Energy" second="Charts">
        <div className="chips" role="group" aria-label="method filter">
          {KIND_GROUPS.map((g) => (
            <button key={g.id} className="pill" aria-pressed={picked.has(g.id)} onClick={() => toggle(g.id)}>{g.label}</button>
          ))}
        </div>
        <Async state={data}>{([rows]) => <Charts rows={visible(rows)} />}</Async>
      </Section>
      <Section id="table" first="Method" second="Table">
        <p className="muted">Ordered from the lowest energy to the highest.</p>
        <Async state={data}>{([rows, results]) => <Table rows={visible(rows)} results={results} />}</Async>
      </Section>
    </>
  );
}
