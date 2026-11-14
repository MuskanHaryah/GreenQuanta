import { useState } from "react";
import { api, type CompareRow, type Results } from "../api";
import { combine, useApi, useSize } from "../hooks";
import { KIND_GROUPS } from "../selectors";
import { joules, kg, kwh, pct, signedPct } from "../format";
import { Async, Check, MethodBadge, Section } from "../components/ui";
import BarChartCard, { type BarRow } from "../components/BarChartCard";

const bars = (rows: CompareRow[], key: "it_energy_kwh" | "co2_kg" | "diff_r1_pct"): BarRow[] =>
  rows.filter((r) => r[key] != null).map((r) => ({ id: r.method_id, label: r.label, value: r[key] as number }));

function Charts({ rows }: { rows: CompareRow[] }) {
  return (
    <div className="charts">
      <BarChartCard title="Energy vs exact optimum R1" rows={bars(rows, "diff_r1_pct")} format={signedPct}
        caption="Linear joules, the optimised objective, in % above R1 (the ranking used everywhere)." />
      <BarChartCard title="IT energy" rows={bars(rows, "it_energy_kwh").sort((a, b) => a.value - b.value)} format={kwh}
        caption="Sorted by kWh. kWh comes from the SPEC power curve; the ranking uses linear joules (left)." />
      <BarChartCard title="CO₂" rows={bars(rows, "co2_kg")} format={kg}
        caption="From the SPEC-curve energy; same order as the table." />
    </div>
  );
}

function Table({ rows, results }: { rows: CompareRow[]; results: Results }) {
  return (
    <div className="table-wrap" tabIndex={0} aria-label="comparison table, scrolls sideways">
      <table>
        <thead>
          <tr>
            <th>Method</th><th>Kind</th><th>Valid real</th><th>Valid quantized</th><th className="num">Active servers</th>
            <th className="num">Linear joules</th><th className="num">vs R1</th><th className="num">vs R2</th>
            <th className="num">Gap to H_min</th><th>CloudSim check</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => {
            const cs = results.methods[r.method_id]?.cloudsim;
            return (
              <tr key={r.method_id} className={r.kind === "quantum" ? "highlight" : undefined} data-method={r.method_id}>
                <td><MethodBadge id={r.method_id} label={r.label} /></td>
                <td className="kindtag">{r.kind}</td>
                <td><Check ok={r.valid_real} /></td>
                <td><Check ok={r.valid_quantized} kind="quantized" /></td>
                <td className="num">{r.active_servers}</td>
                <td className="num">{joules(results.methods[r.method_id]?.linear_joules)}</td>
                <td className="num">{signedPct(r.diff_r1_pct)}</td>
                <td className="num">{signedPct(r.diff_r2_pct)}</td>
                <td className="num">{r.gap_to_h_min_pct == null ? "n/a" : pct(r.gap_to_h_min_pct)}</td>
                <td>{r.cloudsim_ok == null ? "—" : <><Check ok={r.cloudsim_ok} />{" "}
                  {cs && <span className="muted">{signedPct(cs.diff_it_pct_vs_python)}</span>}</>}</td>
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
  const [shown, setShown] = useState(() => new Set(KIND_GROUPS.map((g) => g.id)));
  const kinds = new Set(KIND_GROUPS.filter((g) => shown.has(g.id)).flatMap((g) => g.kinds));
  const toggle = (id: string) => setShown((s) => {
    const next = new Set(s);
    if (!next.delete(id)) next.add(id);
    return next;
  });
  return (
    <>
      <Section id="charts" first="Energy" second="Charts">
        <div className="chips" role="group" aria-label="method filter">
          {KIND_GROUPS.map((g) => (
            <button key={g.id} className="pill" aria-pressed={shown.has(g.id)} onClick={() => toggle(g.id)}>{g.label}</button>
          ))}
        </div>
        <Async state={data}>{([rows]) => <Charts rows={rows.filter((r) => kinds.has(r.kind))} />}</Async>
      </Section>
      <Section id="table" first="Method" second="Table">
        <p className="muted">Order: linear joules (diff to R1), as the API returns it. Invalid methods stay in the table.</p>
        <Async state={data}>{([rows, results]) => <Table rows={rows.filter((r) => kinds.has(r.kind))} results={results} />}</Async>
        <p className="muted" style={{ fontSize: 12 }}>
          A muted ✗ under “valid quantized” means the placement fits the real servers but not the quantum model's rounding
          grid; only a red ✗ (valid real) is invalid. Gap to H_min exists only where the quantized model was solved.
          CloudSim check: the placement replayed in CloudSim Plus, IT-energy difference vs the Python model.
        </p>
      </Section>
    </>
  );
}
