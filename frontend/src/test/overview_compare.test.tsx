import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { CompareRow, Results } from "../api";
import { bestBaseline, relDiffPct } from "../selectors";
import { signedPct } from "../format";
import Overview, { Kpis } from "../pages/Overview";
import Compare from "../pages/Compare";
import { fixtures, mockApi, renderPage } from "./utils";

const results = fixtures.results as unknown as Results;
const rows = fixtures.compare as unknown as CompareRow[];
const clone = <T,>(x: T): T => structuredClone(x);

describe("best baseline", () => {
  it("is the valid_real baseline with the lowest diff_r1_pct, never ranked by kWh", () => {
    const base = { kind: "baseline" as const };
    const picked = bestBaseline([
      ["invalid_cheapest", { ...base, valid_real: false, diff_r1_pct: 1, it_energy_kwh: 1 }],
      ["low_kwh", { ...base, valid_real: true, diff_r1_pct: 20, it_energy_kwh: 2 }],
      ["best", { ...base, valid_real: true, diff_r1_pct: 10, it_energy_kwh: 9 }],
      ["quantum", { kind: "quantum" as const, valid_real: true, diff_r1_pct: 0, it_energy_kwh: 0 }],
    ]);
    expect(picked?.[0]).toBe("best");
  });

  it("on the small_v2 fixture equals the first valid baseline of the API's compare order", () => {
    const expected = rows.find((r) => r.kind === "baseline" && r.valid_real)!;
    expect(bestBaseline(Object.entries(results.methods))?.[0]).toBe(expected.method_id);
  });
});

describe("Overview KPIs", () => {
  it("renders 01-04 with validity and the signed difference to the best baseline", () => {
    render(<Kpis r={results} />);
    const best = bestBaseline(Object.entries(results.methods))![1];
    const q = results.methods.quantum;
    const diff = relDiffPct(q.linear_joules!, best.linear_joules!);
    expect(diff).toBeLessThan(0);   // small_v2: the quantum placement is the optimum, every baseline costs more
    const card = screen.getByRole("article", { name: "vs best baseline" });
    expect(card).toHaveTextContent(signedPct(diff));
    expect(card).toHaveTextContent(/^.*−/);
    expect(card).toHaveTextContent(best.label);
    expect(screen.getByRole("article", { name: "Placement valid" })).toHaveTextContent("✓");
    const q4 = screen.getByRole("article", { name: "Stage-1 quantum queries" });
    expect(q4).toHaveTextContent(String(results.stage1.exhaustive_evaluations));
    expect(q4).toHaveTextContent(/more than exhaustive/);   // m = 8: GAS costs more than exhaustive (honest)
  });

  it("shows a + sign when the quantum result is worse than the best baseline", () => {
    const r = clone(results);
    const best = bestBaseline(Object.entries(r.methods))![1];
    r.methods.quantum.linear_joules = best.linear_joules! * 1.05;
    render(<Kpis r={r} />);
    expect(screen.getByRole("article", { name: "vs best baseline" })).toHaveTextContent("+5.00 %");
  });

  it("loads the selected size's results through the API", async () => {
    const fetchMock = mockApi();
    renderPage(<Overview />, "/?size=small_v2");
    expect(await screen.findByRole("article", { name: "Placement valid" })).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledWith("/api/results/small_v2", undefined);
    expect(screen.getByRole("link", { name: "Run live" })).toHaveAttribute("href", "/pipeline?size=small_v2");
  });
});

describe("Compare", () => {
  it("keeps the API order, highlights the quantum row and shows every method", async () => {
    mockApi();
    renderPage(<Compare />, "/compare?size=small_v2");
    const table = await screen.findByRole("table");
    const ids = within(table).getAllByRole("row").slice(1).map((tr) => tr.getAttribute("data-method"));
    expect(ids).toEqual(rows.map((r) => r.method_id));
    const quantum = table.querySelector('tr[data-method="quantum"]')!;
    expect(quantum).toHaveClass("highlight");
    expect(table.querySelectorAll("tr.highlight")).toHaveLength(1);
  });

  it("renders invalid methods with ✗ (red for valid_real, muted with a tooltip for valid_quantized)", async () => {
    const changed = clone(rows);
    changed[changed.length - 1].valid_real = false;
    changed[changed.length - 1].valid_quantized = false;
    mockApi({ "/api/results/small_v2/compare": changed });
    renderPage(<Compare />, "/compare?size=small_v2");
    const table = await screen.findByRole("table");
    const last = table.querySelector(`tr[data-method="${changed[changed.length - 1].method_id}"]`)!;
    expect(within(last as HTMLElement).getByLabelText("invalid")).toHaveClass("bad");
    const soft = within(last as HTMLElement).getByLabelText(/not on grid/);
    expect(soft).toHaveClass("soft");
    expect(soft).toHaveAttribute("title", "fits the real servers; does not fit the quantum model's rounding grid");
  });

  it("filters rows with the method chips", async () => {
    mockApi();
    renderPage(<Compare />, "/compare?size=small_v2");
    const table = await screen.findByRole("table");
    await userEvent.click(screen.getByRole("button", { name: "Baselines" }));
    const ids = within(table).getAllByRole("row").slice(1).map((tr) => tr.getAttribute("data-method"));
    expect(ids).toEqual(rows.filter((r) => r.kind !== "baseline").map((r) => r.method_id));
  });
});
