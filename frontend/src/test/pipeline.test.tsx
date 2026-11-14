import { act, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { Job, RunEvent } from "../api";
import Pipeline, { POLL_MS } from "../pages/Pipeline";
import { fixtures, json, mockApi, renderPage } from "./utils";

const ev = (kind: RunEvent["kind"], t: number, data: Record<string, unknown> = {}): RunEvent => ({ kind, t, data });
const OUT = {
  chosen_ids: [2, 3], real_valid: true, quantized_valid: true, linear_joules: 39951971.03582648, diff_r1_pct: 0,
  diff_r2_pct: 0, gap_to_h_min_pct: 0, curve_it_energy_kwh: 11.039, gas_oracle_calls: 198, gas_measurements: 153,
  gas_runs: 3, stage2_s: 17.25,
};
const ENERGY = { active_hosts: 2, it_energy_kwh: 11.039100641817516, facility_energy_kwh: 17.000214988398977, co2_kg: 7.395093519953555 };
const job = (status: Job["status"], events: RunEvent[], extra: Partial<Job> = {}): Job => ({
  job_id: "j1", size: "small_v2", stage2: "csqa", status, started_at: "2026-11-14T00:00:00+00:00",
  elapsed_s: events.at(-1)?.t ?? 0, events, result: null, matches_cached: null, error: null, ...extra,
});
const E = [
  ev("PROPOSALS", 2, { gas_runs: 3, gas_oracle_calls: 198, gas_measurements: 153 }),
  ev("STAGE2", 20, { active_ids: [2, 3], time_s: 17.25 }),
  ev("HYBRID", 22, { real_valid: true, diff_r1_pct: 0 }),
  ev("LOGGED", 23), ev("RESULT", 25),
];
const SEQUENCE: Job[] = [
  job("running", []),
  job("running", E.slice(0, 1)),
  job("running", E.slice(0, 2)),
  job("running", E.slice(0, 3)),
  job("done", E, { result: { out: OUT, assignment: {}, energy: ENERGY }, matches_cached: true }),
];

const states = () => [1, 2, 3, 4].map((i) => screen.getByTestId(`step-${i}`).getAttribute("data-state"));
const tick = () => act(() => vi.advanceTimersByTimeAsync(POLL_MS));

function serve(sequence: Job[], post = () => json({ job_id: "j1" }, 202)) {
  let n = 0;
  return mockApi({ "/api/runs": post, "/api/runs/j1": () => json(sequence[Math.min(n++, sequence.length - 1)]) });
}
const runCalls = (m: ReturnType<typeof mockApi>) => m.mock.calls.filter((c) => String(c[0]) === "/api/runs/j1").length;

describe("Pipeline live run", () => {
  beforeEach(() => { vi.useFakeTimers({ shouldAdvanceTime: true }); });
  afterEach(() => { vi.useRealTimers(); });
  const user = () => userEvent.setup({ advanceTimers: vi.advanceTimersByTime });

  it("advances the stepper with the event sequence, disables the button while running, stops polling when done", async () => {
    const fetchMock = serve(SEQUENCE);
    renderPage(<Pipeline />, "/pipeline?size=small_v2");
    const button = screen.getByRole("button", { name: "Run live" });
    await user().click(button);
    expect(fetchMock).toHaveBeenCalledWith("/api/runs", expect.objectContaining({
      method: "POST", body: JSON.stringify({ size: "small_v2", stage2: "csqa" }) }));
    await screen.findByText(/Running in/);
    expect(screen.getByRole("button", { name: "Running" })).toBeDisabled();
    expect(screen.queryByRole("combobox")).toBeNull();   // one run option only: the quantum pipeline
    expect(states()).toEqual(["active", "idle", "idle", "idle"]);
    await tick();
    expect(states()).toEqual(["done", "active", "idle", "idle"]);
    await tick();
    expect(states()).toEqual(["done", "active", "active", "idle"]);
    await tick();
    expect(states()).toEqual(["done", "done", "done", "idle"]);
    await tick();
    expect(states()).toEqual(["done", "done", "done", "done"]);
    expect(screen.getByTestId("detail-1")).toHaveTextContent("3 search runs, 198 oracle calls, 153 measurements.");
    expect(screen.getByTestId("detail-2")).toHaveTextContent("Servers 2, 3 switched on. Placed in 17.3 s.");
    expect(screen.getByTestId("detail-3")).toHaveTextContent("Every VM fits. 0.00 % above the exact optimum.");
    expect(screen.getByTestId("detail-4")).toHaveTextContent("Same as the saved result.");
    expect(document.body).not.toHaveTextContent(/[✓✗]/);
    expect(screen.queryByRole("button", { name: "stop the run" })).toBeNull();   // nothing to stop any more
    expect(screen.getByRole("button", { name: "Run live" })).toBeEnabled();
    const calls = runCalls(fetchMock);
    expect(calls).toBe(SEQUENCE.length);
    await tick(); await tick(); await tick();
    expect(runCalls(fetchMock)).toBe(calls);   // polling stopped
  });

  it("attaches to the running job on 409", async () => {
    const fetchMock = serve(SEQUENCE.slice(1, 2), () =>
      json({ error: { code: "run_in_progress", message: "run j1 is still running" }, job_id: "j1" }, 409));
    renderPage(<Pipeline />, "/pipeline?size=small_v2");
    await user().click(screen.getByRole("button", { name: "Run live" }));
    expect(await screen.findByText("A run is already going, showing that one.")).toBeInTheDocument();
    expect(runCalls(fetchMock)).toBeGreaterThan(0);
    expect(states()[0]).toBe("done");
  });

  it("shows a failed job's error and stops polling", async () => {
    const failed = job("failed", E.slice(0, 1), { error: "timed out after 900 s; process tree killed" });
    const fetchMock = serve([job("running", []), failed]);
    renderPage(<Pipeline />, "/pipeline?size=small_v2");
    await user().click(screen.getByRole("button", { name: "Run live" }));
    await screen.findByText(/Running in/);
    await tick();
    expect(screen.getByRole("alert")).toHaveTextContent("The run failed. timed out after 900 s; process tree killed");
    expect(screen.getByRole("button", { name: "Run live" })).toBeEnabled();
    const calls = runCalls(fetchMock);
    await tick(); await tick();
    expect(runCalls(fetchMock)).toBe(calls);
  });

  it("shows the run's input data: VMs, servers by type with limits and power", async () => {
    serve(SEQUENCE);
    renderPage(<Pipeline />, "/pipeline?size=small_v2");
    const input = await screen.findByTestId("input-data");
    const p = fixtures.results.problem;
    expect(input).toHaveTextContent(`Virtual machines${p.vms}`);
    expect(input).toHaveTextContent(`${p.servers.length} of ${new Set(p.servers.map((x) => x.type)).size} types`);
    for (const t of new Set(p.servers.map((x) => x.type))) expect(input).toHaveTextContent(`Type ${t}`);
    expect(input).toHaveTextContent(`${p.servers[0].idle_w.toFixed(0)} W`);
  });

  it("stops a running job with the Stop button and shows it was stopped", async () => {
    let cancelled = false;
    const running = job("running", E.slice(0, 1));
    const stopped = job("cancelled", E.slice(0, 1), { error: "stopped by the user" });
    const fetchMock = mockApi({
      "/api/runs": () => json({ job_id: "j1" }, 202),
      "/api/runs/j1": () => json(cancelled ? stopped : running),
      "/api/runs/j1/cancel": () => { cancelled = true; return json({ job_id: "j1", status: "cancelled" }); },
    });
    renderPage(<Pipeline />, "/pipeline?size=small_v2");
    await user().click(screen.getByRole("button", { name: "Run live" }));
    await screen.findByText(/Running in/);
    await user().click(screen.getByRole("button", { name: "stop the run" }));
    expect(fetchMock).toHaveBeenCalledWith("/api/runs/j1/cancel", expect.objectContaining({ method: "POST" }));
    await tick();
    expect(screen.getByText("The run was stopped. Nothing was saved.")).toBeInTheDocument();
    expect(screen.getByText(/Stopped in/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Run live" })).toBeEnabled();
    const calls = runCalls(fetchMock);
    await tick(); await tick();
    expect(runCalls(fetchMock)).toBe(calls);   // polling stopped
  });

  it("opens a result window after a successful run, over a dimmed page, and closes it", async () => {
    serve(SEQUENCE);
    renderPage(<Pipeline />, "/pipeline?size=small_v2");
    expect(screen.queryByRole("button", { name: "View results" })).toBeNull();     // nothing to show yet
    await user().click(screen.getByRole("button", { name: "Run live" }));
    for (let i = 0; i < SEQUENCE.length; i++) await tick();
    await user().click(screen.getAllByRole("button", { name: "View results" })[0]);
    const dialog = screen.getByRole("dialog", { name: "Run result" });
    expect(dialog).toHaveAttribute("aria-modal", "true");
    expect(dialog.closest(".modal-layer")!.parentElement).toBe(document.body);     // above the whole page
    expect(dialog).toHaveTextContent("Size Small");
    expect(dialog).toHaveTextContent("0.00 %");
    expect(screen.getByTestId("row-valid")).toHaveTextContent("Yes");
    expect(screen.getByTestId("row-matches")).toHaveTextContent("Yes");
    expect(dialog).toHaveTextContent("39,951,971 J");
    expect(dialog).toHaveTextContent("11.039 kWh");
    expect(dialog).toHaveTextContent("7.395 kg");
    expect(dialog).toHaveTextContent("198 and 153");
    expect(dialog).not.toHaveTextContent(/[✓✗]/);
    await user().click(screen.getByTestId("modal-backdrop"));
    expect(screen.queryByRole("dialog", { name: "Run result" })).toBeNull();
    await user().click(screen.getAllByRole("button", { name: "View results" })[0]);
    await user().keyboard("{Escape}");
    expect(screen.queryByRole("dialog", { name: "Run result" })).toBeNull();
  });
});
