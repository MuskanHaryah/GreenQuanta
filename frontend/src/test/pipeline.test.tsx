import { act, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { Job, RunEvent } from "../api";
import Pipeline, { POLL_MS } from "../pages/Pipeline";
import { json, mockApi, renderPage } from "./utils";

const ev = (kind: RunEvent["kind"], t: number, data: Record<string, unknown> = {}): RunEvent => ({ kind, t, data });
const OUT = {
  chosen_ids: [2, 3], real_valid: true, quantized_valid: true, linear_joules: 39951971.03582648, diff_r1_pct: 0,
  diff_r2_pct: 0, gap_to_h_min_pct: 0, curve_it_energy_kwh: 11.039, gas_oracle_calls: 198, gas_measurements: 153,
};
const job = (status: Job["status"], events: RunEvent[], extra: Partial<Job> = {}): Job => ({
  job_id: "j1", size: "small_v2", stage2: "csqa", status, started_at: "2026-11-14T00:00:00+00:00",
  elapsed_s: events.at(-1)?.t ?? 0, events, result: null, matches_cached: null, error: null, ...extra,
});
const E = [ev("PROPOSALS", 2), ev("STAGE2", 20), ev("HYBRID", 22), ev("LOGGED", 23), ev("RESULT", 25)];
const SEQUENCE: Job[] = [
  job("running", []),
  job("running", E.slice(0, 1)),
  job("running", E.slice(0, 2)),
  job("running", E.slice(0, 3)),
  job("done", E, { result: { out: OUT, assignment: {}, energy: {} }, matches_cached: true }),
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
    await screen.findByText(/running ·/);
    expect(screen.getByRole("button", { name: "Running…" })).toBeDisabled();
    expect(states()).toEqual(["active", "idle", "idle", "idle"]);
    await tick();
    expect(states()).toEqual(["done", "active", "idle", "idle"]);
    await tick();
    expect(states()).toEqual(["done", "active", "active", "idle"]);
    expect(screen.getByTestId("step-2")).toHaveTextContent("1 solve");
    await tick();
    expect(states()).toEqual(["done", "done", "done", "idle"]);
    await tick();
    expect(states()).toEqual(["done", "done", "done", "done"]);
    expect(screen.getByTestId("matches")).toHaveTextContent("matches cached result ✓");
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
    expect(await screen.findByText("A run is already in progress; showing that run.")).toBeInTheDocument();
    expect(runCalls(fetchMock)).toBeGreaterThan(0);
    expect(screen.getByText(/run j1 · small_v2/)).toBeInTheDocument();
    expect(states()[0]).toBe("done");
  });

  it("shows a failed job's error and stops polling", async () => {
    const failed = job("failed", E.slice(0, 1), { error: "timed out after 900 s; process tree killed" });
    const fetchMock = serve([job("running", []), failed]);
    renderPage(<Pipeline />, "/pipeline?size=small_v2");
    await user().click(screen.getByRole("button", { name: "Run live" }));
    await screen.findByText(/running ·/);
    await tick();
    expect(screen.getByRole("alert")).toHaveTextContent("Run failed: timed out after 900 s; process tree killed");
    expect(screen.getByRole("button", { name: "Run live" })).toBeEnabled();
    const calls = runCalls(fetchMock);
    await tick(); await tick();
    expect(runCalls(fetchMock)).toBe(calls);
  });
});
