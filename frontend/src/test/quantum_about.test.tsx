import { act, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { GasTrace } from "../api";
import Quantum, { Player, PLAY_MS, stateColor } from "../pages/Quantum";
import About from "../pages/About";
import { int } from "../format";
import { fixtures, mockApi, renderPage } from "./utils";

const trace = fixtures.trace as unknown as GasTrace;
const slider = () => screen.getByRole("slider", { name: "step" }) as HTMLInputElement;
const shown = () => Number(screen.getByTestId("step-index").textContent);

describe("Grover player", () => {
  it("steps through `steps` in order with next/prev, and the slider equals the step index", async () => {
    render(<Player trace={trace} />);
    expect(shown()).toBe(0);
    for (let i = 1; i < Math.min(trace.steps.length, 6); i++) {
      await userEvent.click(screen.getByRole("button", { name: "next round" }));
      expect(shown()).toBe(i);
      expect(Number(slider().value)).toBe(i);
      expect(screen.getByTestId("oracle-calls")).toHaveTextContent(int(trace.steps[i].oracle_calls));
    }
    await userEvent.click(screen.getByRole("button", { name: "previous round" }));
    expect(Number(slider().value)).toBe(shown());
  });

  it("plays every step in order to the end, then stops", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    try {
      render(<Player trace={trace} />);
      await userEvent.setup({ advanceTimers: vi.advanceTimersByTime }).click(screen.getByRole("button", { name: "Play" }));
      const seen = [shown()];
      for (let i = 0; i < trace.steps.length + 2; i++) {
        await act(() => vi.advanceTimersByTimeAsync(PLAY_MS));
        if (shown() !== seen.at(-1)) seen.push(shown());
      }
      expect(seen).toEqual(trace.steps.map((s) => s.step));
      expect(screen.getByRole("button", { name: "Play" })).toBeInTheDocument();
    } finally {
      vi.useRealTimers();
    }
  });

  it("colours y* coral, marked states periwinkle, the rest muted", () => {
    const states = trace.steps.flatMap((s) => s.top);
    const yStar = states.find((s) => s.index === trace.y_star_index)!;
    expect(stateColor(yStar, trace.y_star_index)).toBe("#F2706A");
    const marked = states.find((s) => s.marked && s.index !== trace.y_star_index);
    if (marked) expect(stateColor(marked, trace.y_star_index)).toBe("#8E9BFF");
    const other = states.find((s) => !s.marked && s.index !== trace.y_star_index)!;
    expect(stateColor(other, trace.y_star_index)).toBe("#D6CFEA");
  });

  it("shows the trace totals and P(y*) of the current step", () => {
    render(<Player trace={trace} />);
    expect(screen.getByTestId("total-queries")).toHaveTextContent(int(trace.totals.oracle_calls + trace.totals.measurements));
    expect(screen.getByTestId("exhaustive")).toHaveTextContent(int(trace.totals.exhaustive));
    expect(screen.getByTestId("p-ystar")).toHaveTextContent(trace.steps[0].p_y_star.toFixed(3));
  });

  it("loads the trace for an allowed seed, and the simulated box from /api/claims", async () => {
    const fetchMock = mockApi();
    renderPage(<Quantum />, "/quantum?size=small_v2");
    expect(await screen.findByRole("slider", { name: "step" })).toBeInTheDocument();
    expect(fetchMock.mock.calls.map((c) => String(c[0]))).toContain("/api/gas-trace/small_v2?seed=0");
    expect(await screen.findByText("What is quantum here, and what is simulated")).toBeInTheDocument();
    expect(screen.getByRole("combobox", { name: "GAS seed" })).toHaveDisplayValue("0");
  });
});

describe("About", () => {
  it("renders every note, every caveat and every method's reference from /api/claims", async () => {
    mockApi();
    renderPage(<About />, "/about?size=small_v2");
    const notes = await screen.findByTestId("notes");
    expect(notes.querySelectorAll("li")).toHaveLength(fixtures.claims.notes.length);
    fixtures.claims.notes.forEach((n) => expect(notes).toHaveTextContent(n));
    expect(screen.getByTestId("caveats").querySelectorAll("li")).toHaveLength(fixtures.claims.caveats.length);
    const methods = screen.getAllByTestId("method");
    expect(methods).toHaveLength(fixtures.claims.methods.length);
    fixtures.claims.methods.forEach((m, i) => expect(methods[i]).toHaveTextContent(m.reference ?? "no reference"));
  });
});
