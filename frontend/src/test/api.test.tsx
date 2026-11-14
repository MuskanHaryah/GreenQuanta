import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { api, ApiError } from "../api";
import { useApi } from "../hooks";
import { Async } from "../components/ui";
import { fixtures, json, mockApi, renderPage } from "./utils";

describe("API client", () => {
  beforeEach(() => { mockApi(); });

  it("parses every fixture into the contract types", async () => {
    const h = await api.health();
    expect(h.status).toBe("ok");
    const sizes = await api.sizes();
    expect(sizes.map((s) => s.id)).toEqual(["small_v2", "medium_v2", "large_v2"]);
    const r = await api.results("small_v2");
    expect(r.size).toBe("small_v2");
    expect(Object.keys(r.methods)).toContain("quantum");
    expect(r.methods.quantum).not.toHaveProperty("assignment");
    expect(r.stage1.exhaustive_evaluations).toBe(2 ** r.stage1.m);
    const rows = await api.compare("small_v2");
    expect(rows).toHaveLength(Object.keys(r.methods).length);
    expect(rows.every((row) => typeof row.valid_real === "boolean")).toBe(true);
    const p = await api.placement("small_v2", "quantum");
    expect(p.method_id).toBe("quantum");
    expect(p.hosts).toHaveLength(r.problem.servers.length);
    const t = await api.gasTrace("small_v2", 0);
    expect(t.steps.length).toBeGreaterThan(0);
    expect(t.steps[0].top.length).toBeLessThanOrEqual(32);
    expect(t.totals.exhaustive).toBe(t.states);
    const c = await api.claims("small_v2");
    expect(c.size).toBe("small_v2");
    expect(c.notes.length).toBeGreaterThan(0);
    expect(c.methods.find((m) => m.method_id === "exact_r1")?.reference).toMatch(/HiGHS/);
  });

  it("turns error JSON into ApiError (code, message, status, 409 job id)", async () => {
    mockApi({
      "/api/results/nope": () => json({ error: { code: "unknown_size", message: "unknown size 'nope'" } }, 404),
      "/api/runs": () => json({ error: { code: "run_in_progress", message: "run abc is still running" }, job_id: "abc" }, 409),
    });
    const e1 = await api.results("nope").catch((e) => e);
    expect(e1).toBeInstanceOf(ApiError);
    expect(e1).toMatchObject({ status: 404, code: "unknown_size", message: "unknown size 'nope'" });
    const e2 = await api.startRun("small_v2", "csqa").catch((e) => e);
    expect(e2).toMatchObject({ status: 409, code: "run_in_progress", jobId: "abc" });
  });

  it("turns a network failure into ApiError status 0", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => { throw new TypeError("Failed to fetch"); }));
    await expect(api.health()).rejects.toMatchObject({ status: 0, code: "network" });
  });
});

describe("Layout", () => {
  beforeEach(() => { mockApi(); });

  it("renders the page links and the size switcher from /api/sizes", async () => {
    renderPage(<p>page</p>);
    const nav = screen.getByRole("navigation", { name: "pages" });
    expect(within(nav).getAllByRole("link").map((a) => a.textContent))
      .toEqual(["Overview", "Pipeline", "Servers", "Compare", "Quantum", "About"]);
    const group = screen.getAllByRole("group", { name: "problem size" })[0];
    const pills = await within(group).findAllByRole("button");
    expect(pills.map((b) => b.textContent)).toEqual(fixtures.sizes.map((s) => s.label));
    expect(within(group).getByRole("button", { name: "Small" })).toHaveAttribute("aria-pressed", "true");
    await userEvent.click(within(group).getByRole("button", { name: "Large" }));
    expect(within(group).getByRole("button", { name: "Large" })).toHaveAttribute("aria-pressed", "true");
  });

  it("toggles the phone menu with the hamburger", async () => {
    renderPage(<p>page</p>);
    const burger = screen.getByRole("button", { name: "menu" });
    expect(burger).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByRole("dialog", { name: "menu" })).toBeNull();
    await userEvent.click(burger);
    expect(burger).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByRole("dialog", { name: "menu" })).toBeInTheDocument();
    await userEvent.click(burger);
    expect(screen.queryByRole("dialog", { name: "menu" })).toBeNull();
  });

  it("shows an error panel with a working retry when the API is down", async () => {
    let up = false;
    mockApi({ "/api/claims?size=small_v2": () => (up ? json(fixtures.claims) : json({ error: { code: "x", message: "down" } }, 503)) });
    renderPage(<ClaimsCount />, "/about?size=small_v2");
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("down");
    up = true;
    await userEvent.click(within(alert).getByRole("button", { name: "Retry" }));
    expect(await screen.findByText(`${fixtures.claims.notes.length} notes`)).toBeInTheDocument();
  });
});

function ClaimsCount() {
  const claims = useApi(api.claims, "small_v2");
  return <Async state={claims}>{(c) => <p>{c.notes.length} notes</p>}</Async>;
}
