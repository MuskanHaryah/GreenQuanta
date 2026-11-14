import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { CompareRow, Placement } from "../api";
import { bestBaseline } from "../selectors";
import Servers from "../pages/Servers";
import { fixtures, mockApi, renderPage } from "./utils";

const quantum = fixtures.placementQuantum as Placement;
const rows = fixtures.compare as unknown as CompareRow[];

describe("Servers", () => {
  it("loads two placements: quantum and the best baseline", async () => {
    const fetchMock = mockApi();
    renderPage(<Servers />, "/servers?size=small_v2");
    const best = bestBaseline(rows.map((r) => [r.method_id, r] as [string, CompareRow]))![0];
    expect(best).toBe("pabfd");   // the second fixture is this placement
    await screen.findByTestId("dc-quantum");
    expect(await screen.findByTestId(`dc-${best}`)).toBeInTheDocument();
    const paths = fetchMock.mock.calls.map((c) => String(c[0]));
    expect(paths).toContain("/api/results/small_v2/placement/quantum");
    expect(paths).toContain(`/api/results/small_v2/placement/${best}`);
  });

  it("has one tile per server, on/off as in `active`", async () => {
    mockApi();
    renderPage(<Servers />, "/servers?size=small_v2");
    const grid = await screen.findByTestId("dc-quantum");
    const tiles = await within(grid).findAllByTestId("tile");
    expect(tiles).toHaveLength(quantum.hosts.length);
    expect(quantum.hosts.length).toBe(fixtures.results.problem.servers.length);
    const byId = new Map(quantum.hosts.map((h) => [h.host_id, h]));
    for (const tile of tiles) {
      const id = Number(tile.getAttribute("aria-label")!.match(/^server (\d+),/)![1]);
      expect(tile.getAttribute("data-active")).toBe(String(byId.get(id)!.active));
      expect(tile).toHaveClass(byId.get(id)!.active ? "on" : "off");
    }
  });

  it("opens a drawer listing exactly the clicked host's VMs", async () => {
    mockApi();
    renderPage(<Servers />, "/servers?size=small_v2");
    const grid = await screen.findByTestId("dc-quantum");
    const host = quantum.hosts.find((h) => h.active)!;
    const tile = (await within(grid).findAllByTestId("tile")).find((t) => t.getAttribute("aria-label")!.startsWith(`server ${host.host_id},`))!;
    await userEvent.click(tile);
    const drawer = screen.getByRole("dialog", { name: `server ${host.host_id} VMs` });
    const vmIds = within(drawer).getAllByTestId("drawer-vm").map((tr) => Number(tr.firstChild!.textContent!.replace(/,/g, "")));
    expect(vmIds).toEqual(host.vms.map((v) => v.vm_id));
    await userEvent.keyboard("{Escape}");
    expect(screen.queryByRole("dialog", { name: `server ${host.host_id} VMs` })).toBeNull();
  });

  it("loads another placement when a selector changes", async () => {
    const fetchMock = mockApi({ "/api/results/small_v2/placement/control": { ...quantum, method_id: "control" } });
    renderPage(<Servers />, "/servers?size=small_v2");
    await screen.findByTestId("dc-quantum");
    await userEvent.selectOptions(screen.getByRole("combobox", { name: "Left method" }), "control");
    expect(await screen.findByTestId("dc-control")).toBeInTheDocument();
    expect(fetchMock.mock.calls.map((c) => String(c[0]))).toContain("/api/results/small_v2/placement/control");
  });
});
