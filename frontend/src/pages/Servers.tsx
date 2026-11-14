import { useState } from "react";
import { createPortal } from "react-dom";
import { api, type CompareRow, type Host } from "../api";
import { useApi, useSize } from "../hooks";
import { bestBaseline } from "../selectors";
import { methodColor, methodName, shown } from "../methods";
import { int } from "../format";
import { Async, MethodBadge, Section, useEscape } from "../components/ui";

const share = (load: number, limit: number) => Math.min(100, (100 * load) / limit);
const ncu = (x: number) => x.toFixed(4);

interface Picked { method: string; host: Host }

function Tile({ host, color, onPick }: { host: Host; color: string; onPick: () => void }) {
  const cpu = share(host.cpu_load, host.cpu_limit), mem = share(host.mem_load, host.mem_limit);
  return (
    <button className={`tile ${host.active ? "on" : "off"}`} style={{ ["--tile-color" as string]: color }}
      data-testid="tile" data-active={host.active} onClick={onPick}
      aria-label={`server ${host.host_id}, type ${host.type}, ${host.active
        ? `on, ${host.vms.length} VMs, CPU ${cpu.toFixed(0)} %, memory ${mem.toFixed(0)} %` : "off"}`}>
      <b>#{host.host_id}</b>
      {host.active ? (
        <>
          {host.vms.length} VMs
          <div className="bar" title={`CPU ${cpu.toFixed(1)} % of limit`}><i style={{ width: `${cpu}%` }} /></div>
          <div className="bar mem" title={`memory ${mem.toFixed(1)} % of limit`}><i style={{ width: `${mem}%` }} /></div>
        </>
      ) : <span className="muted">off</span>}
    </button>
  );
}

export function Datacentre({ size, method, label, onPick }:
  { size: string; method: string; label: string; onPick: (p: Picked) => void }) {
  const placement = useApi(api.placement, size, method);
  return (
    <div className="dc" data-testid={`dc-${method}`}>
      <h3><MethodBadge id={method} label={label} /></h3>
      <Async state={placement}>{(p) => {
        const types = [...new Set(p.hosts.map((h) => h.type))];
        const on = p.hosts.filter((h) => h.active).length;
        return (
          <>
            <p className="muted" style={{ margin: 0, fontSize: 13 }}>{on} of {p.hosts.length} servers on</p>
            {types.map((t) => (
              <div key={t}>
                <div className="dc-type">type {t}</div>
                <div className="tiles">
                  {p.hosts.filter((h) => h.type === t).map((h) => (
                    <Tile key={h.host_id} host={h} color={methodColor(method)} onPick={() => onPick({ method: label, host: h })} />
                  ))}
                </div>
              </div>
            ))}
          </>
        );
      }}</Async>
    </div>
  );
}

function Drawer({ picked, onClose }: { picked: Picked; onClose: () => void }) {
  const h = picked.host;
  useEscape(onClose);
  return (
    <aside className="drawer" role="dialog" aria-label={`server ${h.host_id} VMs`}>
      <button className="circle" style={{ float: "right" }} aria-label="close" onClick={onClose} autoFocus>✕</button>
      <h3>Server #{h.host_id}</h3>
      <p className="muted">{picked.method}, type {h.type}, {h.active ? `${h.vms.length} VMs` : "switched off"}</p>
      <p style={{ fontSize: 13 }}>CPU {ncu(h.cpu_load)} of {ncu(h.cpu_limit)}, memory {ncu(h.mem_load)} of {ncu(h.mem_limit)}</p>
      {h.vms.length > 0 && (
        <div className="table-wrap">
          <table>
            <thead><tr><th>VM</th><th className="num">CPU</th><th className="num">Memory</th></tr></thead>
            <tbody>
              {h.vms.map((v) => (
                <tr key={v.vm_id} data-testid="drawer-vm">
                  <td>{int(v.vm_id)}</td>
                  <td className="num">{ncu(v.cpu_size)}<span className="mini"><i style={{ width: `${share(v.cpu_size, h.cpu_limit)}%` }} /></span></td>
                  <td className="num">{ncu(v.mem_size)}<span className="mini mem"><i style={{ width: `${share(v.mem_size, h.mem_limit)}%` }} /></span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </aside>
  );
}

function Picker({ rows: all, sizeKey }: { rows: CompareRow[]; sizeKey: string }) {
  const rows = all.filter((r) => shown(r.method_id));
  const best = bestBaseline(rows.map((r) => [r.method_id, r] as [string, CompareRow]));
  const [chosen, setChosen] = useState<{ key: string; a: string; b: string } | null>(null);
  const current = chosen?.key === sizeKey ? chosen : { key: sizeKey, a: "quantum", b: best?.[0] ?? rows[0].method_id };
  const [picked, setPicked] = useState<Picked | null>(null);
  const label = (id: string) => methodName(id, rows.find((r) => r.method_id === id)?.label);
  const select = (side: "a" | "b") => (
    <label className="field">
      {side === "a" ? "Left" : "Right"} method
      <select value={current[side]} onChange={(e) => setChosen({ ...current, [side]: e.target.value })}>
        {rows.map((r) => <option key={r.method_id} value={r.method_id}>{methodName(r.method_id, r.label)}</option>)}
      </select>
    </label>
  );
  return (
    <>
      <div className="row" style={{ marginBottom: 18 }}>{select("a")}{select("b")}</div>
      <div className="compare-grids">
        {(["a", "b"] as const).map((side) => (
          <Datacentre key={side} size={sizeKey} method={current[side]} label={label(current[side])} onPick={setPicked} />
        ))}
      </div>
      {picked && createPortal(   // on top of the whole page (top bar, rail), with a backdrop that closes it
        <>
          <div className="drawer-backdrop" onClick={() => setPicked(null)} />
          <Drawer picked={picked} onClose={() => setPicked(null)} />
        </>, document.body)}
    </>
  );
}

export default function Servers() {
  const [size] = useSize();
  const rows = useApi(api.compare, size);
  return (
    <Section id="datacentre" first="Server" second="Datacentre">
      <p className="muted">Each tile is a server. Lit tiles are switched on, the top bar shows CPU use and the bottom
        bar memory. Click a server to see its VMs.</p>
      <Async state={rows}>{(r) => <Picker rows={r} sizeKey={size} />}</Async>
    </Section>
  );
}
