// Small shared building blocks: one component per concept.
import type { ReactNode } from "react";
import type { ApiError } from "../api";
import type { Loaded } from "../hooks";
import { methodColor, OUTLINED } from "../methodColors";
import { two } from "../format";

/** Condensed uppercase headline: first word periwinkle, second white. */
export function Headline({ first, second, small, as: Tag = "h2" }:
  { first: string; second: string; small?: boolean; as?: "h1" | "h2" }) {
  return (
    <Tag className={`headline${small ? " small" : ""}`}>
      <span className="hl-1">{first}</span> <span className="hl-2">{second}</span>
    </Tag>
  );
}

export function Section({ id, first, second, children }: { id: string; first: string; second: string; children: ReactNode }) {
  return (
    <section id={id} className="section" aria-label={`${first} ${second}`}>
      <Headline first={first} second={second} small />
      {children}
    </section>
  );
}

export const Dots = () => <hr className="dots" aria-hidden="true" />;

export function KpiCard({ index, title, value, sub, peach, children }:
  { index: number; title: string; value: ReactNode; sub?: ReactNode; peach?: boolean; children?: ReactNode }) {
  return (
    <div className="kpi">
      <div className="numeral" aria-hidden="true">{two(index)}</div>
      <article className={`card${peach ? " peach" : ""}`} aria-label={title}>
        <h3 className="card-title">{title}</h3>
        <div className="card-value">{value}</div>
        {sub && <p className="card-sub">{sub}</p>}
        {children}
      </article>
    </div>
  );
}

export function MethodBadge({ id, label }: { id: string; label: string }) {
  return (
    <span className="badge">
      <span className={`swatch${OUTLINED.has(id) ? " outlined" : ""}`}
        style={OUTLINED.has(id) ? undefined : { background: methodColor(id) }} aria-hidden="true" />
      {label}
    </span>
  );
}

const QUANTIZED_TIP = "fits the real servers; does not fit the quantum model's rounding grid";

/** ✓/✗ for a validity flag. Only valid_real ✗ is "bad"; a valid_quantized ✗ is muted with an explanation. */
export function Check({ ok, kind = "real" }: { ok: boolean; kind?: "real" | "quantized" }) {
  if (ok) return <span className="check ok" aria-label="valid">✓</span>;
  return kind === "real"
    ? <span className="check bad" aria-label="invalid">✗</span>
    : <span className="check soft" title={QUANTIZED_TIP} aria-label={`not on grid: ${QUANTIZED_TIP}`}>✗</span>;
}

export function ErrorPanel({ error, retry }: { error: ApiError; retry: () => void }) {
  return (
    <div className="error-panel" role="alert">
      <h3>API problem</h3>
      <p>{error.message} <span className="muted">({error.code}{error.status ? `, HTTP ${error.status}` : ""})</span></p>
      <button className="btn btn-coral" onClick={retry}>Retry</button>
    </div>
  );
}

export const Skeleton = () => (
  <div className="skeleton" aria-busy="true" aria-label="loading"><div /><div /><div /></div>
);

/** Skeleton while loading, an error panel with retry on failure, else children(data). */
export function Async<T>({ state, children }: { state: Loaded<T>; children: (data: T) => ReactNode }) {
  if (state.error) return <ErrorPanel error={state.error} retry={state.retry} />;
  if (state.loading || state.data === undefined) return <Skeleton />;
  return <>{children(state.data)}</>;
}

export function CircleButton({ label, onClick, disabled, children }:
  { label: string; onClick: () => void; disabled?: boolean; children: ReactNode }) {
  return <button className="circle" aria-label={label} title={label} onClick={onClick} disabled={disabled}>{children}</button>;
}
