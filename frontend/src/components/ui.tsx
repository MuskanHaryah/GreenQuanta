// Small shared building blocks: one component per concept.
import { useEffect, type ReactNode } from "react";
import { createPortal } from "react-dom";
import type { ApiError } from "../api";
import type { Loaded } from "../hooks";
import { methodColor, methodName, OUTLINED } from "../methods";
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

export type Tone = "peach" | "ok" | "bad" | "periwinkle";

export function KpiCard({ index, title, value, sub, tone, children }:
  { index: number; title: string; value: ReactNode; sub?: ReactNode; tone?: Tone; children?: ReactNode }) {
  return (
    <div className="kpi">
      <div className="numeral" aria-hidden="true">{two(index)}</div>
      <article className="card" aria-label={title}>
        <h3 className="card-title">{title}</h3>
        <div className={`card-value${tone ? ` tone-${tone}` : ""}`}>{value}</div>
        {sub && <p className="card-sub">{sub}</p>}
        {children}
      </article>
    </div>
  );
}

export function MethodBadge({ id, label }: { id: string; label?: string }) {
  return (
    <span className="badge">
      <span className={`swatch${OUTLINED.has(id) ? " outlined" : ""}`}
        style={OUTLINED.has(id) ? undefined : { background: methodColor(id) }} aria-hidden="true" />
      {methodName(id, label)}
    </span>
  );
}

/** Calls onClose when Escape is pressed (dialogs and drawers). */
export function useEscape(onClose: () => void) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
}

/** A dialog over the whole page: dimmed backdrop, closes on Escape or a backdrop click, focus moves to its close button. */
export function Modal({ title, onClose, children, footer }:
  { title: string; onClose: () => void; children: ReactNode; footer?: ReactNode }) {
  useEscape(onClose);
  return createPortal(
    <div className="modal-layer">
      <div className="modal-backdrop" onClick={onClose} data-testid="modal-backdrop" />
      <div className="modal" role="dialog" aria-modal="true" aria-label={title}>
        <header className="modal-head">
          <h2>{title}</h2>
          <button className="circle" aria-label="close" onClick={onClose} autoFocus>✕</button>
        </header>
        <div className="modal-body">{children}</div>
        {footer && <footer className="modal-foot">{footer}</footer>}
      </div>
    </div>,
    document.body,
  );
}

export function ErrorPanel({ error, retry }: { error: ApiError; retry: () => void }) {
  return (
    <div className="error-panel" role="alert">
      <h3>Could not load the data</h3>
      <p>{error.message}</p>
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
