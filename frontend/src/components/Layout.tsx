// Page frame: top bar (logo, tag, links, size switcher), left rail of the page's sections, phone menu, blobs, footer.
import { useEffect, useState } from "react";
import { NavLink, Outlet, useLocation } from "react-router-dom";
import { api } from "../api";
import { useApi, useSize } from "../hooks";
import { PAGES, pageOf } from "../routes";
import Blobs from "./Blobs";

function SizeSwitcher() {
  const [size, setSize] = useSize();
  const sizes = useApi(api.sizes);
  return (
    <div className="pills" role="group" aria-label="problem size">
      {(sizes.data ?? []).map((s) => (
        <button key={s.id} className="pill" aria-pressed={s.id === size} disabled={!s.available}
          title={`${s.vms} VMs on ${s.servers} servers${s.available ? "" : " (results not built)"}`}
          onClick={() => setSize(s.id)}>{s.label}</button>
      ))}
    </div>
  );
}

function Links({ onNavigate }: { onNavigate?: () => void }) {
  const { search } = useLocation();
  return (
    <>
      {PAGES.map((p) => (
        <NavLink key={p.path} to={{ pathname: p.path, search }} end onClick={onNavigate}>{p.label}</NavLink>
      ))}
    </>
  );
}

function SectionLinks({ active, onPick }: { active: string; onPick: (id: string) => void }) {
  const { sections } = pageOf(useLocation().pathname);
  return (
    <>
      {sections.map((s, i) => (
        <span key={s.id} style={{ display: "contents" }}>
          {i > 0 && <span className="dash" aria-hidden="true" />}
          <button className={`rail-label${s.id === active ? " active" : ""}`} aria-current={s.id === active || undefined}
            onClick={() => onPick(s.id)}>{s.label}</button>
        </span>
      ))}
    </>
  );
}

function scrollTo(id: string) {
  const reduce = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
  document.getElementById(id)?.scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "start" });
}

export default function Layout() {
  const { pathname, search } = useLocation();
  const page = pageOf(pathname);
  const [open, setOpen] = useState(false);
  const [visible, setVisible] = useState<string | null>(null);
  const health = useApi(api.health);
  const active = visible && page.sections.some((s) => s.id === visible) ? visible : page.sections[0].id;

  useEffect(() => {
    if (typeof IntersectionObserver === "undefined") return;
    const observer = new IntersectionObserver((entries) => {
      const hit = entries.filter((e) => e.isIntersecting).sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top)[0];
      if (hit) setVisible(hit.target.id);
    }, { rootMargin: "-20% 0px -55% 0px" });
    page.sections.forEach((s) => { const el = document.getElementById(s.id); if (el) observer.observe(el); });
    return () => observer.disconnect();
  }, [page]);

  const pick = (id: string) => { setOpen(false); setVisible(id); scrollTo(id); };

  return (
    <>
      <Blobs />
      <header className="topbar">
        <NavLink to={{ pathname: "/", search }} className="logo">GREENQUANTA</NavLink>
        <span className="tag">quantum vm placement</span>
        <nav className="nav" aria-label="pages"><Links /></nav>
        <SizeSwitcher />
        <button className="hamburger" aria-label="menu" aria-expanded={open} aria-controls="phone-menu"
          onClick={() => setOpen((o) => !o)}><span /><span /><span /></button>
      </header>
      <aside className="rail" aria-label="sections"><SectionLinks active={active} onPick={pick} /></aside>
      {open && (
        <div id="phone-menu" className="menu" role="dialog" aria-label="menu">
          <button className="circle menu-close" aria-label="close menu" onClick={() => setOpen(false)}>✕</button>
          <nav aria-label="pages (menu)"><Links onNavigate={() => setOpen(false)} /></nav>
          <div className="menu-sections"><SectionLinks active={active} onPick={pick} /></div>
          <SizeSwitcher />
        </div>
      )}
      <main className="main"><Outlet /></main>
      <footer className="footer">
        GreenQuanta · mid-evaluation demo · everything quantum is simulated on a classical computer
        {health.data && <> · build {health.data.git_commit.slice(0, 8)}</>}
      </footer>
    </>
  );
}
