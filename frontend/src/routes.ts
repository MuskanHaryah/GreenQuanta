// The pages and the sections each one shows on the left rail (ids are the <section> ids).
export interface PageDef { path: string; label: string; sections: Array<{ id: string; label: string }> }

export const PAGES: PageDef[] = [
  { path: "/", label: "Overview", sections: [{ id: "intro", label: "Intro" }, { id: "kpis", label: "Results" }] },
  { path: "/pipeline", label: "Pipeline",
    sections: [{ id: "stages", label: "Stages" }, { id: "live", label: "Live run" }] },
  { path: "/servers", label: "Servers",
    sections: [{ id: "datacentre", label: "Datacentre" }] },
  { path: "/compare", label: "Compare",
    sections: [{ id: "charts", label: "Charts" }, { id: "table", label: "Table" }] },
  { path: "/quantum", label: "Quantum",
    sections: [{ id: "grover", label: "Grover" }, { id: "simulated", label: "Simulated" }] },
  { path: "/about", label: "About", sections: [{ id: "methods", label: "Methods" }, { id: "claims", label: "Claims" }] },
];

export const pageOf = (pathname: string) => PAGES.find((p) => p.path === pathname) ?? PAGES[0];
