// How each method is shown: a plain name for the audience, a one-line description and one flat colour.
// The API ids and numbers are unchanged; only the wording on screen comes from here.
export interface MethodView { name: string; text: string; color: string }

export const METHODS: Record<string, MethodView> = {
  quantum: { name: "Quantum pipeline", color: "#F2706A",
    text: "Grover search picks which servers to switch on, then quantum annealing places every VM on them." },
  exact_r1: { name: "Exact optimum", color: "#FFFFFF",
    text: "A mathematical solver that finds the lowest energy placement possible. Every method is measured against it." },
  pabfd: { name: "Power Aware Best Fit Decreasing", color: "#FF9FC0",
    text: "Places the largest VMs first, each on the server where it adds the least power." },
  cs_firstfit: { name: "CloudSim Next Fit", color: "#F49A72",
    text: "The CloudSim FirstFit policy. It continues from the last server it used, so it works as next fit." },
  cs_bestfit: { name: "CloudSim Best Fit", color: "#6FC3C8",
    text: "Puts each VM on the busiest server that still has room." },
  cs_worstfit: { name: "CloudSim Worst Fit", color: "#C9B8FF",
    text: "Puts each VM on the server with the most free room." },
  cs_roundrobin: { name: "CloudSim Round Robin", color: "#FFD27A",
    text: "Hands out VMs to the servers in turn." },
  cs_ffd: { name: "First Fit Decreasing", color: "#A0A4B8",
    text: "Sorts VMs from largest to smallest and puts each on the first server that fits." },
  cs_bfd: { name: "Best Fit Decreasing", color: "#7FD6A6",
    text: "Sorts VMs from largest to smallest and puts each on the busiest server that still fits." },
};

/** Methods the site does not show (the classical control run). */
export const HIDDEN = new Set(["control"]);
export const shown = (id: string) => !HIDDEN.has(id);

export const OUTLINED = new Set(["exact_r1"]);
export const methodColor = (id: string) => METHODS[id]?.color ?? "#D6CFEA";
export const methodName = (id: string, fallback = id) => METHODS[id]?.name ?? fallback;
