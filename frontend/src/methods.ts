// One colour per method (flat). quantum = coral, control = periwinkle, exact_r1 = white outline, baselines in order.
export const METHOD_COLORS: Record<string, string> = {
  quantum: "#F2706A",
  control: "#8E9BFF",
  exact_r1: "#FFFFFF",
  cs_firstfit: "#F49A72",
  cs_bestfit: "#6FC3C8",
  cs_worstfit: "#C9B8FF",
  cs_roundrobin: "#FFD27A",
  cs_ffd: "#A0A4B8",
  cs_bfd: "#7FD6A6",
  pabfd: "#FF9FC0",
};
export const OUTLINED = new Set(["exact_r1"]);
export const methodColor = (id: string) => METHOD_COLORS[id] ?? "#D6CFEA";
