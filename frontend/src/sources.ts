// Turns a method's reference text (from /api/claims) into a list of sources with links. Every link below was checked
// on 2026-11-14 to resolve to the cited work (DOI metadata or page title).
const LINKS: Array<[RegExp, string]> = [
  [/Dürr & Høyer/, "https://arxiv.org/abs/quant-ph/9607014"],
  [/Gilliam/, "https://doi.org/10.22331/q-2021-04-08-428"],
  [/Hen & Spedalieri/, "https://doi.org/10.1103/PhysRevApplied.5.034007"],
  [/Hen & Sarandy/, "https://doi.org/10.1103/PhysRevA.93.062312"],
  [/Marto/, "https://doi.org/10.1103/PhysRevB.66.094203"],
  [/Huangfu & Hall/, "https://doi.org/10.1007/s12532-017-0130-5"],
  [/Virtanen/, "https://doi.org/10.1038/s41592-019-0686-2"],
  [/Beloglazov & Buyya/, "https://doi.org/10.1002/cpe.1867"],
  [/Silva Filho/, "https://doi.org/10.23919/INM.2017.7987304"],
  [/^CloudSim Plus 8/, "https://github.com/cloudsimplus/cloudsimplus"],
  [/^Johnson 1973/, "https://hdl.handle.net/1721.1/57819"],
];

export interface Source { text: string; url?: string }

export function sources(reference: string | null): Source[] {
  if (!reference) return [];
  return reference
    .split(/;\s*|\s+and\s+(?=Hen & Sarandy)/)
    .map((text) => text.trim())
    .filter(Boolean)
    .map((text) => ({ text, url: LINKS.find(([re]) => re.test(text))?.[1] }));
}
