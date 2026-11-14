// Display formatting only (fixed decimals: kWh 3, % 2, joules with thousands separators).
const DASH = "—";
const nf = (digits: number) => new Intl.NumberFormat("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });
const f3 = nf(3), f2 = nf(2), f0 = nf(0);

export const kwh = (x: number | null | undefined) => (x == null ? DASH : `${f3.format(x)} kWh`);
export const kg = (x: number | null | undefined) => (x == null ? DASH : `${f3.format(x)} kg`);
export const joules = (x: number | null | undefined) => (x == null ? DASH : `${f0.format(x)} J`);
export const int = (x: number) => f0.format(x);
export const pct = (x: number | null | undefined) => (x == null ? DASH : `${f2.format(x)} %`);
/** Signed percentage: "+7.69 %", "−1.20 %", "0.00 %". */
export const signedPct = (x: number | null | undefined) =>
  x == null ? DASH : `${x > 0 ? "+" : x < 0 ? "−" : ""}${f2.format(Math.abs(x))} %`;
export const signed = (x: number, unit: string, digits = 3) =>
  `${x > 0 ? "+" : x < 0 ? "−" : ""}${nf(digits).format(Math.abs(x))} ${unit}`;
export const prob = (x: number) => f3.format(x);
export const seconds = (x: number) => `${nf(1).format(x)} s`;
export const two = (n: number) => String(n).padStart(2, "0");
/** "large_v2" → "Large" (the audience never sees the internal size ids). */
export const sizeName = (id: string) => id.charAt(0).toUpperCase() + id.split("_")[0].slice(1);
