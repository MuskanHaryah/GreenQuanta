// Typed client for the Guide 2 API (contract: docs/plans/STEP4_PLAN.md). The frontend only displays these numbers.

export type SizeId = string;
export type Stage2 = "csqa" | "csa";
export type Kind = "quantum" | "classical-control" | "exact" | "baseline";

export interface Health { status: "ok"; git_commit: string; results_available: Record<SizeId, boolean> }

export interface SizeInfo { id: SizeId; label: string; vms: number; servers: number; m: number; available: boolean }

export interface Server {
  id: number; type: string; cpu_capacity: number; mem_capacity: number; cpu_limit: number; mem_limit: number;
  idle_w: number; max_w: number;
}

export interface HostEnergy {
  host_id: number; type: string; active: boolean; vms: number; mean_cpu_util_pct: number; peak_cpu_util_pct: number;
  mem_util_pct: number; energy_kwh: number;
}

export interface MethodResult {
  label: string; kind: Kind; algorithm: string; reference: string | null; found: boolean; unplaced_vms: number;
  valid_real: boolean; valid_quantized: boolean; active_servers: number[]; linear_joules: number | null;
  diff_r1_pct: number | null; diff_r2_pct: number | null; gap_to_h_min_pct: number | null;
  energy: { it_energy_kwh: number; facility_energy_kwh: number; co2_kg: number; per_host: HostEnergy[] } | null;
  cloudsim: { it_energy_kwh: number; co2_kg: number; diff_it_pct_vs_python: number; ok: boolean } | null;
  resources: Record<string, number>; runtime_s: number | null;
}

export interface Results {
  size: SizeId; generated_at: string; git_commit: string;
  problem: { vms: number; unit: number; m: number; servers: Server[] };
  references: { h_min: number; r1_joules: number; r2_joules: number; r2_curve_it_kwh: number };
  stage1: {
    m: number; states: number; gas_budget: number; gas_runs: number; gas_oracle_calls: number; gas_measurements: number;
    exhaustive_evaluations: number; y_star_ids: number[]; chosen_ids: number[];
  };
  methods: Record<string, MethodResult>;
  notes: string[];
}

export interface PlacedVm { vm_id: number; cpu_size: number; mem_size: number }
export interface Host {
  host_id: number; type: string; active: boolean; cpu_limit: number; mem_limit: number; cpu_load: number;
  mem_load: number; vms: PlacedVm[];
}
export interface Placement { method_id: string; hosts: Host[] }

export interface CompareRow {
  method_id: string; label: string; kind: Kind; valid_real: boolean; valid_quantized: boolean; active_servers: number;
  diff_r1_pct: number | null; diff_r2_pct: number | null; gap_to_h_min_pct: number | null;
  it_energy_kwh: number | null; co2_kg: number | null; cloudsim_ok: boolean | null;
}

export interface TraceState { index: number; ids: number[]; p: number; cost: number; marked: boolean }
export interface TraceStep {
  step: number; oracle_calls: number; rotations: number; threshold: number | null; marked_count: number;
  measured_index: number; measured_cost: number; improved: boolean; top: TraceState[]; p_y_star: number;
  p_marked_total: number;
}
export interface GasTrace {
  size: SizeId; seed: number; m: number; states: number; budget: number; y_star_index: number; y_star_ids: number[];
  steps: TraceStep[]; totals: { oracle_calls: number; measurements: number; exhaustive: number };
}

export type EventKind = "PROPOSALS" | "SKIP" | "STAGE2" | "HYBRID" | "LOGGED" | "RESULT";
export interface RunEvent { kind: EventKind; t: number; data: Record<string, unknown> }
export interface RunOut {
  chosen_ids: number[]; real_valid: boolean; quantized_valid: boolean; linear_joules: number; diff_r1_pct: number;
  diff_r2_pct: number; gap_to_h_min_pct: number; curve_it_energy_kwh: number; gas_oracle_calls: number;
  gas_measurements: number; gas_runs: number; stage2_s: number;
}
export interface RunEnergy { active_hosts: number; it_energy_kwh: number; facility_energy_kwh: number; co2_kg: number }
export interface Job {
  job_id: string; size: SizeId; stage2: Stage2; status: "queued" | "running" | "done" | "failed" | "cancelled"; started_at: string;
  elapsed_s: number; events: RunEvent[];
  result: { out: RunOut; assignment: Record<string, number> | null; energy: RunEnergy | null } | null;
  matches_cached: boolean | null; error: string | null;
}

export interface MethodMeta { method_id: string; label: string; kind: Kind; algorithm: string; reference: string | null }
export interface Claims { size: SizeId; notes: string[]; caveats: string[]; methods: MethodMeta[] }

/** The one error type: an API error body ({error: {code, message}}), or a network failure (status 0). */
export class ApiError extends Error {
  constructor(public status: number, public code: string, message: string, public jobId?: string) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, init);
  } catch (e) {
    throw new ApiError(0, "network", `the API is not reachable (${(e as Error).message})`);
  }
  const body = await res.json().catch(() => null);
  if (!res.ok) {
    const err = body?.error;
    throw new ApiError(res.status, err?.code ?? "http_error", err?.message ?? `HTTP ${res.status}`, body?.job_id);
  }
  return body as T;
}

const q = encodeURIComponent;
export const api = {
  health: () => request<Health>("/api/health"),
  sizes: () => request<SizeInfo[]>("/api/sizes"),
  results: (size: SizeId) => request<Results>(`/api/results/${q(size)}`),
  placement: (size: SizeId, methodId: string) => request<Placement>(`/api/results/${q(size)}/placement/${q(methodId)}`),
  compare: (size: SizeId) => request<CompareRow[]>(`/api/results/${q(size)}/compare`),
  gasTrace: (size: SizeId, seed: number) => request<GasTrace>(`/api/gas-trace/${q(size)}?seed=${seed}`),
  claims: (size: SizeId) => request<Claims>(`/api/claims?size=${q(size)}`),
  /** 202 → job id; 409 → ApiError with jobId = the running job. */
  startRun: (size: SizeId, stage2: Stage2) =>
    request<{ job_id: string }>("/api/runs", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ size, stage2 }),
    }),
  run: (jobId: string) => request<Job>(`/api/runs/${q(jobId)}`),
  /** Stop a running job (409 not_running if it already finished). */
  cancelRun: (jobId: string) =>
    request<{ job_id: string; status: "cancelled" }>(`/api/runs/${q(jobId)}/cancel`, { method: "POST" }),
};
