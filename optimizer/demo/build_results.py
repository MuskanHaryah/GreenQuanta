"""Guide 1 step 1.5: precompute results/mids/<size>/results.json (the cached data Guide 2 serves).

Methods: quantum (GAS -> csqa, 2D) and control (GAS -> csa, 2D) from s1_end_to_end.hybrid (log rows appended);
exact_r1 from results/phase2/<size>/assignment_exact.csv (read only); the 7 CloudSim baselines. Every placement is
validated with phase2.validation, scored with phase2.energy_model and linear_cost, and replayed in CloudSim.
    .venv-gq/Scripts/python.exe -B optimizer/demo/build_results.py --size small_v2
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demo.baselines import POLICIES, UNIT, run_baseline
from demo.cloudsim import OUT_ROOT, replay
from diagnostics.r3_run import historical_r2, saved_sample
from diagnostics.s1_end_to_end import RUN_CONFIG, hybrid
from diagnostics.common import build_problem
from diagnostics.s1_ground_truth import RUN_CONFIG as GT_CONFIG, ground_truth
from phase2 import config
from phase2.energy_model import evaluate_allocation
from phase2.exact_solver import linear_cost
from phase2.manifest import git_info
from phase2.validation import validate_allocation

BL = "CloudSim Plus 8.0.0 built-in"
META = {
    "quantum": ("Quantum pipeline (GAS -> constraint-preserving SQA)", "quantum",
                "Grover Adaptive Search (statevector simulation) picks the servers to switch on; constraint-preserving "
                "simulated quantum annealing (path-integral Monte Carlo) places the VMs on them.",
                "Dürr & Høyer 1996, arXiv:quant-ph/9607014 (quantum minimum finding); Gilliam, Woerner & Gonciulea "
                "2021, Quantum 5, 428 (Grover Adaptive Search); Hen & Spedalieri 2016, Phys. Rev. Applied 5, 034007 "
                "and Hen & Sarandy 2016, Phys. Rev. A 93, 062312 (constraint-preserving drivers); Martoňák, Santoro & "
                "Tosatti 2002, Phys. Rev. B 66, 094203 (path-integral SQA)"),
    "control": ("Classical control (GAS -> constraint-preserving SA)", "classical-control",
                "Same Stage 1 and the same moves and budget as the quantum pipeline, with classical simulated annealing "
                "in Stage 2.", "Kirkpatrick, Gelatt & Vecchi 1983, Science 220, 671 (simulated annealing)"),
    "exact_r1": ("Exact optimum R1 (MILP, real sizes)", "exact",
                 "Mixed-integer linear program on the real VM sizes (SciPy HiGHS), proven optimal for the linear energy "
                 "model; the reference every method is compared with.",
                 "Huangfu & Hall 2018, Math. Prog. Comp. 10, 119 (HiGHS); Virtanen et al. 2020, Nature Methods 17, 261 "
                 "(SciPy milp)"),
    "cs_firstfit": ("CloudSim FirstFit (next-fit)", "baseline",
                    f"{BL} VmAllocationPolicyFirstFit, VMs in input order; it resumes at the last used host, so it "
                    "behaves as circular next-fit.",
                    "CloudSim Plus 8.0.0, VmAllocationPolicyFirstFit; Silva Filho et al. 2017, IFIP/IEEE IM "
                    "(CloudSim Plus)"),
    "cs_bestfit": ("CloudSim BestFit", "baseline",
                   f"{BL} VmAllocationPolicyBestFit (active host with the fewest free CPU units), VMs in input order.",
                   "CloudSim Plus 8.0.0, VmAllocationPolicyBestFit; Silva Filho et al. 2017, IFIP/IEEE IM "
                   "(CloudSim Plus)"),
    "cs_worstfit": ("CloudSim WorstFit (Simple)", "baseline",
                    f"{BL} VmAllocationPolicySimple (active host with the most free CPU units), VMs in input order.",
                    "CloudSim Plus 8.0.0, VmAllocationPolicySimple; Silva Filho et al. 2017, IFIP/IEEE IM "
                    "(CloudSim Plus)"),
    "cs_roundrobin": ("CloudSim RoundRobin", "baseline",
                      f"{BL} VmAllocationPolicyRoundRobin, VMs in input order.",
                      "CloudSim Plus 8.0.0, VmAllocationPolicyRoundRobin; Silva Filho et al. 2017, IFIP/IEEE IM "
                      "(CloudSim Plus)"),
    "cs_ffd": ("First-Fit Decreasing", "baseline",
               "Textbook first-fit (lowest-index host that fits) with VMs sorted by CPU size descending, run in "
               "CloudSim Plus.", "Johnson 1973 (bin-packing FFD)"),
    "cs_bfd": ("Best-Fit Decreasing", "baseline",
               f"{BL} VmAllocationPolicyBestFit with VMs sorted by CPU size descending.", "Johnson 1973 (bin-packing BFD)"),
    "pabfd": ("PABFD (implemented per Beloglazov & Buyya 2012)", "baseline",
              "Power-Aware Best-Fit Decreasing: VMs by CPU descending, each to the host with the smallest power "
              "increase (SPEC curve; an empty host is off). Not built into CloudSim Plus 8; implemented on its "
              "allocation hook.",
              "Beloglazov & Buyya 2012, Concurrency Computat.: Pract. Exper. 24(13), 1397 (PABFD)"),
}
HOST_KEYS = ("host_id", "type", "active", "vms", "mean_cpu_util_pct", "peak_cpu_util_pct", "mem_util_pct", "energy_kwh")


def entry(method_id, vms, servers, refs, assignment, found=True, unplaced=0, energy=None, joules=None, cs=None,
          gap=None, resources=None, runtime=None, out_root=OUT_ROOT, size=None):
    """One methods[<id>] record; energy / joules / CloudSim are computed here unless the caller already has them."""
    label, kind, algorithm, reference = META[method_id]
    complete = assignment is not None and len(assignment) == len(vms) and not unplaced
    if complete and energy is None:
        energy = evaluate_allocation(vms, assignment, servers)
        joules = linear_cost(assignment, vms, servers)
        cs = replay(size, method_id, vms, servers, assignment, out_root)
    assignment = assignment or {}
    return {
        "label": label, "kind": kind, "algorithm": algorithm, "reference": reference,
        "found": bool(found and complete), "unplaced_vms": int(unplaced),
        "valid_real": validate_allocation(vms, assignment, servers)["valid"],
        "valid_quantized": validate_allocation(vms, assignment, servers, quantized_unit=UNIT)["valid"],
        "active_servers": sorted({servers[h].server_id for h in assignment.values()}),
        "assignment": {str(v): servers[h].server_id for v, h in sorted(assignment.items())},
        "linear_joules": joules,
        "diff_r1_pct": None if joules is None else 100 * (joules - refs["r1_joules"]) / refs["r1_joules"],
        "diff_r2_pct": None if joules is None else 100 * (joules - refs["r2_joules"]) / refs["r2_joules"],
        "gap_to_h_min_pct": gap,
        "energy": None if energy is None else {
            "it_energy_kwh": energy["it_energy_kwh"], "facility_energy_kwh": energy["facility_energy_kwh"],
            "co2_kg": energy["co2_kg"], "per_host": [{k: h[k] for k in HOST_KEYS} for h in energy["per_host"]]},
        "cloudsim": None if cs is None else {k: cs[k] for k in ("it_energy_kwh", "co2_kg", "diff_it_pct_vs_python", "ok")},
        "resources": resources or {},
        "runtime_s": runtime,
    }


def notes(size, problem, references, stage1, gt):
    calls = stage1["gas_oracle_calls"] + stage1["gas_measurements"]
    loss = 100 * (gt["step0_joules"] - references["r1_joules"]) / references["r1_joules"]
    return [
        "Everything quantum is simulated on a classical computer (statevector Grover Adaptive Search, path-integral "
        "Monte Carlo SQA); no quantum speed-up is claimed.",
        "Stage 2 shows no SQA advantage over classical SA: the classical control was better on medium and large at "
        "both budgets, every seed (Step 2D.4).",
        f"Stage 1 on {size}: {calls} GAS oracle calls + measurements vs {stage1['exhaustive_evaluations']} exhaustive "
        "evaluations. At m = 8 (small, medium) GAS costs more than exhaustive (R7); on large it is 1,053 vs 4,096.",
        "The constraint-preserving SQA (one-hot hopping driver) is not runnable on today's quantum annealers, which "
        "only offer a transverse-field driver.",
        f"Quantization loss on {size}: the best placement of the quantized model (unit {problem['unit']} NCU) costs "
        f"{loss:+.3f} % joules vs the real-size exact optimum R1, so a quantized optimum can still lose to R1.",
    ]


def build(size, out_root=OUT_ROOT):
    vms, servers, bqm, _ = build_problem(size)
    gt = ground_truth(size)
    m = len(servers)
    _, r1_assignment = saved_sample(config.RESULTS_DIR / size / "assignment_exact.csv", servers, bqm)
    _, r2_joules, r2_kwh = historical_r2(size)
    references = {"h_min": GT_CONFIG["step0_h_min"][size], "r1_joules": linear_cost(r1_assignment, vms, servers),
                  "r2_joules": r2_joules, "r2_curve_it_kwh": r2_kwh}
    problem = {"vms": len(vms), "unit": UNIT, "m": m, "servers": [
        {"id": s.server_id, "type": s.type_name, "cpu_capacity": s.cpu_capacity, "mem_capacity": s.mem_capacity,
         "cpu_limit": s.cpu_limit, "mem_limit": s.mem_limit, "idle_w": s.power_w[0], "max_w": s.power_w[-1]}
        for s in servers]}
    methods, stage1 = {}, None
    for method_id, stage2 in (("quantum", "csqa"), ("control", "csa")):
        out, assignment = hybrid(size, "gas", stage2, "2D", log=True)
        resources = {k: out[k] for k in ("gas_budget", "gas_runs", "gas_oracle_calls", "gas_measurements",
                                         "stage2_solves", "stage2_move_attempts")}
        methods[method_id] = entry(method_id, vms, servers, references, assignment, found=out["found"],
                                   gap=out.get("gap_to_h_min_pct"), resources=resources, runtime=out["wall_s"],
                                   out_root=out_root, size=size)
        if method_id == "quantum":
            stage1 = {"m": m, "states": 2 ** m, **{k: out[k] for k in ("gas_budget", "gas_runs", "gas_oracle_calls",
                                                                       "gas_measurements")},
                      "exhaustive_evaluations": 2 ** m, "y_star_ids": gt["y_star_ids"],
                      "chosen_ids": out.get("chosen_ids")}
    methods["exact_r1"] = entry("exact_r1", vms, servers, references, r1_assignment, out_root=out_root, size=size)
    for policy in POLICIES:
        r = run_baseline(size, policy, vms, servers, out_root)
        method_id = "pabfd" if policy == "pabfd" else f"cs_{policy}"
        methods[method_id] = entry(method_id, vms, servers, references, r["assignment"], found=r["found"],
                                   unplaced=r["unplaced_vms"], energy=r["energy"], joules=r["linear_joules"],
                                   cs=r["cloudsim"], runtime=r["runtime_s"], out_root=out_root, size=size)
    result = {"size": size, "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "git_commit": git_info(config.ROOT)["code_revision"], "problem": problem, "references": references,
              "stage1": stage1, "methods": methods, "notes": notes(size, problem, references, stage1, gt)}
    path = Path(out_root) / size / "results.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return result, path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", choices=RUN_CONFIG["sizes"], required=True)
    started = time.perf_counter()
    result, path = build(parser.parse_args().size)
    for method_id, r in result["methods"].items():
        cs = r["cloudsim"] or {}
        print(f"RESULT | {result['size']} | {method_id} | found={r['found']} | valid={r['valid_real']}/"
              f"{r['valid_quantized']} | active={r['active_servers']} | dR1={r['diff_r1_pct']} | "
              f"dR2={r['diff_r2_pct']} | IT={r['energy'] and r['energy']['it_energy_kwh']} | "
              f"CS diff={cs.get('diff_it_pct_vs_python')} ok={cs.get('ok')} |", flush=True)
    print(f"wrote {path} in {time.perf_counter() - started:.0f} s")


if __name__ == "__main__":
    main()
