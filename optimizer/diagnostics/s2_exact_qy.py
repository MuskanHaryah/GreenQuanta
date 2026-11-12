"""Step 2.2: exact check of the reduced Stage-2 QUBO Q_y at y = y* on the real sizes. Stdout + one log row per size.

Exact fixed-y minimum at lambda_y = factor * U_y (s1_ground_truth.stage2 with lam replaced; r3c_run.solve_fixed).
Gates: equal to V(y*) from 1.3 (exact sum, |diff| <= tie_tol), same linear joules (rel 1e-9), real + quantized
valid, proven; the solver's minimiser lies in the Q_y domain (off-host x and slack = 0) and Q_y.energy of it
agrees with the exact H (bqm_agrees). Also records the coefficient profile (review R3), variable counts
(review R5) and the load factor, for: full QUBO at lambda = 1.1*U, fixed-y only at 1.1*U (Step 1 probe),
Q_y at 1.1*U, Q_y at lambda_y.
"""
import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diagnostics.exact_qubo import bqm_agrees
from diagnostics.s1_ground_truth import RUN_CONFIG as S1, candidates, emit, ground_truth, log_run, records, stage2
from diagnostics.s1_stage1 import bits
from diagnostics.s2_stage2 import coefficient_profile, load_factor, stage2_bqm

RUN_CONFIG = {"sizes": S1["sizes"], "unit": S1["unit"], "factor": S1["factor"], "tie_tol": S1["tie_tol"],
              "joule_rel_tol": 1e-9}


def run(size):
    vms, servers, data, costs, _, _ = candidates(size)
    m, unit, factor = len(servers), RUN_CONFIG["unit"], RUN_CONFIG["factor"]
    gt = ground_truth(size)
    index = gt["y_star"]
    v13 = [r for r in records("V", size) if r["index"] == index][-1]
    y = bits(index, m)
    q_y, qinfo = stage2_bqm(vms, servers, y, unit, factor)               # lambda_y
    q_u, uinfo = stage2_bqm(vms, servers, y, unit, lam=data["lam"])      # lambda = 1.1*U, reduced
    row, result = stage2(vms, servers, dict(data, lam=qinfo["lam"]), costs, index)
    sample = result["sample"]
    in_domain = sample is not None and all(int(sample[k]) == v for k, v in qinfo["fixed"].items())
    restricted = {k: int(sample[k]) for k in q_y.variables} if in_domain else None
    q_energy = float(q_y.energy(restricted)) if in_domain else None
    agrees = in_domain and bqm_agrees(q_y, restricted, row["V"], q_energy)
    y_fixed = {k: v for k, v in qinfo["fixed"].items() if k.startswith("y_")}
    profiles = {"full_1.1U": coefficient_profile(uinfo["full"], uinfo["energy"], {}),
                "fixed_y_only_1.1U": coefficient_profile(uinfo["full"], uinfo["energy"], y_fixed),
                "Q_y_1.1U": coefficient_profile(uinfo["full"], uinfo["energy"], uinfo["fixed"]),
                "Q_y_lambda_y": coefficient_profile(qinfo["full"], qinfo["energy"], qinfo["fixed"])}
    diff = row["V"] - v13["V"]
    joules_ok = (row["linear_joules"] is not None and
                 math.isclose(row["linear_joules"], v13["linear_joules"], rel_tol=RUN_CONFIG["joule_rel_tol"]))
    gate = bool(row["proven"] and abs(diff) <= RUN_CONFIG["tie_tol"] and joules_ok and row["real_valid"]
                and row["quantized_valid"] and agrees)
    summary = {"size": size, "y_star": index, "active_ids": row["active_ids"], "U": data["U"],
               "lam": data["lam"], "U_y": qinfo["U_y"], "lam_y": qinfo["lam"], "lam_y_over_lam": qinfo["lam"] / data["lam"],
               "W_lambda_y": row["V"], "V_1_3": v13["V"], "W_minus_V": diff, "joules": row["linear_joules"],
               "joules_1_3": v13["linear_joules"], "proven": row["proven"], "status": row["status"],
               "real_valid": row["real_valid"], "quantized_valid": row["quantized_valid"],
               "minimiser_in_Q_y_domain": in_domain, "Q_y_energy": q_energy, "Q_y_agrees": bool(agrees),
               "Q_y_variables": qinfo["num_variables"], "Q_y_x_variables": qinfo["x_variables"],
               "Q_y_slack_variables": qinfo["slack_variables"], "load_factor": load_factor(qinfo["info"], qinfo["on"]),
               "profiles": profiles, "time_s": row["time_s"], "GATE": gate}
    emit("QY_EXACT", **summary)
    p = profiles
    print(f"TABLE | {size} | {row['active_ids']} | {row['V']:.12g} | {v13['V']:.12g} | {diff:.3g} | "
          f"{'Y' if joules_ok else 'N'} | {'Y' if row['real_valid'] and row['quantized_valid'] else 'N'} | "
          f"{'Y' if agrees else 'N'} | {'PASS' if gate else 'FAIL'} |", flush=True)
    print(f"PROFILE | {size} | vars full/fixed-y/Q_y = {p['full_1.1U']['variables']}/"
          f"{p['fixed_y_only_1.1U']['variables']}/{p['Q_y_lambda_y']['variables']} "
          f"(slack {qinfo['slack_variables']}) | lam {data['lam']:.6g} -> lam_y {qinfo['lam']:.6g} | "
          + " | ".join(f"{k}: max|coef| {v['max_abs_coef']:.6g}, pen/energy {v['penalty_to_energy']:.6g}"
                       for k, v in p.items())
          + " | load " + ", ".join(f"{r} {v:.4f}" for r, v in summary["load_factor"].items()) + " |", flush=True)
    log_run("s2_exact_qy", dict(RUN_CONFIG, size=size), 0, size, vms, summary, "PASS" if gate else "FAIL")
    emit("DONE", size=size)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", choices=RUN_CONFIG["sizes"], required=True)
    args = parser.parse_args()
    print("RUN_CONFIG=" + json.dumps(RUN_CONFIG, sort_keys=True), flush=True)
    run(args.size)


if __name__ == "__main__":
    main()
