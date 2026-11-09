"""Local R3/R3v diagnostics; no sampling, runners, or result-directory writes.

Use --audit to capture prerequisites, diagnostics, hashes and tests in a new
stepR3b_raw_output.txt. Without --audit this script only prints diagnostics.
"""
import argparse
import contextlib
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

import numpy as np
from scipy.optimize import LinearConstraint
from scipy.sparse import coo_matrix, vstack

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diagnostics.common import CENTER, ENCODING, LAMBDA_CONFIG, build_problem
from diagnostics.d0_formulation import assignment_sample, independent_terms, state_validation
from diagnostics.d2_scale import TERMS, decompose, score_assignment
from diagnostics import exact_qubo
from diagnostics.exact_quantized import solve_exact_quantized
from phase1.qubo_builder import x_var, y_var
from phase2 import config
from phase2.energy_model import evaluate_allocation, linear_energy_terms
from phase2.exact_solver import linear_cost
from phase2.qubo_builder import units
from phase2.qubo_builder_v2 import build_qubo
from phase2.validation import validate_allocation

SIZES = ("small_v2", "medium_v2", "large_v2")
TIME_LIMIT = 600
MAX_ITERS = 50
MARGIN = 1 - 1e-6
TOLERANCE = 1e-9
REPORT_DIR = config.ROOT / "docs" / "reports"


def score_state(sample, vms, servers, bqm, info, energy):
    terms = independent_terms(sample, vms, servers, info, energy)
    independent_h = math.fsum(terms.values())
    bqm_h = float(bqm.energy(sample))
    if abs(independent_h - bqm_h) > TOLERANCE:
        raise AssertionError(("independent/BQM mismatch", independent_h, bqm_h))
    idle, _, e_ref = linear_energy_terms(vms, servers)
    tie = math.fsum((energy.linear[y_var(h)] - idle[h] / e_ref) * sample[y_var(h)]
                    for h in range(len(servers)))
    return {"sample": sample, "bqm_H": bqm_h, "independent_H": independent_h,
            "incumbent_true_H": independent_h,
            "breakdown": {**terms, "energy": terms["energy"] - tie, "tie": tie}}


def strict_state(sample, vms, servers):
    """Shared validator plus explicit bits and the requested tolerance guard."""
    check = state_validation(sample, vms, servers)
    margin_excess = []
    for h, server in enumerate(servers):
        for resource in ("cpu", "mem"):
            load = math.fsum(getattr(vm, f"{resource}_size") * sample[x_var(vm.vm_id, h)]
                             for vm in vms)
            limit = getattr(server, f"{resource}_limit") * MARGIN
            if load > limit:
                margin_excess.append({"host": h, "resource": resource,
                                      "load": load, "guarded_limit": limit})
    return {**check, "valid": check["valid"] and not margin_excess,
            "margin_excess": margin_excess}


def solve_valid_qubo(vms, servers, time_limit=TIME_LIMIT, max_iters=MAX_ITERS):
    """Append hard rows to R3a's unchanged square-epigraph/cutting-plane MILP.

    R3a's all-zero fallback is not feasible here. Track and score feasible MILP
    candidates separately, and certify them only against its relaxation bounds.
    """
    start = time.perf_counter()
    n, m = len(vms), len(servers)
    nx, nb = n * m, n * m + m
    bqm, info = build_qubo(vms, servers, lambda_config=LAMBDA_CONFIG,
                           encoding=ENCODING, center=CENTER)
    energy, _ = build_qubo(vms, servers, lambda_config="L0", encoding=ENCODING, center=CENTER)
    names = [x_var(vm.vm_id, h) for vm in vms for h in range(m)]
    names += [y_var(h) for h in range(m)]
    original_milp = exact_qubo.milp
    best = None

    def constrained_milp(cost, *, constraints, **kwargs):
        nonlocal best
        rr, cc, vv, lows, highs = [], [], [], [], []

        def row(coeffs, lo, hi):
            r = len(lows)
            for column, value in coeffs.items():
                rr.append(r)
                cc.append(column)
                vv.append(value)
            lows.append(lo)
            highs.append(hi)

        for i in range(n):
            row({i * m + h: 1 for h in range(m)}, 1, 1)
        for h, server in enumerate(servers):
            for resource in ("cpu", "mem"):
                row({**{i * m + h: getattr(vm, f"{resource}_size")
                         for i, vm in enumerate(vms)},
                     nx + h: -getattr(server, f"{resource}_limit") * MARGIN}, -np.inf, 0)
        for i in range(n):
            for h in range(m):
                row({i * m + h: 1, nx + h: -1}, -np.inf, 0)
        extra = coo_matrix((vv, (rr, cc)), shape=(len(lows), len(cost))).tocsc()
        augmented = LinearConstraint(vstack([constraints.A, extra], format="csc"),
                                     np.r_[constraints.lb, lows], np.r_[constraints.ub, highs])
        # Include adapter construction in the same wall-clock budget.
        kwargs["options"]["time_limit"] = max(1e-6, time_limit - (time.perf_counter() - start))
        result = original_milp(cost, constraints=augmented, **kwargs)
        if result.x is not None:
            bits = np.rint(result.x[:nb]).astype(int)
            if np.max(np.abs(result.x[:nb] - bits)) > 1e-7:
                raise AssertionError("nonbinary constrained incumbent")
            sample = dict(zip(names, map(int, bits)))
            validation = strict_state(sample, vms, servers)
            if not validation["valid"]:
                raise AssertionError(("constrained incumbent failed strict validation", validation))
            candidate = score_state(sample, vms, servers, bqm, info, energy)
            if best is None or candidate["bqm_H"] < best["bqm_H"]:
                best = {**candidate, "validation": validation}
        return result

    with patch.object(exact_qubo, "milp", constrained_milp):
        base = exact_qubo.solve_exact_qubo(vms, servers, time_limit=time_limit,
                                          max_iters=max_iters)
    # A solved relaxation is a certificate for this restricted domain, even if
    # R3a retained its lower-H but now inadmissible all-zero fallback.
    proven = (best is not None and base["milp_status"] == 0 and base["converged"]
              and abs(best["bqm_H"] - base["lower_bound"]) <= TOLERANCE)
    if best is None:
        best = {"sample": None, "bqm_H": None, "independent_H": None,
                "incumbent_true_H": None, "breakdown": None,
                "validation": {"valid": False, "reason": "no feasible incumbent"}}
        assignment, hosts = {}, set()
    else:
        sample = best["sample"]
        assignment = {vm.vm_id: next(h for h in range(m) if sample[x_var(vm.vm_id, h)])
                      for vm in vms}
        hosts = {h for h in range(m) if sample[y_var(h)]}
    gap = None if best["bqm_H"] is None else max(0, best["bqm_H"] - base["lower_bound"]) / max(
        abs(best["bqm_H"]), 1e-10)
    return {**best, "assignment": assignment, "hosts": hosts,
            "status": "OPTIMAL" if proven else "UNPROVEN", "proven_optimal": bool(proven),
            "lower_bound": base["lower_bound"], "mip_gap": gap,
            "iterations": base["iterations"], "reason": base["reason"],
            "time_s": time.perf_counter() - start}


def mechanical_label(result, real_valid, cost, reference_cost):
    if not result["proven_optimal"]:
        return "UNPROVEN", None
    if not real_valid:
        return "INVALID", None
    pct = 100 * (cost - reference_cost) / reference_cost
    return ("MATCH" if abs(pct) <= 0.1 else "DIFFERENT"), pct


def saved_sample(path, servers, bqm):
    indices = {s.server_id: h for h, s in enumerate(servers)}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        pairs = [(int(r["vm_id"]), indices[int(r["host_id"])]) for r in csv.DictReader(stream)]
    if len(dict(pairs)) != len(pairs):
        raise ValueError(f"duplicate saved VM: {path}")
    return assignment_sample(dict(pairs), bqm), dict(pairs)


def recorded_gate():
    print("=== A: large_v2 saved-assignment safeguard (D3 has no BQM hash) ===", flush=True)
    if (config.UNIT_NCU, config.UNIT_NMU) != (0.0025, 0.0025):
        raise ValueError("resource units differ from D2/D3")
    vms, servers, bqm, _ = build_problem("large_v2")
    parts, _ = decompose(vms, servers, bqm)
    with (config.ROOT / "results/diagnostics/d2/large_v2.csv").open(newline="") as stream:
        records = {r["assignment"]: r for r in csv.DictReader(stream) if r["kind"] == "assignment"}
    for name in ("assignment_exact.csv", "assignment_annealer_only.csv"):
        actual = score_assignment(config.RESULTS_DIR / "large_v2" / name, vms, servers, bqm, parts)
        for key in ("total_h", *(f"{t}_h" for t in TERMS)):
            expected = float(records[name][key])
            print(f"{name} {key}: actual={actual[key]:.17g} recorded={expected:.17g}")
            if not math.isclose(actual[key], expected, rel_tol=0, abs_tol=TOLERANCE):
                raise AssertionError(("D2 safeguard mismatch", name, key, actual[key], expected))
        rounded = ({"total_h": (0.4870, 4)} if name == "assignment_exact.csv" else
                   {"total_h": (1.134, 3), "energy_h": (0.864, 3),
                    "cpu_h": (0.147, 3), "ram_h": (0.123, 3)})
        for key, (expected, digits) in rounded.items():
            if round(actual[key], digits) != expected:
                raise AssertionError(("user rounded safeguard mismatch", name, key))
    with (config.ROOT / "results/diagnostics/d3/large_v2.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            if abs(float(row["h_exact"]) - float(records["assignment_exact.csv"]["total_h"])) > TOLERANCE:
                raise AssertionError("D3 h_exact safeguard mismatch")
    print("Saved large_v2 total/split safeguard: PASS", flush=True)


def print_hosts(hosts, servers):
    return [{"index": h, "host_id": servers[h].server_id, "type": servers[h].type_name}
            for h in sorted(hosts)]


def analyze(result, vms, servers, bqm):
    sample = result["sample"]
    print("servers_on=", len(result["hosts"]), "hosts=", print_hosts(result["hosts"], servers))
    placements = {vm.vm_id: [h for h in range(len(servers)) if sample[x_var(vm.vm_id, h)]]
                  for vm in vms}
    print("unplaced_VMs=", [v for v, hs in placements.items() if not hs])
    print("duplicated_VMs=", {v: hs for v, hs in placements.items() if len(hs) > 1})
    for h, server in enumerate(servers):
        for resource, unit in (("cpu", config.UNIT_NCU), ("mem", config.UNIT_NMU)):
            k = math.floor(getattr(server, f"{resource}_limit") / unit + 1e-9)
            selected = [vm for vm in vms if h in placements[vm.vm_id]]
            load_units = sum(units(getattr(vm, f"{resource}_size"), unit) for vm in selected)
            load = math.fsum(getattr(vm, f"{resource}_size") for vm in selected)
            limit = getattr(server, f"{resource}_limit")
            print(f"host_id={server.server_id} type={server.type_name} on={sample[y_var(h)]} "
                  f"{resource}: units={load_units} K={k} overload_units={max(0, load_units-k)} "
                  f"overload_units_pct={100*max(0, load_units-k)/k:.9f} "
                  f"real_load={load:.17g} limit={limit:.17g} "
                  f"real_overload={max(0, load-limit):.17g} "
                  f"real_overload_pct={100*max(0, load-limit)/limit:.9f}")
    print("H_breakdown=", result["breakdown"])
    real = state_validation(sample, vms, servers)
    assignment = result["assignment"]
    quant = validate_allocation(vms, assignment, servers, quantized_unit=config.UNIT_NCU)
    quant["valid"] = quant["valid"] and real["valid"]
    print("real_size_validator=", real)
    print("quantized_validator=", quant)
    complete = all(len(hs) == 1 for hs in placements.values())
    cost = None
    if complete:
        cost = linear_cost(assignment, vms, servers)
        print(f"linear_cost_J={cost:.17g} SPEC_curve_energy_kWh="
              f"{evaluate_allocation(vms, assignment, servers)['it_energy_kwh']:.17g}")
    else:
        print("linear cost and SPEC energy: unavailable (not a complete singleton assignment)")
    return real["valid"], cost


def historical_r2(size):
    text = (REPORT_DIR / "step2e_raw_output.txt").read_text(encoding="utf-8")
    section = text.split(f"=== {size} ===", 1)[1].split("===", 1)[0]
    section = section.split("quantized MILP:", 1)[1]
    match = re.search(r"host_ids=(\[[^\]]+\]).*?linear_cost_J=([\d.]+) "
                      r"curve_it_energy_kWh=([\d.]+)", section, re.S)
    if match is None:
        raise ValueError(f"Step 2e R2 reference missing: {size}")
    return json.loads(match[1]), float(match[2]), float(match[3])


def diagnose(size):
    print(f"\n=== {size}: {LAMBDA_CONFIG}, {ENCODING}, center={CENTER}, "
          f"unit={config.UNIT_NCU}, tie_break={config.TIE_BREAK}, seed={config.SEED} ===", flush=True)
    vms, servers, bqm, info = build_problem(size)
    print(f"VMs={len(vms)} servers={len(servers)} time_limit_s={TIME_LIMIT} max_iters={MAX_ITERS}")
    print("B: global QUBO", flush=True)
    result = exact_qubo.solve_exact_qubo(vms, servers, time_limit=TIME_LIMIT, max_iters=MAX_ITERS)
    energy, _ = build_qubo(vms, servers, lambda_config="L0", encoding=ENCODING, center=CENTER)
    rescored = score_state(result["sample"], vms, servers, bqm, info, energy)
    print("R3=", {key: result[key] for key in ("status", "proven_optimal", "reason", "lower_bound",
          "incumbent_true_H", "independent_H", "bqm_H", "mip_gap", "iterations", "time_s")})
    print("recomputed_H=", {k: rescored[k] for k in ("bqm_H", "independent_H")})
    real_valid, cost = analyze(result, vms, servers, bqm)

    print("D: R3v; same H plus singleton, links and real-size guarded capacity rows", flush=True)
    valid = solve_valid_qubo(vms, servers, time_limit=TIME_LIMIT, max_iters=MAX_ITERS)
    print("R3v=", {k: valid[k] for k in ("status", "proven_optimal", "lower_bound", "bqm_H",
          "independent_H", "mip_gap", "iterations", "time_s", "validation")})
    print("H_valid_min=", valid["bqm_H"] if valid["proven_optimal"] else "UNPROVEN",
          "H_valid_incumbent=", valid["bqm_H"])
    print("R3v servers=", print_hosts(valid["hosts"], servers))
    if valid["sample"] is not None:
        print("R3v linear_cost_J=", linear_cost(valid["assignment"], vms, servers))

    print("E: same-BQM references", flush=True)
    r1_sample, r1 = saved_sample(config.RESULTS_DIR / size / "assignment_exact.csv", servers, bqm)
    h_real = float(bqm.energy(r1_sample))
    print(f"R1 saved real-MILP: H={h_real:.17g} linear_cost_J={linear_cost(r1,vms,servers):.17g} "
          f"SPEC_curve_energy_kWh={evaluate_allocation(vms,r1,servers)['it_energy_kwh']:.17g}")
    print("R1 real_size_validator=", state_validation(r1_sample, vms, servers))
    print("R1 quantized_validator=", validate_allocation(vms, r1, servers, quantized_unit=config.UNIT_NCU))
    r2_hosts, r2_cost, r2_curve = historical_r2(size)
    print(f"Step 2e R2: host_ids={r2_hosts} linear_cost_J={r2_cost:.17g} "
          f"SPEC_curve_energy_kWh={r2_curve:.17g}; label_reference=R2 linear joules")
    quantized, qinfo = solve_exact_quantized(vms, servers, time_limit=TIME_LIMIT)
    print("R2 recomputation=", qinfo)
    if quantized is None:
        raise RuntimeError("quantized MILP reference unavailable; stop")
    h_quantized = float(bqm.energy(assignment_sample(quantized, bqm)))
    qcost = linear_cost(quantized, vms, servers)
    print(f"R2 H_quantized={h_quantized:.17g} linear_cost_J={qcost:.17g} "
          f"SPEC_curve_energy_kWh={evaluate_allocation(vms,quantized,servers)['it_energy_kwh']:.17g} "
          f"hosts={print_hosts(set(quantized.values()), servers)}")
    print("R2 validators=", validate_allocation(vms, quantized, servers),
          validate_allocation(vms, quantized, servers, quantized_unit=config.UNIT_NCU))
    if not math.isclose(qcost, r2_cost, rel_tol=1e-9, abs_tol=1e-6):
        raise AssertionError(("R2 cost differs from Step 2e; stop", qcost, r2_cost))
    d3_path = config.ROOT / "results" / "diagnostics" / "d3" / f"{size}.csv"
    best_annealer = None
    if d3_path.is_file():
        with d3_path.open(newline="") as stream:
            values = [float(r["min_h_valid"]) for r in csv.DictReader(stream) if r["min_h_valid"]]
        best_annealer = min(values, default=None)
    print("D3 best valid H (recorded)=", best_annealer)
    print("D3 bqm.energy reference: SKIPPED; best-sample x/y bits are absent from D3 CSVs. "
          "Recorded scalar only; no sampling or state reconstruction.")
    short, pct = mechanical_label(result, real_valid, cost, r2_cost)
    label = {"UNPROVEN": "UNPROVEN", "INVALID": "R3_INVALID", "MATCH": "R3_VALID_MATCH",
             "DIFFERENT": "R3_VALID_DIFFERENT"}[short]
    print(f"LABEL {size}: {label}; linear_cost_difference_vs_R2_pct={pct}")
    if label == "UNPROVEN":
        print(f"UNPROVEN lower_bound={result['lower_bound']:.17g} "
              f"incumbent_true_H={result['incumbent_true_H']:.17g}")
    print("| H_min (or incumbent if UNPROVEN) | H_valid_min (or incumbent if UNPROVEN) | "
          "H_quantized | best annealer H (D3 recorded only) |")
    print("|---|---|---|---|")
    print(f"| {result['bqm_H']:.17g} ({result['status']}) | {valid['bqm_H']} ({valid['status']}) | "
          f"{h_quantized:.17g} | {best_annealer} |", flush=True)


def snapshot():
    output = {}
    for directory in ("results/phase2", "results/runs"):
        files = [(p.relative_to(config.ROOT).as_posix(), hashlib.sha256(p.read_bytes()).hexdigest())
                 for p in sorted((config.ROOT / directory).rglob("*")) if p.is_file()]
        digest = hashlib.sha256(json.dumps(files, separators=(",", ":")).encode()).hexdigest()
        output[directory] = (files, digest)
    return output


def print_snapshot(label, snap):
    for directory, (files, digest) in snap.items():
        print(f"{label} {directory} count={len(files)} sha256={digest}")
        for path, sha in files:
            print(f"{label} {sha}  {path}")


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def command(args, timeout):
    print("COMMAND:", subprocess.list2cmdline(args), flush=True)
    start = time.perf_counter()
    try:
        proc = subprocess.run(args, cwd=config.ROOT, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as error:
        output = error.stdout or b""
        print(output.decode(errors="replace") if isinstance(output, bytes) else output)
        print(f"TIMEOUT after {timeout}s")
        raise
    print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n")
    print(f"exit_code={proc.returncode} total_time_s={time.perf_counter()-start:.3f}")
    if proc.returncode:
        raise RuntimeError(f"command failed ({proc.returncode})")


def audit(sizes):
    raw = REPORT_DIR / "stepR3b_raw_output.txt"
    if (REPORT_DIR / "stepR3b_report.md").exists():
        raise FileExistsError("R3b report already exists; refusing to overwrite")
    before = snapshot()
    log = config.ROOT / "results/experiment_log.csv"
    log_hash = hashlib.sha256(log.read_bytes()).hexdigest() if log.is_file() else None
    with raw.open("x", encoding="utf-8") as stream, contextlib.redirect_stdout(Tee(sys.stdout, stream)), \
            contextlib.redirect_stderr(Tee(sys.stderr, stream)):
        print("Step R3b; local only; dependencies added: none")
        print_snapshot("BEFORE", before)
        try:
            command(["git", "--no-pager", "branch", "--show-current"], 10)
            branch = subprocess.check_output(["git", "--no-pager", "branch", "--show-current"],
                                             cwd=config.ROOT, text=True).strip()
            if branch != "step3-r3-exact-qubo":
                raise RuntimeError(f"unexpected branch: {branch}")
            command([sys.executable, "-B", "-m", "pytest", "tests/test_exact_qubo.py", "-v",
                     "-p", "no:cacheprovider"], 600)
            recorded_gate()
            command([sys.executable, "-B", "-m", "pytest", "tests/test_r3_run.py", "-v",
                     "-p", "no:cacheprovider"], 600)
            for size in sizes:
                diagnose(size)
            command([sys.executable, "-B", "-m", "pytest", "tests", "-v",
                     "-p", "no:cacheprovider"], 600)
        except BaseException:
            traceback.print_exc()
            raise
        finally:
            after = snapshot()
            print_snapshot("AFTER", after)
            for directory in before:
                print(f"{directory} identical={'yes' if before[directory] == after[directory] else 'no'} "
                      f"counts={len(before[directory][0])}/{len(after[directory][0])}")
            current = hashlib.sha256(log.read_bytes()).hexdigest() if log.is_file() else None
            print(f"experiment_log identical={'yes' if log_hash == current else 'no'}")
            command(["git", "--no-pager", "diff", "--exit-code"], 10)
            command(["git", "--no-pager", "diff", "--cached", "--exit-code"], 10)
            if before != after or log_hash != current:
                raise AssertionError("protected files changed")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", choices=SIZES, default=list(SIZES))
    parser.add_argument("--audit", action="store_true", help="capture raw gates in a new docs report file")
    args = parser.parse_args(argv)
    if len(args.sizes) != len(set(args.sizes)):
        parser.error("--sizes must not repeat")
    if args.audit:
        audit(args.sizes)
    else:
        recorded_gate()
        for size in args.sizes:
            diagnose(size)


if __name__ == "__main__":
    main()
