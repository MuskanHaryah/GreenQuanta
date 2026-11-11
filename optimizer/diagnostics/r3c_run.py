"""R3c local formulation/fixed-y diagnostics. Outputs only stdout or own reports."""
import argparse
import contextlib
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
import re
import subprocess
import sys
import time
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diagnostics import common, exact_qubo, r3_run
from diagnostics.exact_quantized import solve_exact_quantized
from diagnostics.r3c_build import build_problem_custom, build_qubo_custom
from phase1.qubo_builder import x_var, y_var
from phase2 import config
from phase2.exact_solver import linear_cost
from phase2.fingerprint import qubo_fingerprint
from phase2.qubo_builder import units

ROOT = config.ROOT
RAW = ROOT / "docs/reports/stepR3c_raw_output.txt"
REPORT = ROOT / "docs/reports/stepR3c_report.md"
RUN_CONFIG = {"seeds": {"small_v2": [0, 1, 2], "medium_v2": [0, 1, 2], "large_v2": [0]},
              "reads": 200, "sweeps": 2000, "sample_cap": 120, "solve_cap": 600,
              "max_iters": 50, "center": 0.02,
              "active_ids": {"small_v2": [2, 3], "medium_v2": [2, 3],
                             "large_v2": [4, 5, 9, 10, 11]}}
SIZES = tuple(RUN_CONFIG["seeds"])
VARIANTS = [{"id": "V0", "lambdas": [0.05] * 4, "unit": 0.0025}]
VARIANTS += [{"id": f"V1-L{v:g}", "lambdas": [v] * 4, "unit": 0.0025}
             for v in (0.2, 1, 5, 20, 100)]
VARIANTS += [{"id": f"V2-A{v:g}", "lambdas": [v, 0.05, 0.05, 0.05], "unit": 0.0025}
             for v in (0.5, 2)]
VARIANTS += [{"id": f"V3-U{v:g}", "lambdas": [0.05] * 4, "unit": v}
             for v in (0.001, 0.0005)]


def emit(kind, **data):
    print(kind + "_JSON=" + json.dumps(data, sort_keys=True, allow_nan=False), flush=True)


def records(kind):
    prefix = kind + "_JSON="
    return [json.loads(line[len(prefix):]) for line in RAW.read_text(encoding="utf-8").splitlines()
            if line.startswith(prefix)]


def problem(size, variant):
    return build_problem_custom(size, *variant["lambdas"], RUN_CONFIG["center"], variant["unit"])


def independent_terms(sample, vms, servers, info, energy):
    """Direct x/y residual formula, independent of penalty BQM coefficients."""
    lam = info["lambdas"]
    terms = {"energy": float(energy.energy(sample)), "assign": 0., "cpu": 0., "ram": 0., "link": 0.}
    terms["assign"] = lam["assign"] * math.fsum(
        (sum(sample[x_var(v.vm_id, h)] for h in range(len(servers))) - 1) ** 2 for v in vms)
    terms["link"] = lam["link"] * sum(sample[x_var(v.vm_id, h)] * (1 - sample[y_var(h)])
                                             for v in vms for h in range(len(servers)))
    for resource, tag, key in (("cpu", "cpu", "unit_ncu"), ("mem", "ram", "unit_nmu")):
        unit = info[key]
        values = []
        for h, server in enumerate(servers):
            cap = math.floor(getattr(server, resource + "_limit") / unit + 1e-9)
            load = sum(units(getattr(v, resource + "_size"), unit) * sample[x_var(v.vm_id, h)]
                       for v in vms)
            g = (cap * sample[y_var(h)] - load) / cap
            values.append(lam[tag] * (g * g - 2 * info["unbalanced_center"] * g))
        terms[tag] = math.fsum(values)
    return terms


def solver_context(variant):
    """Bind library functions to private globals; retain R3a's solver unchanged.

    R3v already intercepts the MILP to append hard rows. Its interception is
    confined to this private module, never the imported shared module/config.
    """
    unit = variant["unit"]
    local_config = types.SimpleNamespace(**vars(config))
    local_config.UNIT_NCU = local_config.UNIT_NMU = unit

    def builder(vms, servers, lambda_config=None, encoding="unbalanced", center=None):
        if encoding != "unbalanced":
            raise ValueError("R3c supports unbalanced only")
        weights = [0.] * 4 if lambda_config in ("L0", "L0.0") else variant["lambdas"]
        return build_qubo_custom(vms, servers, *weights, RUN_CONFIG["center"] if center is None else center, unit)

    local = types.ModuleType("r3c_private_exact")
    local.__dict__.update(vars(exact_qubo))
    local.__dict__.update(config=local_config, build_qubo=builder, independent_terms=independent_terms)
    source = exact_qubo.solve_exact_qubo
    local.solve_exact_qubo = types.FunctionType(source.__code__, local.__dict__, source.__name__, source.__defaults__)
    local.solve_exact_qubo.__kwdefaults__ = source.__kwdefaults__
    valid_globals = dict(vars(r3_run), exact_qubo=local, config=local_config,
                         build_qubo=builder, independent_terms=independent_terms)
    source = r3_run.score_state
    valid_globals["score_state"] = types.FunctionType(source.__code__, valid_globals, source.__name__, source.__defaults__)
    source = r3_run.solve_valid_qubo
    valid = types.FunctionType(source.__code__, valid_globals, source.__name__, source.__defaults__)
    return local.solve_exact_qubo, valid


def solve_fixed(vms, servers, variant, active, time_limit=600):
    solve, _ = solver_context(variant)
    fixed = {y_var(h): int(h in active) for h in range(len(servers))}
    return solve(vms, servers, center=RUN_CONFIG["center"], time_limit=time_limit,
                 max_iters=RUN_CONFIG["max_iters"], fixed_sample=fixed)


def pass_rule(global_valid, diff_r1, diff_r2, unit):
    diff = diff_r2 if unit == 0.0025 else diff_r1
    return bool(global_valid and diff is not None and math.isfinite(diff) and abs(diff) <= 0.1)


def score(sample, vms, servers, bqm, info, energy):
    h = math.fsum(independent_terms(sample, vms, servers, info, energy).values())
    bqm_h = float(bqm.energy(sample))
    if abs(h - bqm_h) > 1e-9:
        raise AssertionError(("independent/BQM mismatch", h, bqm_h))
    return h


def metrics(sample, vms, servers):
    check = r3_run.state_validation(sample, vms, servers)
    placements = {v.vm_id: [h for h in range(len(servers)) if sample[x_var(v.vm_id, h)]] for v in vms}
    complete = all(len(hs) == 1 for hs in placements.values())
    overload = max((100 * max(0, math.fsum(getattr(v, resource + "_size") * sample[x_var(v.vm_id, h)]
                       for v in vms) - getattr(s, resource + "_limit")) / getattr(s, resource + "_limit")
                    for h, s in enumerate(servers) for resource in ("cpu", "mem")), default=0.)
    cost = None
    if complete:
        assignment = {vm_id: hs[0] for vm_id, hs in placements.items()}
        # Include every ON server, even if empty, using the one energy function.
        idle, _, _ = r3_run.linear_energy_terms(vms, servers)
        used = set(assignment.values())
        cost = linear_cost(assignment, vms, servers) + math.fsum(
            idle[h] * (sample[y_var(h)] - int(h in used)) for h in range(len(servers)))
    return {"GLOBAL_VALID": bool(check["valid"] and complete and overload == 0), "n_unplaced": sum(not hs for hs in placements.values()),
            "n_duplicate": sum(len(hs) > 1 for hs in placements.values()),
            "max_real_overload_pct": overload, "linear_joules": cost, "validation": check}


def references(size, vms, servers, bqm):
    _, assignment = r3_run.saved_sample(config.RESULTS_DIR / size / "assignment_exact.csv", servers, bqm)
    r1 = linear_cost(assignment, vms, servers)
    _, r2, _ = r3_run.historical_r2(size)
    return r1, r2


def certificate(result, vms, servers, bqm, info, energy):
    if result["sample"] is not None:
        h = score(result["sample"], vms, servers, bqm, info, energy)
        assert abs(h - result["bqm_H"]) <= 1e-9
    return {k: result.get(k) for k in ("status", "proven_optimal", "bqm_H", "independent_H",
            "lower_bound", "mip_gap", "iterations", "time_s", "reason")}


def promoted():
    rows = records("A")
    good = []
    for v in VARIANTS:
        if all(any(r["size"] == s and r["variant"] == v["id"] and r["PASS"]
                   and r["status"] == "OPTIMAL" for r in rows) for s in SIZES[:2]):
            good.append(v)
    return sorted(good, key=lambda v: (max(v["lambdas"]), v["lambdas"][0], -v["unit"], v["id"]))[:2]


def part_a(size):
    variants = promoted() if size == "large_v2" else VARIANTS
    if not variants:
        emit("SKIP", part="A", size=size, reason="No variant PASS on both small and medium; large skipped")
    completed = {r["variant"] for r in records("A") if r["size"] == size}
    for variant in variants:
        if variant["id"] in completed:
            continue
        emit("START", part="A", size=size, variant=variant["id"], config=variant)
        vms, servers, bqm, info = problem(size, variant)
        energy, _ = build_qubo_custom(vms, servers, 0, 0, 0, 0, RUN_CONFIG["center"], variant["unit"])
        solve, valid_solve = solver_context(variant)
        result = solve(vms, servers, center=RUN_CONFIG["center"], time_limit=600, max_iters=50)
        valid = valid_solve(vms, servers, time_limit=600, max_iters=50)
        global_cert = certificate(result, vms, servers, bqm, info, energy)
        valid_cert = certificate(valid, vms, servers, bqm, info, energy)
        data = metrics(result["sample"], vms, servers)
        r1, r2 = references(size, vms, servers, bqm)
        cost = data["linear_joules"]
        d1, d2 = (None, None) if cost is None else (100 * (cost - r1) / r1, 100 * (cost - r2) / r2)
        emit("A", size=size, variant=variant["id"], config=variant, **data,
             status=result["status"], valid_status=valid["status"],
             H_min=result["bqm_H"] if result["proven_optimal"] else None,
             H_valid_min=valid["bqm_H"] if valid["proven_optimal"] else None,
             R3=global_cert, R3v=valid_cert, diff_r1_pct=d1, diff_r2_pct=d2, r1_joules=r1, r2_joules=r2,
             PASS=pass_rule(data["GLOBAL_VALID"], d1, d2, variant["unit"]),
             bqm_hash=qubo_fingerprint(bqm)["sha256"], sample=result["sample"])
    rows = [r for r in records("A") if r["size"] == size and r["variant"].startswith("V1-")
            and r["GLOBAL_VALID"] and r["status"] == "OPTIMAL"]
    emit("LAMBDA_STAR", size=size, value=min((r["config"]["lambdas"][0] for r in rows), default=None))
    emit("DONE", part="A", size=size)


def active_controls(servers, size):
    ids = RUN_CONFIG["active_ids"][size]
    active = [h for h, s in enumerate(servers) if s.server_id in ids]
    if len(active) != len(ids):
        raise ValueError("fixed active server IDs absent")
    controls = [("D1", active)]
    if size != "large_v2":
        order = list(dict.fromkeys(s.type_name for s in servers))
        last = max(order.index(servers[h].type_name) for h in active)
        next_type = order[last + 1]
        extras = [h for h, s in enumerate(servers) if s.type_name == next_type and h not in active]
        if len(extras) < 2:
            raise ValueError("next type has fewer than two extra servers")
        controls += [("D1+1", active + extras[:1]), ("D1+2", active + extras[:2])]
    return controls


def sample_worker(pipe, size, variant, active, seed, exact_h):
    try:
        from dwave.samplers import SimulatedAnnealingSampler
        vms, servers, bqm, info = problem(size, variant)
        energy, _ = build_qubo_custom(vms, servers, 0, 0, 0, 0, RUN_CONFIG["center"], variant["unit"])
        reduced = bqm.copy()
        fixed = {y_var(h): int(h in active) for h in range(len(servers))}
        reduced.fix_variables(fixed)
        started = time.perf_counter()
        samples = SimulatedAnnealingSampler().sample(reduced, num_reads=RUN_CONFIG["reads"],
                    num_sweeps=RUN_CONFIG["sweeps"], seed=seed)
        valid_count = optimal_count = total = 0
        min_h = None
        for row in samples.data(fields=["sample", "energy", "num_occurrences"]):
            sample = {**{k: int(v) for k, v in row.sample.items()}, **fixed}
            h = score(sample, vms, servers, bqm, info, energy)
            assert abs(h - row.energy) <= 1e-9
            count = int(row.num_occurrences)
            total += count
            if metrics(sample, vms, servers)["GLOBAL_VALID"]:
                valid_count += count
                min_h = h if min_h is None else min(min_h, h)
                if exact_h is not None and abs(h - exact_h) <= 1e-9:
                    optimal_count += count
        assert total == RUN_CONFIG["reads"]
        pipe.send({"status": "DONE", "valid_read_pct": 100 * valid_count / total,
                   "min_valid_H": min_h, "exact_fixed_y_H": exact_h,
                   "gap_pct": None if min_h is None or exact_h is None else
                       100 * (min_h - exact_h) / max(abs(exact_h), 1e-10),
                   "pure_and_optimal_count": optimal_count, "sampler_wall_s": time.perf_counter() - started})
    except BaseException as exc:
        pipe.send({"status": "ERROR", "error": repr(exc)})
    finally:
        pipe.close()


def sampled_run(size, variant, active, seed, exact_h):
    ctx = mp.get_context("spawn")
    receive, send = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=sample_worker, args=(send, size, variant, active, seed, exact_h))
    started = time.perf_counter()
    proc.start()
    send.close()
    try:
        if receive.poll(RUN_CONFIG["sample_cap"]):
            result = receive.recv()
        else:
            result = {"status": "TIMEOUT", "valid_read_pct": None, "min_valid_H": None,
                      "exact_fixed_y_H": exact_h, "gap_pct": None, "pure_and_optimal_count": None}
    finally:
        if proc.is_alive():
            proc.terminate()
        proc.join(timeout=5)
        receive.close()
    result["wall_s"] = time.perf_counter() - started
    return result


def part_b(size):
    rows = [r for r in records("A") if r["size"] == size and r["PASS"] and r["status"] == "OPTIMAL"]
    best = min(rows, key=lambda r: (abs(r["diff_r2_pct"] if r["config"]["unit"] == 0.0025 else
                                      r["diff_r1_pct"]), max(r["config"]["lambdas"]), r["variant"]), default=None)
    variants = [VARIANTS[0]]
    if best is not None and best["variant"] != "V0":
        variants.append(best["config"])
    for variant in variants:
        vms, servers, bqm, info = problem(size, variant)
        # Verify that the requested D1 IDs match the existing quantized R2 reference.
        ids, _, _ = r3_run.historical_r2(size)
        assert ids == RUN_CONFIG["active_ids"][size], (size, ids)
        for label, active in active_controls(servers, size):
            emit("START", part="B", size=size, variant=variant["id"], control=label,
                 active_ids=[servers[h].server_id for h in active])
            exact = solve_fixed(vms, servers, variant, active)
            energy, _ = build_qubo_custom(vms, servers, 0, 0, 0, 0, RUN_CONFIG["center"], variant["unit"])
            cert = certificate(exact, vms, servers, bqm, info, energy)
            exact_h = exact["bqm_H"] if exact["proven_optimal"] else None
            emit("FIXED", size=size, variant=variant["id"], control=label, certificate=cert,
                 validation=metrics(exact["sample"], vms, servers), sample=exact["sample"])
            for seed in RUN_CONFIG["seeds"][size]:
                result = sampled_run(size, variant, active, seed, exact_h)
                emit("B", size=size, variant=variant["id"], control=label, seed=seed,
                     seed_label="1-3 seeds", active_ids=[servers[h].server_id for h in active],
                     exact_status=exact["status"], **result)
    emit("DONE", part="B", size=size)


def protected_snapshot():
    roots = ["results/phase2", "results/runs", "results/experiment_log.csv"]
    return {r: {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted((ROOT / r).rglob("*")) if p.is_file()}
            if (ROOT / r).is_dir() else {r: hashlib.sha256((ROOT / r).read_bytes()).hexdigest()} for r in roots}


def fmt(value):
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "Y" if value else "N"
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


def finish():
    text = RAW.read_text(encoding="utf-8")
    before = json.loads(next(line[len("BEFORE_JSON="):] for line in text.splitlines() if line.startswith("BEFORE_JSON=")))
    after = protected_snapshot()
    emit("AFTER", manifest=after, identical=before == after,
         counts={k: len(v) for k, v in after.items()})
    diffs = []
    for command in (["git", "--no-pager", "diff", "--exit-code"],
                    ["git", "--no-pager", "diff", "--cached", "--exit-code"]):
        result = subprocess.run(command, capture_output=True, text=True, cwd=ROOT)
        print("COMMAND:", " ".join(command), "\n" + result.stdout + result.stderr, "EXIT=", result.returncode, flush=True)
        diffs.append(result.returncode)
    tests = re.findall(r"=+ ([^\n]*\bpassed[^\n]*) =+", text)
    rows_a, rows_b = records("A"), records("B")
    stars = {r["size"]: r["value"] for r in records("LAMBDA_STAR")}
    eq_count = sum(f"EQUIVALENCE {size} PASS" in text for size in SIZES)
    baseline = re.search(r"BASELINE OPTIMAL ([\d.e+-]+) OPTIMAL ([\d.e+-]+)", text)
    baseline_ok = baseline is not None and abs(float(baseline[1]) - 0.3106438073) <= 1e-9 and abs(float(baseline[2]) - 0.4304366536) <= 1e-9
    test_runs = re.findall(r"COMMAND: [^\n]* -m pytest [^\n]*\n(.*?)EXIT=(\d+)", text, re.S)
    tests_ok = len(test_runs) >= 3 and all(code == "0" and re.search(r"\bpassed in\b", output) for output, code in test_runs)
    lines = ["Branch: `step3-r3c-sweep`; new files: `optimizer/diagnostics/r3c_build.py`, `optimizer/diagnostics/r3c_run.py`, `tests/test_r3c_run.py`; overwritten own `docs/reports/stepR3c_{raw_output.txt,report.md}`.",
             "Commands: `.venv-gq/Scripts/python.exe -B` for equivalence/baseline; `-m pytest tests/test_exact_qubo.py -v -p no:cacheprovider`; `-u optimizer/diagnostics/r3c_run.py --part A/B --size small_v2/medium_v2/large_v2` (six commands); focused/full pytest; `--part finish`; git status/diffs; SHA256 manifests (raw). "
             "Builder: minimal unbalanced assembly copied; existing lower-level builder lacks explicit lambdas/unit; existing energy/penalty functions reused. Dependencies added: none. Sampling: 1-3 seeds; pure-and-optimal = real-valid singleton/on-host reads within 1e-9 of proven fixed-y H.",
             "| Part A size / variant | R3 / R3v | H_min / H_valid_min | GLOBAL_VALID | Unplaced | Max overload % | Linear J | Diff R1 / R2 % | PASS |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows_a:
        lines.append("| " + " | ".join([r["size"] + "/" + r["variant"], r["status"] + "/" + r["valid_status"],
            fmt(r["H_min"]) + "/" + fmt(r["H_valid_min"]), fmt(r["GLOBAL_VALID"]), fmt(r["n_unplaced"]),
            fmt(r["max_real_overload_pct"]), fmt(r["linear_joules"]), fmt(r["diff_r1_pct"]) + "/" + fmt(r["diff_r2_pct"]), fmt(r["PASS"])]) + " |")
    if not any(r["size"] == "large_v2" for r in rows_a):
        lines.append("| large_v2 | SKIPPED | — | — | — | — | — | — | — |")
    lines[1] += " lambda_star: " + "; ".join(s + "=" + fmt(stars.get(s)) for s in SIZES) + ". V0=L0.05/u0.0025; V1=equal L; V2=assign only; V3=unit only; c=0.02. UNPROVEN minima are —; incumbents/certificates are in raw."
    lines += ["| Part B size / variant / set (1-3 seeds) | seed: valid % / min valid H / exact H / gap % / pure-optimal reads / wall s (status) |",
              "|---|---|"]
    groups = {}
    for r in rows_b:
        groups.setdefault((r["size"], r["variant"]), []).append(r)
    for key, group in groups.items():
        values = [r["control"] + str(r["active_ids"]) + " seed " + str(r["seed"]) + ": " + "/".join(fmt(r.get(k)) for k in ("valid_read_pct", "min_valid_H", "exact_fixed_y_H", "gap_pct", "pure_and_optimal_count", "wall_s")) + " (" + r["status"] + ")" for r in group]
        lines.append("| " + "/".join(key) + " | " + "; ".join(values) + " |")
    lines += ["| Gate | Expected | Actual | Pass/fail |", "|---|---|---|---|",
              "| Equivalence / baseline | Exact 3 sizes; H=0.3106438073/0.4304366536 ±1e-9 | " + str(eq_count) + "/3; " + (baseline[1] + "/" + baseline[2] if baseline else "MISSING") + " | " + ("PASS" if eq_count == 3 and baseline_ok else "FAIL") + " |",
              "| R3a / focused / full tests | All pass | " + "; ".join(tests) + " | " + ("PASS" if tests_ok else "FAIL") + " |" ,
              "| Protected SHA256/counts; git diff/cached | Identical 105/24/1; 0/0 | " + str(before == after) + "; " + "/".join(map(str, diffs)) + " | " + ("PASS" if before == after and diffs == [0, 0] else "FAIL") + " |"]
    issues = ["fixed-y UNPROVEN" for r in rows_b if r["exact_status"] != "OPTIMAL"]
    issues += [r["status"] for r in rows_b if r["status"] != "DONE"]
    issues += ["Part A UNPROVEN" for r in rows_a if r["status"] != "OPTIMAL" or r["valid_status"] != "OPTIMAL"]
    lines.append("Open issues: " + (", ".join(sorted(set(issues))) if issues else "none") + ".")
    if len(lines) > 40:
        raise RuntimeError(("report exceeds 40 lines", len(lines)))
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    emit("FINISH", report_lines=len(lines), protected_identical=before == after, diffs=diffs)
    assert before == after and diffs == [0, 0] and eq_count == 3 and baseline_ok and tests_ok


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", choices=["A", "B", "finish"], required=True)
    parser.add_argument("--size", choices=SIZES)
    args = parser.parse_args()
    if args.part != "finish" and args.size is None:
        parser.error("--size is required")
    print("RUN_CONFIG=" + json.dumps(RUN_CONFIG, sort_keys=True), flush=True)
    if args.part == "A":
        part_a(args.size)
    elif args.part == "B":
        part_b(args.size)
    else:
        finish()


if __name__ == "__main__":
    main()
