"""The /api blueprint (contract in docs/plans/STEP4_PLAN.md). Cached data is read from results/mids/<size>/results.json
on every request; VM sizes for the placement view come from diagnostics.common.build_problem."""
from functools import lru_cache
import json
import math

from flask import Blueprint, current_app, jsonify, request

from backend import ROOT, gas_trace
from demo.build_results import META
from diagnostics.common import build_problem
from diagnostics.s1_end_to_end import RUN_CONFIG
from phase2.manifest import git_info

bp = Blueprint("api", __name__)
SIZES = RUN_CONFIG["sizes"]
CAVEATS = [
    "Validity of a method is valid_real: the placement fits the real server capacities (shared validator).",
    "valid_quantized only says whether the placement also fits the quantum model's rounding grid (unit 0.0025 NCU); "
    "the baselines and R1 work at real sizes, so a failed quantized check is expected for them and is not an error.",
    "The baselines' CloudSim host model rounds CPU to that grid and keeps memory at its real size, the generous choice "
    "for the baselines.",
    "Methods are ranked by linear joules (diff_r1_pct), the optimised objective. it_energy_kwh comes from the SPEC power "
    "curve, on which the quantum placement can score below R1 (small, medium) although R1 is optimal for the linear "
    "objective; this is not a win over the optimum.",
    "The cached quantum and control rows come from one Stage-2 seed (pipe_stage2_seed 0). The multi-seed comparison "
    "(Step 2D.4: held-out seeds 0-2, budgets B and 4B, medium and large) found classical SA better than SQA in 4 of 4 "
    "cells, for every seed.",
    "A live run uses the same config seeds as the cached result, so it must reproduce it; matches_cached reports this.",
]


class ApiError(Exception):
    def __init__(self, status, code, message):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


@bp.errorhandler(ApiError)
def api_error(e):
    return jsonify(error={"code": e.code, "message": e.message}), e.status


def check_size(size):
    if size not in SIZES:
        raise ApiError(404, "unknown_size", f"unknown size {size!r}; one of {SIZES}")


def results_path(size):
    return current_app.config["RESULTS_ROOT"] / size / "results.json"


def load(size):
    check_size(size)
    path = results_path(size)
    if not path.is_file():
        raise ApiError(404, "results_missing", f"no cached results for {size}; build them with: "
                       f".venv-gq/Scripts/python.exe -B optimizer/demo/build_results.py --size {size}")
    return json.loads(path.read_text(encoding="utf-8"))


def method(result, method_id):
    if method_id not in result["methods"]:
        raise ApiError(404, "unknown_method", f"unknown method {method_id!r}; one of {list(result['methods'])}")
    return result["methods"][method_id]


@lru_cache(maxsize=None)
def problem(size):
    vms, servers, _, _ = build_problem(size)
    return vms, servers


@bp.get("/health")
def health():
    return jsonify(status="ok", git_commit=git_info(ROOT)["code_revision"],
                   results_available={s: results_path(s).is_file() for s in SIZES})


@bp.get("/sizes")
def sizes():
    rows = []
    for size in SIZES:
        available = results_path(size).is_file()
        if available:
            p = load(size)["problem"]
            vms, m = p["vms"], p["m"]
        else:
            vms, servers = problem(size)
            vms, m = len(vms), len(servers)
        rows.append({"id": size, "label": size.split("_")[0].title(), "vms": vms, "servers": m, "m": m,
                     "available": available})
    return jsonify(rows)


@bp.get("/results/<size>")
def results(size):
    result = load(size)
    for r in result["methods"].values():
        r.pop("assignment")
    return jsonify(result)


@bp.get("/results/<size>/placement/<method_id>")
def placement(size, method_id):
    result = load(size)
    assignment = {int(v): h for v, h in method(result, method_id)["assignment"].items()}
    vms, _ = problem(size)
    hosts = []
    for s in result["problem"]["servers"]:
        placed = [v for v in vms if assignment.get(v.vm_id) == s["id"]]
        hosts.append({"host_id": s["id"], "type": s["type"], "active": bool(placed),
                      "cpu_limit": s["cpu_limit"], "mem_limit": s["mem_limit"],
                      "cpu_load": math.fsum(v.cpu_size for v in placed),
                      "mem_load": math.fsum(v.mem_size for v in placed),
                      "vms": [{"vm_id": v.vm_id, "cpu_size": v.cpu_size, "mem_size": v.mem_size} for v in placed]})
    return jsonify(method_id=method_id, hosts=hosts)


@bp.get("/results/<size>/compare")
def compare(size):
    rows = [{"method_id": k, "label": r["label"], "kind": r["kind"], "valid_real": r["valid_real"],
             "valid_quantized": r["valid_quantized"], "active_servers": len(r["active_servers"]),
             "diff_r1_pct": r["diff_r1_pct"], "diff_r2_pct": r["diff_r2_pct"],
             "gap_to_h_min_pct": r["gap_to_h_min_pct"],
             "it_energy_kwh": r["energy"] and r["energy"]["it_energy_kwh"],
             "co2_kg": r["energy"] and r["energy"]["co2_kg"],
             "cloudsim_ok": r["cloudsim"] and r["cloudsim"]["ok"]}
            for k, r in load(size)["methods"].items()]
    rows.sort(key=lambda r: (r["diff_r1_pct"] is None, r["diff_r1_pct"] or 0.0, r["method_id"]))
    return jsonify(rows)


@bp.get("/claims")
def claims():
    """Honest claims: the notes of one size's results.json (?size=, default the largest size) plus the fixed caveats and
    the method descriptions and references of build_results.META."""
    size = request.args.get("size", SIZES[-1])
    return jsonify(size=size, notes=load(size)["notes"], caveats=CAVEATS,
                   methods=[{"method_id": k, "label": label, "kind": kind, "algorithm": algorithm, "reference": ref}
                            for k, (label, kind, algorithm, ref) in META.items()])


@bp.get("/gas-trace/<size>")
def gas_trace_view(size):
    check_size(size)
    seeds = RUN_CONFIG["pipe_gas_seeds"]
    seed = request.args.get("seed", str(seeds[0]))
    if not (seed.lstrip("-").isdigit() and int(seed) in seeds):
        raise ApiError(400, "bad_seed", f"seed must be one of the pipeline's GAS seeds {seeds}, got {seed!r}")
    return jsonify(gas_trace.trace(size, int(seed)))
