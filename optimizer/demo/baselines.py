"""Guide 1 step 1.4: CloudSim Plus standard placement baselines, checked by the shared validator.

run_baseline(size, policy) writes the CloudSim inputs (dummy assignment, ignored) plus the PE host model, runs
greenquanta.BaselinePlacement, reads the placement back and validates it (real and quantized). A complete placement is
scored with evaluate_allocation and linear_cost and replayed in CloudSim. Unplaced or invalid placements are reported
as such and never repaired.
"""
import math
from pathlib import Path
import time

import pandas as pd

from demo.cloudsim import OUT_ROOT, mvn, replay
from diagnostics.common import build_problem
from phase2 import config
from phase2.energy_model import evaluate_allocation
from phase2.exact_solver import linear_cost
from phase2.pipeline import write_cloudsim_inputs
from phase2.validation import validate_allocation

POLICIES = ["firstfit", "bestfit", "worstfit", "roundrobin", "ffd", "bfd", "pabfd"]
UNIT = config.UNIT_NCU       # one CloudSim PE = one quantization unit (0.0025 NCU)
MB_PER_NMU = 1_000_000       # Phase2Simulation's memory scale


def write_pe_model(cs_in, servers, vms, unit=UNIT):
    """Host PEs = floor(cpu_limit / unit), VM PEs = max(1, ceil(cpu_size / unit)) (the quantized model's rounding);
    RAM = safe memory limit in MB."""
    pd.DataFrame({"host_id": [s.server_id for s in servers],
                  "pes": [int(math.floor(s.cpu_limit / unit + 1e-9)) for s in servers],
                  "ram_mb": [int(math.floor(s.mem_limit * MB_PER_NMU)) for s in servers]}) \
        .to_csv(cs_in / "baseline_hosts.csv", index=False)
    pd.DataFrame({"vm_id": [v.vm_id for v in vms],
                  "pes": [max(1, math.ceil(v.cpu_size / unit - 1e-9)) for v in vms],
                  "ram_mb": [int(math.ceil(v.mem_size * MB_PER_NMU)) for v in vms]}) \
        .to_csv(cs_in / "baseline_vms.csv", index=False)


def run_baseline(size, policy, vms=None, servers=None, out_root=OUT_ROOT):
    started = time.perf_counter()
    if vms is None:
        vms, servers, _, _ = build_problem(size)
    out_dir = Path(out_root) / size / f"cs_{policy}"
    base_in, placed_dir = out_dir / "baseline_input", out_dir / "placement"
    write_cloudsim_inputs(base_in, servers, vms, {v.vm_id: 0 for v in vms})
    write_pe_model(base_in, servers, vms)
    mvn("greenquanta.BaselinePlacement", base_in, placed_dir, policy)
    rows = pd.read_csv(placed_dir / "vms_assignment.csv")
    assignment = {int(v): int(h) for v, h in zip(rows["vm_id"], rows["host_id"])}
    unplaced = pd.read_csv(placed_dir / "failed_vms.csv")["vm_id"].astype(int).tolist()
    real = validate_allocation(vms, assignment, servers)
    quant = validate_allocation(vms, assignment, servers, quantized_unit=UNIT)
    result = {"policy": policy, "found": not unplaced, "unplaced_vms": len(unplaced), "assignment": assignment,
              "valid_real": real["valid"], "valid_quantized": quant["valid"],
              "violations": real["violations"] + quant["violations"],
              "submission_order": pd.read_csv(placed_dir / "submitted_order.csv")["vm_id"].astype(int).tolist(),
              "energy": None, "linear_joules": None, "cloudsim": None}
    if not unplaced:
        result["energy"] = evaluate_allocation(vms, assignment, servers)
        result["linear_joules"] = linear_cost(assignment, vms, servers)
        result["cloudsim"] = replay(size, f"cs_{policy}", vms, servers, assignment, out_root)
    result["runtime_s"] = time.perf_counter() - started
    return result
