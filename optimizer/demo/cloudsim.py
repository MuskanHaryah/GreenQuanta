"""Guide 1 step 1.3: CloudSim Plus replay of any placement into results/mids (never results/phase2).

replay(size, label, vms, servers, assignment, out_root) writes the inputs with phase2.pipeline.write_cloudsim_inputs
to <out_root>/<size>/<label>/cloudsim_input/, runs Phase2Simulation with explicit input/output folders, reads the two
CSVs and cross-checks them against phase2.energy_model.evaluate_allocation with report_phase2.compare.
Maven needs a valid JAVA_HOME (JDK >= 17).
"""
import os
from pathlib import Path
import shutil
import subprocess

import pandas as pd

from phase2 import config
from phase2.energy_model import evaluate_allocation
from phase2.pipeline import write_cloudsim_inputs
from report_phase2 import compare

SIM_DIR = config.ROOT / "simulation"
OUT_ROOT = config.ROOT / "results" / "mids"
TOLERANCE_PCT = 1.0          # historical Python vs CloudSim cross-check bar


def mvn(main_class, *args):
    """Run a simulation class with mvn exec:java (cwd simulation/); returns stdout. Paths go relative to simulation/
    because exec.args splits on whitespace and the repo path has spaces."""
    exe = shutil.which("mvn")
    if exe is None:
        raise RuntimeError("mvn not found on PATH")
    java_home = os.environ.get("JAVA_HOME")
    if java_home and not Path(java_home).is_dir():
        raise RuntimeError(f"JAVA_HOME points to a missing folder: {java_home}")
    rel = [os.path.relpath(a, SIM_DIR) if isinstance(a, Path) else str(a) for a in args]
    if any(any(c.isspace() for c in a) for a in rel):
        raise ValueError(f"exec.args cannot hold whitespace: {rel}")
    proc = subprocess.run([exe, "-q", "compile", "exec:java", f"-Dsim.main={main_class}",
                           "-Dexec.args=" + " ".join(rel)], cwd=SIM_DIR, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{main_class} {rel} failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}")
    return proc.stdout


def replay(size, label, vms, servers, assignment, out_root=OUT_ROOT):
    """CloudSim replay of `assignment` {vm_id: host index} and its cross-check against the Python energy model."""
    out_dir = Path(out_root) / size / label
    cs_in = out_dir / "cloudsim_input"
    write_cloudsim_inputs(cs_in, servers, vms, assignment)
    mvn("greenquanta.Phase2Simulation", size, cs_in, out_dir)
    summary = pd.read_csv(out_dir / "cloudsim_summary.csv").set_index("metric")["value"]
    hosts = pd.read_csv(out_dir / "cloudsim_energy_hosts.csv")
    energy = evaluate_allocation(vms, assignment, servers)
    rows, worst = compare(energy, summary)
    it_diff = next(d for name, _, _, d in rows if name.startswith("IT energy"))
    return {"it_energy_kwh": float(summary["it_energy_kwh"]),
            "facility_energy_kwh": float(summary["facility_energy_kwh"]), "co2_kg": float(summary["co2_kg"]),
            "active_hosts": int(summary["active_hosts"]), "diff_it_pct_vs_python": it_diff,
            "worst_diff_pct": worst, "ok": bool(worst <= TOLERANCE_PCT), "rows": rows,
            "per_host_kwh": [{"host_id": int(h["host_id"]), "python": p["energy_kwh"], "cloudsim": float(h["energy_kwh"])}
                             for h, p in zip(hosts.to_dict("records"), energy["per_host"]) if p["active"]]}
