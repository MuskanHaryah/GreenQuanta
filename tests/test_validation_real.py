"""Read-only smoke check of saved small_v2 placements using real VM/Server classes."""
import csv
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "optimizer"))

from phase1.trace_loader import load_trace
from phase1.vm_builder import build_vms
from phase2 import config
from phase2.catalog import build_servers
from phase2.pipeline import prepare
from phase2.pipeline_v2 import auto_mix
from phase2.validation import validate_allocation


def test_saved_small_v2_assignments():
    trace = config.TRACE_DIR / config.SCENARIOS["small_v2"]["trace"]
    sizing_vms, _ = build_vms(
        load_trace(trace), config.INSTANCES_PER_VM, config.SLA_MAX_VIOLATION_PCT,
    )
    servers = build_servers(auto_mix(sizing_vms))
    vms, _, _, _ = prepare(trace, servers)
    host_indices = {server.server_id: index for index, server in enumerate(servers)}

    for filename in ("assignment_exact.csv", "assignment_qa.csv"):
        path = config.RESULTS_DIR / "small_v2" / filename
        with path.open(newline="", encoding="utf-8-sig") as stream:
            assignment = {int(row["vm_id"]): host_indices[int(row["host_id"])]
                          for row in csv.DictReader(stream)}
        real = validate_allocation(vms, assignment, servers)
        quantized = validate_allocation(vms, assignment, servers, quantized_unit=0.0025)
        print(f"{filename} real: {real}")
        print(f"{filename} quantized: {quantized}")
        assert real["valid"], real
