"""Build the default v2 workload and QUBO without invoking any runner."""
from phase1.trace_loader import load_trace
from phase1.vm_builder import build_vms
from phase2 import config, qubo_builder_v2
from phase2.catalog import build_servers
from phase2.pipeline import prepare
from phase2.pipeline_v2 import auto_mix


ENCODING = "unbalanced"
LAMBDA_CONFIG = "L0.05"
CENTER = 0.02


def build_problem(size, lambda_config=LAMBDA_CONFIG, center=CENTER):
    """Return (vms, servers, bqm, info), using default resource units.

    Match managed-run setup and test_validation_real: size once for auto_mix,
    construct the catalog servers, then call prepare for the actual workload.
    """
    scenario = config.SCENARIOS[size]
    trace = config.TRACE_DIR / scenario["trace"]
    sizing_vms, _ = build_vms(
        load_trace(trace), config.INSTANCES_PER_VM, config.SLA_MAX_VIOLATION_PCT,
    )
    mix = auto_mix(sizing_vms) if scenario["servers"] == "auto" else scenario["servers"]
    servers = build_servers(mix)
    vms, _, _, _ = prepare(trace, servers)
    bqm, info = qubo_builder_v2.build_qubo(
        vms, servers, lambda_config=lambda_config, encoding=ENCODING, center=center,
    )
    return vms, servers, bqm, info
