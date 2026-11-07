"""User inputs for Phase 1 (one homogeneous server type + SLA), with validation."""
from dataclasses import dataclass, asdict

from . import config


@dataclass(frozen=True)
class ServerConfig:
    """One server type. Phase 1 assumes every server is identical."""
    count: int
    cpu_capacity: float      # NCU
    mem_capacity: float      # NMU
    idle_power_w: float
    peak_power_w: float
    safe_utilization: float  # 0..1

    def validate(self):
        if self.count < 1:
            raise ValueError("Number of servers must be at least 1.")
        if self.cpu_capacity <= 0 or self.mem_capacity <= 0:
            raise ValueError("Server CPU and memory capacity must be positive.")
        if not 0 <= self.idle_power_w <= self.peak_power_w:
            raise ValueError("Need 0 <= idle power <= peak power.")
        if not 0 < self.safe_utilization <= 1:
            raise ValueError("Safe utilization must be in (0, 1].")


@dataclass(frozen=True)
class Phase1Inputs:
    server: ServerConfig
    sla_max_violation_pct: float   # user SLA limit, % of VM time that may be throttled
    instances_per_vm: int
    trace_csv: str

    def validate(self):
        self.server.validate()
        if not 0 <= self.sla_max_violation_pct < 100:
            raise ValueError("SLA limit must be between 0 and 100 (percent).")
        if self.instances_per_vm < 1:
            raise ValueError("instances_per_vm must be at least 1.")

    def to_dict(self):
        return asdict(self)


def default_inputs() -> Phase1Inputs:
    """The hardcoded development inputs from config.py."""
    return Phase1Inputs(
        server=ServerConfig(
            count=config.NUM_SERVERS,
            cpu_capacity=config.SERVER_CPU_CAPACITY,
            mem_capacity=config.SERVER_MEM_CAPACITY,
            idle_power_w=config.SERVER_IDLE_POWER_W,
            peak_power_w=config.SERVER_PEAK_POWER_W,
            safe_utilization=config.SAFE_UTILIZATION,
        ),
        sla_max_violation_pct=config.SLA_MAX_VIOLATION_PCT,
        instances_per_vm=config.INSTANCES_PER_VM,
        trace_csv=str(config.TRACE_CSV),
    )
