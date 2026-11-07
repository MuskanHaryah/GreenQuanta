"""
Server catalog: Google machine types (capacity) + SPECpower curves (watts).

Power model used for EVALUATION (Python and CloudSim):
    P(u) = linear interpolation between the 11 SPECpower points (idle, 10%, ..., 100%)
Power model used inside the QUBO (QUBO terms must be linear/quadratic):
    P(u) ~ fit_idle + fit_slope * u, a least-squares line through the 10%..100% points
    (a server that is ON in our placement always runs some load, so the idle point is left out).
"""
import json
from dataclasses import dataclass, field

import numpy as np

from . import config

LOAD_POINTS = np.linspace(0.0, 1.0, 11)


@dataclass
class Server:
    server_id: int
    type_name: str
    cpu_capacity: float          # NCU
    mem_capacity: float          # NMU
    power_w: list                # 11 SPEC points
    safe_utilization: float
    spec_server: str = ""
    fit_idle_w: float = field(init=False)
    fit_slope_w: float = field(init=False)

    def __post_init__(self):
        if len(self.power_w) != 11:
            raise ValueError(f"Type {self.type_name}: power_w needs 11 values (idle, 10%, ..., 100%).")
        if not 0 < self.safe_utilization <= 1:
            raise ValueError(f"Type {self.type_name}: safe_utilization must be in (0, 1].")
        slope, intercept = np.polyfit(LOAD_POINTS[1:], np.array(self.power_w[1:], dtype=float), 1)
        self.fit_idle_w = float(intercept)
        self.fit_slope_w = float(slope)

    def power(self, util):
        """Real SPEC curve, interpolated. util between 0 and 1 (array or number)."""
        return np.interp(np.clip(util, 0.0, 1.0), LOAD_POINTS, self.power_w)

    @property
    def cpu_limit(self):
        return self.safe_utilization * self.cpu_capacity

    @property
    def mem_limit(self):
        return self.safe_utilization * self.mem_capacity


def load_catalog(path=None):
    with open(path or config.CATALOG_JSON) as f:
        return json.load(f)["types"]


def build_servers(mix, overrides=None, catalog=None):
    """
    mix:       {"A": 2, "B": 1, ...}  number of servers of each type
    overrides: {"A": {"safe_utilization": 0.8}, ...}  optional user changes per type
    Servers are numbered type by type (all A, then all B, ...).
    """
    catalog = catalog or load_catalog()
    overrides = overrides or {}
    servers = []
    for type_name in sorted(mix):
        if type_name not in catalog:
            raise ValueError(f"Unknown server type '{type_name}'. Known: {sorted(catalog)}")
        spec = {**catalog[type_name], **overrides.get(type_name, {})}
        for _ in range(int(mix[type_name])):
            servers.append(Server(
                server_id=len(servers),
                type_name=type_name,
                cpu_capacity=float(spec["google_cpu_ncu"]),
                mem_capacity=float(spec["google_mem_nmu"]),
                power_w=[float(p) for p in spec["power_w"]],
                safe_utilization=float(spec["safe_utilization"]),
                spec_server=spec.get("spec_server", ""),
            ))
    if not servers:
        raise ValueError("No servers given.")
    return servers


def parse_mix(text):
    """'A=2,B=1,C=2,D=1' -> {'A': 2, 'B': 1, 'C': 2, 'D': 1}"""
    mix = {}
    for part in text.split(","):
        name, count = part.split("=")
        mix[name.strip().upper()] = int(count)
    return mix


def catalog_table(servers):
    rows, seen = [], set()
    for s in servers:
        if s.type_name in seen:
            continue
        seen.add(s.type_name)
        rows.append({
            "type": s.type_name,
            "count": sum(1 for x in servers if x.type_name == s.type_name),
            "cpu_ncu": s.cpu_capacity,
            "mem_nmu": s.mem_capacity,
            "idle_w": s.power_w[0],
            "peak_w": s.power_w[-1],
            "fit_idle_w": round(s.fit_idle_w, 1),
            "fit_slope_w": round(s.fit_slope_w, 1),
            "safe_util": s.safe_utilization,
            "spec_server": s.spec_server,
        })
    return rows
