"""
Turn real Google trace instances into VMs.

Rule (no synthetic data):
  1. A VM belongs to ONE job (collection), like a VM rented by one customer.
  2. Sibling instances of that job (sorted by instance_index) are packed
     INSTANCES_PER_VM at a time into one VM.
  3. The VM's CPU demand in every 5-minute window = sum of its instances' real usage.
  4. VM CPU size  = percentile p of that 24 h demand, where p = 100 - SLA limit
                    (SLA 5%  -> P95 size, SLA 0% -> peak size).
     VM memory size = maximum memory used in the day (memory cannot be throttled).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config


@dataclass
class VirtualMachine:
    vm_id: int
    collection_id: int
    instance_indexes: list
    cpu_demand: np.ndarray      # NCU, one value per 5-minute window (288 values)
    mem_demand: np.ndarray      # NMU, one value per window
    cpu_size: float             # NCU reserved for the VM
    mem_size: float             # NMU reserved for the VM

    @property
    def cpu_served(self) -> np.ndarray:
        """CPU the VM actually gets: demand, capped at its size."""
        return np.minimum(self.cpu_demand, self.cpu_size)

    @property
    def cpu_mean_served(self) -> float:
        return float(self.cpu_served.mean())


def _instance_series(df_inst: pd.DataFrame, column: str) -> np.ndarray:
    """288-value series for one instance; short gaps are filled from neighbours."""
    s = df_inst.set_index("window_idx")[column].reindex(range(config.WINDOWS_PER_DAY))
    return s.ffill().bfill().fillna(0.0).to_numpy(dtype=float)


def build_vms(trace: pd.DataFrame, instances_per_vm: int, sla_max_violation_pct: float):
    percentile = 100.0 - sla_max_violation_pct
    vms = []
    for collection_id, df_job in trace.groupby("collection_id", sort=True):
        indexes = sorted(df_job["instance_index"].unique())
        for start in range(0, len(indexes), instances_per_vm):
            group = indexes[start:start + instances_per_vm]
            cpu = np.zeros(config.WINDOWS_PER_DAY)
            mem = np.zeros(config.WINDOWS_PER_DAY)
            for idx in group:
                df_inst = df_job[df_job["instance_index"] == idx]
                cpu += _instance_series(df_inst, "avg_cpu")
                mem += _instance_series(df_inst, "avg_mem")

            # "higher" picks a real observed value, so throttled time never exceeds the SLA limit
            cpu_size = float(np.percentile(cpu, percentile, method="higher"))
            mem_size = float(mem.max())
            if cpu_size <= 0:
                continue  # an idle VM all day adds nothing to place
            vms.append(VirtualMachine(
                vm_id=len(vms),
                collection_id=int(collection_id),
                instance_indexes=[int(i) for i in group],
                cpu_demand=cpu,
                mem_demand=mem,
                cpu_size=cpu_size,
                mem_size=mem_size,
            ))
    if not vms:
        raise ValueError("No VM could be built from the trace.")
    return vms, percentile


def vms_table(vms) -> pd.DataFrame:
    return pd.DataFrame([{
        "vm_id": v.vm_id,
        "collection_id": v.collection_id,
        "instance_indexes": " ".join(map(str, v.instance_indexes)),
        "cpu_size_ncu": round(v.cpu_size, 6),
        "mem_size_nmu": round(v.mem_size, 6),
        "cpu_mean_demand_ncu": round(float(v.cpu_demand.mean()), 6),
        "cpu_peak_demand_ncu": round(float(v.cpu_demand.max()), 6),
    } for v in vms])
