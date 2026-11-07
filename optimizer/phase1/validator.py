"""
Turn an annealer sample (0/1 switches) into a placement and check every rule:
  1. every VM on exactly one server
  2. CPU and memory of each server within the safe limit (checked with real, un-rounded sizes)
  3. no VM on a server whose ON switch is 0
If a sample breaks a rule, `repair` fixes it with a small first-fit step,
and that is always reported, so pure annealer results stay distinguishable.
"""
from .qubo_builder import x_var, y_var


def decode(sample, vms, server):
    """Raw decoding. Returns (assignment or None, list of broken rules)."""
    broken = []
    assignment = {}
    for v in vms:
        chosen = [h for h in range(server.count) if sample[x_var(v.vm_id, h)] == 1]
        if len(chosen) != 1:
            broken.append(f"VM {v.vm_id} on {len(chosen)} servers")
        else:
            assignment[v.vm_id] = chosen[0]
            if sample[y_var(chosen[0])] == 0:
                broken.append(f"VM {v.vm_id} on switched-off server {chosen[0]}")
    if len(assignment) == len(vms):
        broken += capacity_problems(assignment, vms, server)
    return (assignment if not broken else None), broken


def capacity_problems(assignment, vms, server):
    cpu_limit = server.safe_utilization * server.cpu_capacity
    mem_limit = server.safe_utilization * server.mem_capacity
    problems = []
    for h in range(server.count):
        on_h = [v for v in vms if assignment.get(v.vm_id) == h]
        cpu = sum(v.cpu_size for v in on_h)
        mem = sum(v.mem_size for v in on_h)
        if cpu > cpu_limit + 1e-12:
            problems.append(f"server {h} CPU {cpu:.4f} > {cpu_limit:.4f}")
        if mem > mem_limit + 1e-12:
            problems.append(f"server {h} memory {mem:.4f} > {mem_limit:.4f}")
    return problems


def repair(sample, vms, server):
    """
    Keep every VM the annealer placed validly, then first-fit the rest.
    Returns (assignment or None, number of VMs kept from the annealer, number placed by repair).
    """
    cpu_limit = server.safe_utilization * server.cpu_capacity
    mem_limit = server.safe_utilization * server.mem_capacity
    by_id = {v.vm_id: v for v in vms}
    assignment = {}

    # a VM the annealer put on exactly one ON server is kept there
    for v in vms:
        chosen = [h for h in range(server.count) if sample[x_var(v.vm_id, h)] == 1]
        if len(chosen) == 1 and sample[y_var(chosen[0])] == 1:
            assignment[v.vm_id] = chosen[0]

    # empty overloaded servers, biggest VMs first, until they fit
    for h in range(server.count):
        on_h = sorted([i for i, s in assignment.items() if s == h],
                      key=lambda i: by_id[i].cpu_size, reverse=True)
        while on_h and (sum(by_id[i].cpu_size for i in on_h) > cpu_limit + 1e-12
                        or sum(by_id[i].mem_size for i in on_h) > mem_limit + 1e-12):
            del assignment[on_h.pop(0)]
    kept = len(assignment)

    # first-fit decreasing for the remaining VMs, preferring servers that are already ON
    def used(h, attr):
        return sum(getattr(by_id[i], attr) for i, s in assignment.items() if s == h)

    unplaced = sorted([v for v in vms if v.vm_id not in assignment], key=lambda v: v.cpu_size, reverse=True)
    for v in unplaced:
        active = sorted(set(assignment.values()))
        candidates = active + [h for h in range(server.count) if h not in active]
        for h in candidates:
            if (used(h, "cpu_size") + v.cpu_size <= cpu_limit + 1e-12
                    and used(h, "mem_size") + v.mem_size <= mem_limit + 1e-12):
                assignment[v.vm_id] = h
                break
        else:
            return None, kept, 0  # not enough servers at all
    return assignment, kept, len(vms) - kept
