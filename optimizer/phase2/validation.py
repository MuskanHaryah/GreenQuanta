"""Allocation checks independent of optimization, energy evaluation, and file I/O."""
import math
from numbers import Integral


def validate_allocation(vms, assignment, servers, quantized_unit=None) -> dict:
    """Check real reservations and, optionally, conservative integer capacities.

    ``assignment`` maps VM IDs to zero-based indices into ``servers``. Host
    on/off state is derived from assignment, so "VM on an off host" has no
    separate meaning and is not checked. Invalid VM resources are reported
    even for unassigned VMs and excluded from that resource's load totals.

    Quantized checks are separate from real checks: sum ceil(size / unit)
    is compared with floor(limit / unit), without the real-resource tolerance.
    A supplied unit must be finite and positive; otherwise ValueError is raised.
    """
    if quantized_unit is not None:
        if not math.isfinite(quantized_unit) or quantized_unit <= 0:
            raise ValueError("quantized_unit must be finite and positive")

    vms = list(vms)
    known_ids = {vm.vm_id for vm in vms}
    violations = []
    loads = {resource: [0.0] * len(servers) for resource in ("cpu", "mem")}
    quantized_loads = {resource: [0] * len(servers) for resource in ("cpu", "mem")}

    for vm in vms:
        if vm.vm_id not in assignment:
            violations.append({"type": "missing_vm", "detail": f"VM {vm.vm_id!r} has no assignment"})

    for vm_id, host in assignment.items():
        if vm_id not in known_ids:
            violations.append({"type": "unknown_vm", "detail": f"Unknown VM {vm_id!r}"})
        if isinstance(host, bool) or not isinstance(host, Integral) or not 0 <= host < len(servers):
            violations.append({"type": "unknown_host",
                               "detail": f"VM {vm_id!r} has invalid host index {host!r}"})

    for vm in vms:
        host = assignment.get(vm.vm_id)
        host_ok = (not isinstance(host, bool) and isinstance(host, Integral)
                   and 0 <= host < len(servers))
        for resource in ("cpu", "mem"):
            size = getattr(vm, f"{resource}_size")
            if not math.isfinite(size) or size < 0:
                violations.append({"type": "non_finite_or_negative",
                                   "detail": f"VM {vm.vm_id!r} {resource}_size is {size!r}"})
                continue
            if host_ok:
                loads[resource][host] += size
                if quantized_unit is not None:
                    quantized_loads[resource][host] += math.ceil(size / quantized_unit)

    for host, server in enumerate(servers):
        for resource in ("cpu", "mem"):
            limit = getattr(server, f"{resource}_limit")
            load = loads[resource][host]
            if load > limit + 1e-9:
                violations.append({"type": f"{resource}_overload",
                                   "detail": f"Host {host} {resource} load {load!r} exceeds limit {limit!r}"})
            if quantized_unit is not None:
                quantized_load = quantized_loads[resource][host]
                quantized_limit = math.floor(limit / quantized_unit)
                if quantized_load > quantized_limit:
                    violations.append({
                        "type": f"quantized_{resource}_overload",
                        "detail": f"Host {host} {resource} units {quantized_load} exceed limit {quantized_limit}",
                    })

    return {"valid": not violations, "violations": violations}


def validate_assignment_pairs(pairs) -> dict:
    """Report repeated VM IDs before assignment pairs are collapsed into a dict."""
    seen = set()
    violations = []
    for vm_id, host in pairs:
        if vm_id in seen:
            violations.append({"type": "duplicate_vm",
                               "detail": f"VM {vm_id!r} appears again with host {host!r}"})
        seen.add(vm_id)
    return {"valid": not violations, "violations": violations}
