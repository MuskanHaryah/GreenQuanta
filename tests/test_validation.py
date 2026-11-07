"""Tiny allocation cases; stand-ins avoid trace, catalog, and solver imports."""
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace


_spec = importlib.util.spec_from_file_location(
    "allocation_validation",
    Path(__file__).resolve().parents[1] / "optimizer" / "phase2" / "validation.py",
)
_validation = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_validation)
validate_allocation = _validation.validate_allocation
validate_assignment_pairs = _validation.validate_assignment_pairs


def vm(vm_id, cpu=0.25, mem=0.25):
    return SimpleNamespace(vm_id=vm_id, cpu_size=cpu, mem_size=mem)


def server(cpu=1.0, mem=1.0):
    return SimpleNamespace(cpu_limit=cpu, mem_limit=mem)


def assert_violations(result, *types):
    assert result["valid"] is False
    assert sorted(v["type"] for v in result["violations"]) == sorted(types)
    assert all(isinstance(v["detail"], str) and v["detail"] for v in result["violations"])


def test_fully_valid():
    vms = [vm(10, 0.5, 0.25), vm(20, 0.5, 0.75), vm(30, 0.0, 0.0)]
    assignment = {10: 0, 20: 0, 30: 1}
    servers = [server(), server()]
    expected = {"valid": True, "violations": []}
    assert validate_allocation(vms, assignment, servers) == expected
    assert validate_allocation(vms, assignment, servers, quantized_unit=0.25) == expected
    assert validate_assignment_pairs(list(assignment.items())) == expected
    assert validate_allocation([], {}, []) == expected
    assert validate_allocation([vm(1, 1.0 + 0.5e-9)], {1: 0}, [server()]) == expected


def test_missing_vm():
    assert_violations(validate_allocation([vm(1)], {}, [server()]), "missing_vm")


def test_unknown_vm():
    assert_violations(validate_allocation([vm(1)], {1: 0, 99: 0}, [server()]), "unknown_vm")


def test_unknown_host():
    for host in (-1, 1, 100, 0.5, "0", None, True):
        assert_violations(validate_allocation([vm(1)], {1: host}, [server()]), "unknown_host")
    assert_violations(validate_allocation([vm(1)], {1: 0}, []), "unknown_host")


def test_cpu_overload():
    assert_violations(
        validate_allocation([vm(1, 0.6), vm(2, 0.5)], {1: 0, 2: 0}, [server()]),
        "cpu_overload",
    )


def test_mem_overload():
    assert_violations(
        validate_allocation([vm(1, mem=0.6), vm(2, mem=0.5)], {1: 0, 2: 0}, [server()]),
        "mem_overload",
    )


def test_non_finite_or_negative():
    for resource in ("cpu", "mem"):
        for size in (math.nan, math.inf, -math.inf, -0.1):
            bad_vm = vm(1, **{resource: size})
            assert_violations(
                validate_allocation([bad_vm], {1: 0}, [server()], quantized_unit=0.25),
                "non_finite_or_negative",
            )
    assert_violations(
        validate_allocation([vm(1, math.nan)], {}, [server()]),
        "missing_vm", "non_finite_or_negative",
    )


def test_quantized_cpu_overload():
    vms = [vm(1, 0.26), vm(2, 0.26)]
    assignment = {1: 0, 2: 0}
    servers = [server(cpu=0.99)]
    assert validate_allocation(vms, assignment, servers)["valid"] is True
    assert_violations(
        validate_allocation(vms, assignment, servers, quantized_unit=0.25),
        "quantized_cpu_overload",
    )
    assert_violations(
        validate_allocation([vm(1, 1.25)], {1: 0}, [server()], quantized_unit=0.25),
        "cpu_overload", "quantized_cpu_overload",
    )


def test_quantized_mem_overload():
    vms = [vm(1, mem=0.26), vm(2, mem=0.26)]
    assignment = {1: 0, 2: 0}
    servers = [server(mem=0.99)]
    assert validate_allocation(vms, assignment, servers)["valid"] is True
    assert_violations(
        validate_allocation(vms, assignment, servers, quantized_unit=0.25),
        "quantized_mem_overload",
    )
    assert_violations(
        validate_allocation([vm(1, mem=1.25)], {1: 0}, [server()], quantized_unit=0.25),
        "mem_overload", "quantized_mem_overload",
    )


def test_quantized_float_edge():
    for size in (0.35, 0.35000000000000003):
        assert validate_allocation(
            [vm(1, cpu=size, mem=size)], {1: 0},
            [server(cpu=0.35, mem=0.35)], quantized_unit=0.05,
        ) == {"valid": True, "violations": []}
    assert_violations(
        validate_allocation([vm(1, cpu=0.0, mem=0.0)], {1: 0},
                            [server(cpu=0.0, mem=0.0)], quantized_unit=0.05),
        "quantized_cpu_overload", "quantized_mem_overload",
    )


def test_duplicate_vm():
    for pairs in ([(1, 0), (1, 1)], [(1, 0), (1, 0)]):
        assert_violations(validate_assignment_pairs(pairs), "duplicate_vm")
