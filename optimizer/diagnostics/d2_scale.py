"""Decompose QUBO scales and score saved placements; never run a solver."""
import argparse
import csv
import json
import math
from pathlib import Path
import sys
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diagnostics.common import CENTER, ENCODING, LAMBDA_CONFIG, build_problem
from phase1.qubo_builder import x_var, y_var
from phase2 import config, qubo_builder_v2
from phase2.validation import validate_allocation, validate_assignment_pairs


TERMS = ("energy", "assign", "cpu", "ram", "link")
TOLERANCE = 1e-12
OUTPUT_DIR = config.ROOT / "results" / "diagnostics" / "d2"
CSV_FIELDS = (
    "size", "kind", "term", "max_abs_coef", "min_nonzero_abs_coef", "ratio",
    "nonzero_count", "servers", "e_ref_joule", "y_min", "y_max", "x_min", "x_max",
    "assignment", "servers_on", "valid", "validation", "total_h",
    "energy_h", "assign_h", "cpu_h", "ram_h", "link_h", "lower_total_h",
    "max_coefficient_error", "status",
)


def assert_decomposition(full, parts):
    """Check every linear/quadratic coefficient and offset with absolute tolerance."""
    variables = set(full.variables)
    edges = {frozenset(edge) for edge in full.quadratic}
    for part in parts.values():
        variables.update(part.variables)
        edges.update(frozenset(edge) for edge in part.quadratic)
    errors = []

    def check(label, actual, values):
        reconstructed = math.fsum(values)
        error = abs(actual - reconstructed)
        if not math.isfinite(error) or error > TOLERANCE:
            raise AssertionError(
                f"QUBO decomposition failed at {label}: full={actual:.17g}, "
                f"parts={reconstructed:.17g}, error={error:.17g}, tolerance={TOLERANCE}"
            )
        errors.append(error)

    for variable in variables:
        check(variable, full.linear.get(variable, 0.0),
              (part.linear.get(variable, 0.0) for part in parts.values()))
    for edge in edges:
        u, v = tuple(edge)
        check((u, v), full.get_quadratic(u, v, default=0.0),
              (part.get_quadratic(u, v, default=0.0) for part in parts.values()))
    check("offset", full.offset, (part.offset for part in parts.values()))
    return max(errors, default=0.0)


def decompose(vms, servers, full):
    """Isolate penalties via temporary lambda_setup patches, restoring on exit."""
    def build_with(lambdas):
        with patch.object(qubo_builder_v2, "lambda_setup", return_value=(lambdas, "D2 isolation")):
            bqm, _ = qubo_builder_v2.build_qubo(
                vms, servers, lambda_config=LAMBDA_CONFIG, encoding=ENCODING, center=CENTER,
            )
        return bqm

    zero = {term: 0.0 for term in TERMS[1:]}
    energy = build_with(zero)
    parts = {"energy": energy}
    for term in TERMS[1:]:
        isolated = build_with({**zero, term: 0.05})
        # Each build includes the same energy objective; remove it exactly once.
        isolated -= energy
        parts[term] = isolated
    error = assert_decomposition(full, parts)
    return parts, error


def coefficient_stats(bqm):
    """Linear and quadratic coefficients only; offsets are not coefficients."""
    nonzero = [abs(float(value)) for values in (bqm.linear, bqm.quadratic)
               for value in values.values() if value != 0.0]
    largest = max(nonzero, default=0.0)
    smallest = min(nonzero, default=0.0)
    return {
        "max_abs_coef": largest,
        "min_nonzero_abs_coef": smallest,
        "ratio": largest / smallest if smallest else 0.0,
        "nonzero_count": len(nonzero),
    }


def score_assignment(path, vms, servers, full, parts):
    host_indices = {server.server_id: index for index, server in enumerate(servers)}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        pairs = [(int(row["vm_id"]), host_indices[int(row["host_id"])])
                 for row in csv.DictReader(stream)]
    pair_check = validate_assignment_pairs(pairs)
    if not pair_check["valid"]:
        raise ValueError(f"{path}: {pair_check}")
    assignment = dict(pairs)
    validity = validate_allocation(vms, assignment, servers)
    sample = {variable: 0 for variable in full.variables}
    for vm_id, host in pairs:
        name = x_var(vm_id, host)
        if name not in sample:
            raise ValueError(f"{path}: assignment has unknown QUBO variable {name}")
        sample[name] = 1
        sample[y_var(host)] = 1
    total = float(full.energy(sample))
    split = {f"{term}_h": float(parts[term].energy(sample)) for term in TERMS}
    return {
        "assignment": path.name,
        "servers_on": len(set(assignment.values())),
        "valid": validity["valid"],
        "validation": json.dumps(validity, sort_keys=True),
        "total_h": total,
        **split,
    }


def diagnose(size):
    print(f"\n=== {size}: encoding={ENCODING}, lambda_config={LAMBDA_CONFIG}, center={CENTER} ===")
    vms, servers, full, info = build_problem(size)
    parts, error = decompose(vms, servers, full)
    print(f"Decomposition: PASS (all linear/quadratic coefficients and offset; "
          f"max error={error:.12e}, tolerance={TOLERANCE:.12e})")
    rows = [{"kind": "decomposition", "max_coefficient_error": error, "status": "pass"}]
    print("term        max |coef|         min nonzero |coef|  ratio              nonzero count")
    for term in TERMS:
        stats = coefficient_stats(parts[term])
        print(f"{term:<11} {stats['max_abs_coef']:.12e} {stats['min_nonzero_abs_coef']:.12e} "
              f"{stats['ratio']:.12e} {stats['nonzero_count']}")
        rows.append({"kind": "term", "term": term, **stats})
    y_coeffs = [float(parts["energy"].linear[y_var(h)]) for h in range(len(servers))]
    x_coeffs = [float(parts["energy"].linear[x_var(vm.vm_id, h)])
                for vm in vms for h in range(len(servers))]
    details = {
        "servers": len(servers), "e_ref_joule": info["e_ref_joule"],
        "y_min": min(y_coeffs), "y_max": max(y_coeffs),
        "x_min": min(x_coeffs), "x_max": max(x_coeffs),
    }
    print(f"Energy: servers={len(servers)}, e_ref_joule={info['e_ref_joule']:.12e}")
    print(f"Energy y (cost of opening one server, normalized units): "
          f"min={details['y_min']:.12e}, max={details['y_max']:.12e}")
    print(f"Energy x (VM-on-server, normalized units): "
          f"min={details['x_min']:.12e}, max={details['x_max']:.12e}")
    rows.append({"kind": "energy_details", **details})
    scored = {}
    for filename in ("assignment_exact.csv", "assignment_annealer_only.csv"):
        path = config.RESULTS_DIR / size / filename
        if not path.is_file():
            print(f"Missing assignment: {path}; continuing.")
            rows.append({"kind": "assignment", "assignment": filename, "status": "missing"})
            continue
        score = score_assignment(path, vms, servers, full, parts)
        scored[filename] = score
        rows.append({"kind": "assignment", **score})
        print(f"{filename}: servers_on={score['servers_on']}, "
              f"real_resource_valid={score['valid']}, total_H={score['total_h']:.12e}")
        print("  H split: " + " / ".join(f"{term}={score[f'{term}_h']:.12e}" for term in TERMS))
        print(f"  Real-resource validation: {score['validation']}")
    if len(scored) == 2:
        exact = scored["assignment_exact.csv"]["total_h"]
        annealer = scored["assignment_annealer_only.csv"]["total_h"]
        lower = ("assignment_exact.csv" if exact < annealer else
                 "assignment_annealer_only.csv" if annealer < exact else "equal")
        print(f"Lower total H: {lower}")
    else:
        lower = "unavailable (missing assignment)"
        print(f"Lower total H: {lower}")
    rows.append({"kind": "comparison", "lower_total_h": lower})
    output = OUTPUT_DIR / f"{size}.csv"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with output.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows({"size": size, **row} for row in rows)
    print(f"Saved: results/diagnostics/d2/{size}.csv")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", choices=("small_v2", "medium_v2", "large_v2"),
                        default=["small_v2", "medium_v2", "large_v2"])
    args = parser.parse_args(argv)
    if len(args.sizes) != len(set(args.sizes)):
        parser.error("--sizes must not repeat a size")
    for size in args.sizes:
        output = OUTPUT_DIR / f"{size}.csv"
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite existing diagnostic output: {output}")
    for size in args.sizes:
        diagnose(size)


if __name__ == "__main__":
    main()
