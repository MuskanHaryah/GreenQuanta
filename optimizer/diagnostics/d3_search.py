"""Read-only D3 annealing search diagnostics; write only new diagnostic CSVs."""
import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
from statistics import median
import sys
import time


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dwave.samplers import SimulatedAnnealingSampler

from diagnostics.common import CENTER, ENCODING, LAMBDA_CONFIG, build_problem
from diagnostics.d2_scale import decompose, score_assignment
from phase1.qubo_builder import x_var, y_var
from phase2 import config
from phase2.solver import decode
from phase2.validation import validate_allocation, validate_assignment_pairs


OUTPUT_DIR = config.ROOT / "results" / "diagnostics" / "d3"
TOLERANCE = 1e-9
KNOWN_SMALL_H = {
    "assignment_exact.csv": 0.4366451311159,
    "assignment_annealer_only.csv": 0.5012005266234,
}
CSV_FIELDS = (
    "size", "setting", "num_reads", "num_sweeps", "seed", "beta_range",
    "t_hot", "t_cold", "h_exact", "saved_raw_valid_pct", "reproduced_raw_valid_pct",
    "min_h_all", "min_h_valid", "valid_reads", "valid_reads_pct",
    "reads_le_exact", "reads_le_exact_pct", "valid_reads_le_exact",
    "valid_servers_on_histogram", "wall_time_s", "leak_reads", "leak_broken_rules",
    "local_min_delta", "local_negative_flips", "local_negative_flip_details",
)


def encode_assignment(path, servers, bqm):
    """Mirror D2's CSV-to-sample conversion; verify against D2 before sampling."""
    host_indices = {server.server_id: index for index, server in enumerate(servers)}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        pairs = [(int(row["vm_id"]), host_indices[int(row["host_id"])])
                 for row in csv.DictReader(stream)]
    pair_check = validate_assignment_pairs(pairs)
    if not pair_check["valid"]:
        raise ValueError(f"{path}: {pair_check}")
    sample = {variable: 0 for variable in bqm.variables}
    for vm_id, host in pairs:
        name = x_var(vm_id, host)
        if name not in sample:
            raise ValueError(f"{path}: assignment has unknown QUBO variable {name}")
        sample[name] = 1
        sample[y_var(host)] = 1
    return sample


def checked_assignments(size, vms, servers, bqm, parts):
    samples = {}
    for filename in KNOWN_SMALL_H:
        path = config.RESULTS_DIR / size / filename
        sample = encode_assignment(path, servers, bqm)
        actual = float(bqm.energy(sample))
        # D2's public key is total_h (lowercase), not total_H.
        expected = score_assignment(path, vms, servers, bqm, parts)["total_h"]
        assert math.isclose(actual, expected, rel_tol=0.0, abs_tol=TOLERANCE), (
            f"{filename}: encoded H={actual:.16g}, D2 total_h={expected:.16g}"
        )
        if size == "small_v2":
            assert math.isclose(actual, KNOWN_SMALL_H[filename], rel_tol=0.0,
                                abs_tol=TOLERANCE), (
                f"{filename}: encoded H={actual:.16g}, known H={KNOWN_SMALL_H[filename]:.16g}"
            )
        print(f"Encoding safeguard: {filename}: H={actual:.16g}, "
              f"D2 total_h={expected:.16g}, PASS")
        samples[filename] = sample
    return samples


def flip_bound(bqm, variable):
    return abs(float(bqm.linear[variable])) + math.fsum(
        abs(float(bias)) for bias in bqm.adj[variable].values()
    )


def raw_valid_pct(sampleset, vms, servers):
    valid, total = 0, 0
    for rec in sampleset.data(fields=["sample", "num_occurrences"], sorted_by=None):
        count = int(rec.num_occurrences)
        total += count
        if decode(rec.sample, vms, servers)[0] is not None:
            valid += count
    return round(100.0 * valid / total, 2)


def read_landscape(sampleset, vms, servers, h_exact):
    """Count reads (including multiplicities), using SampleSet energies and decode."""
    total, valid, below, valid_below = 0, 0, 0, 0
    min_all, min_valid = math.inf, None
    histogram, leak_rules, leaks = Counter(), Counter(), []
    for index, rec in enumerate(sampleset.data(
            fields=["sample", "energy", "num_occurrences"], sorted_by=None)):
        count, energy = int(rec.num_occurrences), float(rec.energy)
        assignment, broken = decode(rec.sample, vms, servers)
        is_valid = assignment is not None
        servers_on = sum(rec.sample[y_var(h)] == 1 for h in range(len(servers)))
        total += count
        min_all = min(min_all, energy)
        if is_valid:
            valid += count
            min_valid = energy if min_valid is None else min(min_valid, energy)
            histogram[int(servers_on)] += count
        if energy <= h_exact + TOLERANCE:
            below += count
            if is_valid:
                valid_below += count
            else:
                rules = sorted(broken)
                leaks.append({"record": index, "reads": count, "h": energy,
                              "broken_rules": rules})
                for rule in rules:
                    leak_rules[rule] += count
    return {
        "min_h_all": min_all, "min_h_valid": min_valid,
        "valid_reads": valid, "valid_reads_pct": 100.0 * valid / total,
        "reads_le_exact": below, "reads_le_exact_pct": 100.0 * below / total,
        "valid_reads_le_exact": valid_below,
        "valid_servers_on_histogram": dict(sorted(histogram.items())),
        "leak_reads": below - valid_below,
        "leak_broken_rules": dict(sorted(leak_rules.items())),
    }, leaks


def local_minimum(bqm, sample, h_exact, vms, servers):
    deltas, negative = [], []
    for variable in bqm.variables:
        flipped = dict(sample)
        flipped[variable] = 1 - flipped[variable]
        delta = float(bqm.energy(flipped)) - h_exact
        deltas.append(delta)
        if delta < 0.0:
            assignment, broken = decode(flipped, vms, servers)
            validity = (validate_allocation(vms, assignment, servers)
                        if assignment is not None else "not decodable")
            detail = {"variable": variable, "delta_h": delta,
                      "real_resource_validity": validity, "broken_rules": sorted(broken)}
            negative.append(detail)
            print("Negative flip: " + json.dumps(detail, sort_keys=True))
    result = {"local_min_delta": min(deltas), "local_negative_flips": len(negative),
              "local_negative_flip_details": negative}
    print(f"Local minimum: min delta H={result['local_min_delta']:.16g}, "
          f"negative flips={len(negative)}")
    return result


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows({key: json.dumps(value, sort_keys=True)
                          if isinstance(value, (dict, list, tuple)) else value
                          for key, value in row.items()} for row in rows)


def sample(sampler, bqm, reads, sweeps, beta_range=None):
    kwargs = {"num_reads": reads, "num_sweeps": sweeps, "seed": config.SEED}
    if beta_range is not None:
        kwargs["beta_range"] = beta_range
    start = time.perf_counter()
    sampleset = sampler.sample(bqm, **kwargs)
    return sampleset, time.perf_counter() - start


def diagnose(size):
    print(f"\n=== {size}: encoding={ENCODING}, lambda_config={LAMBDA_CONFIG}, center={CENTER} ===")
    vms, servers, bqm, _ = build_problem(size)
    parts, _ = decompose(vms, servers, bqm)
    try:
        encoded = checked_assignments(size, vms, servers, bqm, parts)
    except (AssertionError, ValueError, KeyError, OSError) as error:
        print(f"Encoding safeguard: FAIL: {error}; STOP {size} before sampling")
        return False

    with (config.RESULTS_DIR / size / "summary.json").open(encoding="utf-8") as stream:
        saved = json.load(stream)["solver"]["valid_reads_raw_pct"]
    sampler = SimulatedAnnealingSampler()
    baseline, baseline_time = sample(sampler, bqm, 200, 2000)
    reproduced = raw_valid_pct(baseline, vms, servers)
    passed = saved == reproduced
    print(f"Reproduction gate (solver.valid_reads_raw_pct): saved={saved:.2f}%, "
          f"reproduced={reproduced:.2f}%, {'PASS' if passed else 'FAIL'}")
    if not passed:
        print(f"STOP {size}: reproduction gate differs; no later diagnostics or CSV")
        return False

    exact = encoded["assignment_exact.csv"]
    h_exact = float(bqm.energy(exact))
    print(f"H_exact={h_exact:.16g}")
    print(f"S0 default beta_range={baseline.info['beta_range']}")
    t_hot = median(flip_bound(bqm, variable) for variable in bqm.variables)
    t_cold = 0.01 * min(float(parts["energy"].linear[y_var(h)])
                        for h in range(len(servers)))
    if not (0.0 < t_cold < t_hot):
        raise ValueError(f"Invalid explicit temperatures: T_hot={t_hot}, T_cold={t_cold}")
    settings = (
        ("S0", 200, 2000, None, None),
        ("S1", 200, 10000, None, None),
        ("S2", 1000, 2000, None, None),
        ("S3", 200, 2000, (1.0 / t_hot, 1.0 / t_cold), t_hot),
        ("S4", 200, 2000, (1.0 / (10.0 * t_hot), 1.0 / t_cold), 10.0 * t_hot),
    )
    rows = []
    for name, reads, sweeps, beta_range, hot in settings:
        if beta_range is not None:
            print(f"{name}: T_hot={hot:.16g}, T_cold={t_cold:.16g}, beta_range={beta_range}")
        # S0 is precisely the successful reproduction sample, not an additional run.
        sampleset, wall_time = ((baseline, baseline_time) if name == "S0" else
                                sample(sampler, bqm, reads, sweeps, beta_range))
        metrics, leaks = read_landscape(sampleset, vms, servers, h_exact)
        print(f"{name}: " + json.dumps({**metrics, "wall_time_s": wall_time}, sort_keys=True))
        if leaks:
            print(f"Leak check {name}: {metrics['leak_reads']} invalid reads with H <= H_exact + 1e-9")
            for leak in leaks:
                print("  " + json.dumps(leak, sort_keys=True))
        else:
            print(f"Leak check {name}: none")
        rows.append({"size": size, "setting": name, "num_reads": reads,
                     "num_sweeps": sweeps, "seed": config.SEED,
                     "beta_range": sampleset.info["beta_range"],
                     "t_hot": hot, "t_cold": t_cold if hot is not None else None,
                     "h_exact": h_exact, "saved_raw_valid_pct": saved,
                     "reproduced_raw_valid_pct": reproduced,
                     **metrics, "wall_time_s": wall_time})
    local = local_minimum(bqm, exact, h_exact, vms, servers)
    for row in rows:
        row.update(local)
    write_csv(OUTPUT_DIR / f"{size}.csv", rows)
    print(f"Saved: results/diagnostics/d3/{size}.csv")
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", choices=("small_v2", "medium_v2", "large_v2"),
                        default=["small_v2", "medium_v2", "large_v2"])
    args = parser.parse_args(argv)
    if len(args.sizes) != len(set(args.sizes)):
        parser.error("--sizes must not repeat a size")
    if config.SEED != 42:
        parser.error(f"D3 requires config.SEED == 42; found {config.SEED}")
    for size in args.sizes:
        output = OUTPUT_DIR / f"{size}.csv"
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite existing diagnostic output: {output}")
    passed = [diagnose(size) for size in args.sizes]
    return 0 if all(passed) else 1


if __name__ == "__main__":
    raise SystemExit(main())
