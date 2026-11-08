"""Deterministic fingerprints of QUBO coefficients (including explicit zeros)."""
import hashlib


def qubo_fingerprint(bqm) -> dict:
    """Hash UTF-8 lines: name:value, name1,name2:value, then offset:value.

    Names are converted to strings; linear names and canonical quadratic pairs
    are lexically sorted. Values use %.12e, with newlines between lines and no
    trailing newline. The BQM is not modified.
    """
    lines = [f"{name}:{float(bqm.linear[name]):.12e}"
             for name in sorted(bqm.variables, key=str)]
    quadratic = [(tuple(sorted((str(u), str(v)))), float(value))
                 for (u, v), value in bqm.quadratic.items()]
    lines.extend(f"{u},{v}:{value:.12e}" for (u, v), value in sorted(quadratic))
    lines.append(f"offset:{float(bqm.offset):.12e}")
    return {
        "sha256": hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest(),
        "num_variables": len(bqm.variables),
        "num_interactions": len(bqm.quadratic),
        "offset": float(bqm.offset),
    }
