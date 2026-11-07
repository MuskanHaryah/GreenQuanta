"""Run provenance and append-only local outputs; no optimizer execution."""
import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path


LOG_COLUMNS = (
    "run_id", "git_commit", "config_hash", "sampler_seed", "workload_seed",
    "method", "metrics_json", "gate_result", "timestamp",
)


def canonical_hash(obj) -> str:
    """Hash a compact, key-sorted JSON representation as UTF-8."""
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_hash(path) -> str:
    """Hash file bytes without loading the whole dataset into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_info(repo_dir) -> dict:
    """Return HEAD and tracked/untracked dirtiness without optional Git writes.

    On Git failure, the revision is "unknown" and dirtiness is conservatively
    True: an unavailable status must not be mistaken for a clean worktree.
    """
    try:
        revision = subprocess.run(
            ["git", "--no-pager", "rev-parse", "HEAD"], cwd=repo_dir,
            check=True, capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "--no-optional-locks", "--no-pager", "status", "--porcelain"],
            cwd=repo_dir, check=True, capture_output=True, text=True, timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {"code_revision": "unknown", "dirty_worktree": True}
    return {"code_revision": revision, "dirty_worktree": bool(status.strip())}


def dependency_versions(names) -> dict:
    """Look up installed distribution versions without importing packages."""
    versions = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "missing"
    return versions


def make_run_id(method, config_hash, seed) -> str:
    """Build a UTC, microsecond-resolution run ID; writers reject collisions."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{timestamp}_{method}_{config_hash[:8]}_s{seed}"


def build_manifest(method, config, sampler_seed, workload_seed, dataset_path,
                   workload_obj, repo_dir, package_names) -> dict:
    """Collect provenance without creating files or modifying the input config.

    Config hashing uses exactly the supplied config. Callers must include seeds
    in that config if changing a seed should change its hash.
    """
    config_digest = canonical_hash(config)
    return {
        "run_id": make_run_id(method, config_digest, sampler_seed),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **git_info(repo_dir),
        "dataset_hash": file_hash(dataset_path),
        "workload_hash": canonical_hash(workload_obj),
        "config_hash": config_digest,
        "dependency_versions": dependency_versions(package_names),
        "sampler_seed": sampler_seed,
        "workload_seed": workload_seed,
        "method": method,
    }


def write_manifest(runs_root, manifest) -> Path:
    """Create a new run directory and return its manifest path; never overwrite."""
    run_dir = Path(runs_root) / manifest["run_id"]
    run_dir.mkdir(parents=True, exist_ok=False)
    path = run_dir / "manifest.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, default=str)
        stream.write("\n")
    return path


def append_experiment_log(log_path, row: dict) -> None:
    """Append a fixed-schema CSV row, adding a header only when creating the file."""
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        stream = path.open("x", newline="", encoding="utf-8")
        is_new = True
    except FileExistsError:
        stream = path.open("a", newline="", encoding="utf-8")
        is_new = False
    with stream:
        writer = csv.DictWriter(stream, fieldnames=LOG_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)
