"""Manifest regressions; all file writes are confined to pytest tmp_path."""
import csv
import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


_spec = importlib.util.spec_from_file_location(
    "run_manifest",
    Path(__file__).resolve().parents[1] / "optimizer" / "phase2" / "manifest.py",
)
manifest = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(manifest)


@pytest.fixture
def provenance(tmp_path):
    dataset = tmp_path / "dataset.csv"
    dataset.write_bytes(b"vm_id,cpu\n1,0.25\n")
    return {
        "method": "test", "sampler_seed": 10, "workload_seed": 20,
        "dataset_path": dataset, "workload_obj": [{"vm_id": 1, "cpu_size": 0.25}],
        "repo_dir": tmp_path, "package_names": [],
    }


def test_canonical_hash_ignores_key_order():
    first = {"method": "test", "nested": {"cpu": 1, "ram": 2}}
    second = {"nested": {"ram": 2, "cpu": 1}, "method": "test"}
    assert manifest.canonical_hash(first) == manifest.canonical_hash(second)
    assert len(manifest.canonical_hash(first)) == 64


def test_sampler_seed_changes_config_hash(provenance):
    first = manifest.build_manifest(config={"sampler_seed": 10}, **provenance)
    second = manifest.build_manifest(
        config={"sampler_seed": 11}, **{**provenance, "sampler_seed": 11},
    )
    assert first["config_hash"] != second["config_hash"]
    assert first["sampler_seed"] == 10
    assert second["sampler_seed"] == 11
    assert first["workload_seed"] == provenance["workload_seed"]
    assert first["dataset_hash"] == manifest.file_hash(provenance["dataset_path"])
    assert first["workload_hash"] == manifest.canonical_hash(provenance["workload_obj"])
    assert first["code_revision"] == "unknown"
    assert first["dirty_worktree"] is True
    assert first["untracked_count"] is None
    assert first["run_id"].endswith(f"_test_{first['config_hash'][:8]}_s10")
    assert set(first) == {
        "run_id", "timestamp", "code_revision", "dirty_worktree", "dataset_hash",
        "workload_hash", "config_hash", "dependency_versions", "sampler_seed",
        "workload_seed", "method", "untracked_count",
    }


def test_write_manifest_never_overwrites(tmp_path, provenance):
    run = manifest.build_manifest(config={"sampler_seed": 10}, **provenance)
    path = manifest.write_manifest(tmp_path / "runs", run)
    original = path.read_bytes()
    assert path == tmp_path / "runs" / run["run_id"] / "manifest.json"
    assert json.loads(original) == run
    with pytest.raises(FileExistsError):
        manifest.write_manifest(tmp_path / "runs", {**run, "method": "changed"})
    assert path.read_bytes() == original
    assert list(path.parent.iterdir()) == [path]


def test_experiment_log_appends_without_repeating_header(tmp_path):
    path = tmp_path / "experiment_log.csv"
    columns = [
        "run_id", "git_commit", "config_hash", "sampler_seed", "workload_seed",
        "method", "metrics_json", "gate_result", "timestamp",
    ]
    row = dict(zip(columns, ["first", "abc", "hash", 10, 20, "test",
                            json.dumps({"energy": 1, "valid": True}), "pass", "utc"]))
    manifest.append_experiment_log(path, row)
    first_bytes = path.read_bytes()
    manifest.append_experiment_log(path, {**row, "run_id": "second"})
    assert path.read_bytes().startswith(first_bytes)
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    assert len(rows) == 3
    assert rows[0] == columns
    assert rows[1] == [str(row[column]) for column in columns]
    assert rows[2] == ["second"] + rows[1][1:]
    assert rows.count(columns) == 1


def test_file_hash_changes_with_one_byte(tmp_path):
    path = tmp_path / "bytes.bin"
    path.write_bytes(b"abc")
    original = manifest.file_hash(path)
    assert original == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    path.write_bytes(b"abd")
    assert manifest.file_hash(path) != original


def test_dependency_versions_reports_missing():
    name = "greenquanta-deliberately-nonexistent-distribution-7d824cf9"
    assert manifest.dependency_versions([name]) == {name: "missing"}


def test_git_info_distinguishes_untracked_files(tmp_path):
    git = shutil.which("git")
    if git is None:
        pytest.skip("git executable is missing")

    def run_git(*args):
        return subprocess.run(
            [git, "--no-pager", *args], cwd=tmp_path,
            env={**os.environ, "GIT_EDITOR": "true"}, check=True,
            capture_output=True, text=True,
        )

    run_git("init")
    run_git("config", "--local", "user.name", "Manifest Test")
    run_git("config", "--local", "user.email", "manifest-test@example.invalid")
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("original\n", encoding="utf-8")
    run_git("add", "tracked.txt")
    run_git("-c", "commit.gpgsign=false", "-c", f"core.hooksPath={tmp_path / 'no-hooks'}",
            "commit", "-m", "Add tracked fixture")

    clean = manifest.git_info(tmp_path)
    assert clean["code_revision"] == run_git("rev-parse", "HEAD").stdout.strip()
    assert clean["dirty_worktree"] is False
    assert clean["untracked_count"] == 0

    (tmp_path / "untracked.txt").write_text("untracked\n", encoding="utf-8")
    untracked = manifest.git_info(tmp_path)
    assert untracked["dirty_worktree"] is False
    assert untracked["untracked_count"] == 1

    tracked.write_text("modified\n", encoding="utf-8")
    modified = manifest.git_info(tmp_path)
    assert modified["dirty_worktree"] is True
    assert modified["untracked_count"] == 1
