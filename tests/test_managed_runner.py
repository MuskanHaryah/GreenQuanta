"""Unit tests for managed-run configuration provenance; no optimizer runs."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "optimizer"))

from run_phase2_v2_managed import build_args, build_run_config
from phase2.manifest import canonical_hash


def test_build_run_config_seed_changes_hash():
    first = build_run_config(build_args(["--seed", "42"]))
    second = build_run_config(build_args(["--seed", "43"]))
    assert canonical_hash(first) != canonical_hash(second)


def test_build_run_config_hash_ignores_dict_key_order():
    config = build_run_config(build_args(["--seed", "42"]))
    reordered = dict(reversed(list(config.items())))
    assert list(config) != list(reordered)
    assert canonical_hash(config) == canonical_hash(reordered)


def test_build_run_config_contains_seed():
    args = build_args([])
    config = build_run_config(args)
    assert "seed" in config
    assert config["seed"] == args.seed
