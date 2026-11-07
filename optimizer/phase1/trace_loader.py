"""
Load the Google Cluster Trace 2019 extract (CSV saved from BigQuery).

Expected columns (one row = one instance in one 5-minute window):
    collection_id, instance_index, window_idx, avg_cpu, max_cpu, avg_mem
CPU is in NCU and memory in NMU, exactly as Google publishes them.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

REQUIRED_COLUMNS = ["collection_id", "instance_index", "window_idx", "avg_cpu", "max_cpu", "avg_mem"]


def load_trace(csv_path) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Trace file not found: {path}\n"
            "Run optimizer/sql/phase1_usage_query.sql in BigQuery and save the result "
            "as CSV at this path (see the Phase 1 guide)."
        )

    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Trace CSV is missing columns: {missing}. Found: {list(df.columns)}")

    df = df[REQUIRED_COLUMNS].copy()
    df["collection_id"] = df["collection_id"].astype(np.int64)
    df["instance_index"] = df["instance_index"].astype(np.int64)
    df["window_idx"] = df["window_idx"].astype(np.int64)
    for col in ["avg_cpu", "max_cpu", "avg_mem"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).clip(lower=0.0)

    # keep only the 24-hour window
    df = df[(df["window_idx"] >= 0) & (df["window_idx"] < config.WINDOWS_PER_DAY)]

    # if BigQuery returned two rows for the same instance and window, merge them
    df = (
        df.groupby(["collection_id", "instance_index", "window_idx"], as_index=False)
        .agg(avg_cpu=("avg_cpu", "sum"), max_cpu=("max_cpu", "max"), avg_mem=("avg_mem", "sum"))
    )

    # keep only instances that were running for (almost) the whole day
    seen = df.groupby(["collection_id", "instance_index"])["window_idx"].transform("count")
    df = df[seen >= config.MIN_WINDOWS_PRESENT]
    if df.empty:
        raise ValueError("No instance in the trace covers the full day. Check the BigQuery export.")
    return df.reset_index(drop=True)


def trace_summary(df: pd.DataFrame) -> dict:
    return {
        "collections": int(df["collection_id"].nunique()),
        "instances": int(df.groupby(["collection_id", "instance_index"]).ngroups),
        "rows": int(len(df)),
    }
