#!/usr/bin/env python3
# make_pseudolabels.py
# Build pseudolabels via MiniBatchKMeans and save to CSV + metadata JSON.
# Works in Jupyter (call make_pseudolabels(...)) and as a CLI script.

import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Union

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans

TYPICAL_DOSE_NAMES = {"dose_val_rx", "dose", "dosage", "dose_val", "abx_dose"}

def _norm_list(x: Optional[Union[str, List[str]]], default: List[str]) -> List[str]:
    """Accept list or comma-separated string; return clean list."""
    if x is None:
        return list(default)
    if isinstance(x, str):
        return [p.strip() for p in x.split(",") if p.strip()]
    return list(x)

def _find_dose_col(df: pd.DataFrame, user_col: Optional[str]) -> str:
    if user_col and user_col in df.columns:
        return user_col
    lower = {c.lower(): c for c in df.columns}
    for name in TYPICAL_DOSE_NAMES:
        if name in lower:
            return lower[name]
    raise ValueError(
        f"Could not find dose column. Tried {sorted(TYPICAL_DOSE_NAMES)}. "
        f"Available columns (first 20): {list(df.columns)[:20]}"
    )

def make_pseudolabels(
    input_path: str,
    output_csv: Optional[str] = None,
    meta_json: Optional[str] = None,
    dose_col: Optional[str] = None,
    id_cols: Optional[Union[str, List[str]]] = None,
    drop_cols: Optional[Union[str, List[str]]] = None,
    k: int = 4,
    margin_pct: float = 0.0,
    random_state: int = 1337,
    batch_size: int = 1024,
    max_iter: int = 200,
    n_init: int = 5,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Build pseudolabels: MiniBatchKMeans on numeric features -> per-cluster median dose ->
    label 1=increase if dose < (1-margin)*median else 0. Writes CSV + JSON and returns (df, meta).
    """

    in_path = Path(input_path).expanduser()
    if not in_path.exists():
        cwd = Path.cwd()
        raise FileNotFoundError(
            f"Input file not found: {in_path}\n"
            f"Current working directory: {cwd}\n"
            f"Tip: In Windows Jupyter, use raw strings like r'C:\\path\\file.csv'"
        )

    # Default outputs next to input if not provided
    if output_csv is None:
        output_csv = str(in_path.with_name("pseudolabel_dataset.csv"))
    if meta_json is None:
        meta_json = str(in_path.with_name("pseudolabel_meta.json"))

    out_csv = Path(output_csv).expanduser()
    meta_path = Path(meta_json).expanduser()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    id_cols = _norm_list(id_cols, default=["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID"])
    drop_cols = _norm_list(drop_cols, default=["is_dead"])

    print(f"[Info] Reading: {in_path}")
    df = pd.read_csv(in_path, low_memory=False)
    print(f"[Info] Shape: {df.shape}")

    # Dose column
    dose_col = _find_dose_col(df, dose_col)
    df[dose_col] = pd.to_numeric(df[dose_col], errors="coerce")
    print(f"[Info] Using dose column: {dose_col}")

    # Keep only IDs that actually exist
    id_cols_found = [c for c in id_cols if c in df.columns]
    if id_cols and not id_cols_found:
        print(f"[Warn] None of the requested id_cols are present; continuing without IDs.")

    # Numeric features (exclude IDs, dose, and extra drop cols)
    drop_set = set(id_cols_found + [dose_col] + drop_cols)
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in drop_set]
    if not num_cols:
        raise ValueError(
            "No numeric feature columns remain after dropping IDs/dose/drop_cols.\n"
            f"Dropped: {sorted(drop_set)}\n"
            f"Numeric columns available: {list(df.select_dtypes(include=[np.number]).columns)[:20]}"
        )
    print(f"[Info] #numeric feature columns: {len(num_cols)}")

    # Preprocess: impute + scale
    preprocess = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    X = df[num_cols].copy()
    X_mat = preprocess.fit_transform(X)

    # Fast clustering
    print(f"[Info] Clustering with MiniBatchKMeans(k={k}) ...")
    kmeans = MiniBatchKMeans(
        n_clusters=k,
        random_state=random_state,
        batch_size=batch_size,
        n_init=n_init,
        max_iter=max_iter,
        verbose=0
    )
    clusters = kmeans.fit_predict(X_mat)

    # Per-cluster median dose + pseudolabels
    df["_cluster"] = clusters
    med = df.groupby("_cluster")[dose_col].median()
    df["_cluster_median_dose"] = df["_cluster"].map(med)

    thr = (1.0 - margin_pct) * df["_cluster_median_dose"]
    df["_pseudo_label"] = np.where(
        df[dose_col].notna(),
        (df[dose_col] < thr).astype(int),
        np.nan
    )

    keep_cols = id_cols_found + [dose_col, "_cluster", "_cluster_median_dose"] + num_cols
    pseudo_df = df.loc[df["_pseudo_label"].notna(), keep_cols].copy()
    pseudo_df["_pseudo_label"] = df.loc[pseudo_df.index, "_pseudo_label"].astype(int)
    pseudo_df["_pseudo_label_str"] = pseudo_df["_pseudo_label"].map({0: "decrease", 1: "increase"})

    # Save outputs
    pseudo_df.to_csv(out_csv, index=False)
    meta = {
        "input_path": str(in_path),
        "output_csv": str(out_csv),
        "dose_col": dose_col,
        "id_cols": id_cols_found,
        "feature_cols": num_cols,
        "k": k,
        "margin_pct": margin_pct,
        "random_state": random_state,
        "cluster_medians": {int(k_): (None if pd.isna(v) else float(v)) for k_, v in med.items()}
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[OK] Wrote pseudolabel dataset: {out_csv.resolve()}")
    print(f"[OK] Wrote metadata JSON:       {meta_path.resolve()}")
    print(f"[Info] Rows with pseudolabels:  {len(pseudo_df)} / {len(df)}")

    return pseudo_df, meta

# ---------------- Script mode (still available) ----------------
def _running_in_notebook() -> bool:
    return "ipykernel" in sys.modules

if __name__ == "__main__" and not _running_in_notebook():
    import argparse
    ap = argparse.ArgumentParser(description="Create pseudolabel dataset from raw antibiotic table")
    ap.add_argument("--input", required=True, help="Path to raw CSV (e.g., output_with_is_dead.csv)")
    ap.add_argument("--output-csv", default=None, help="Where to write pseudolabel dataset (default: next to input)")
    ap.add_argument("--meta-json", default=None, help="Where to write metadata JSON (default: next to input)")
    ap.add_argument("--dose-col", default=None, help="Dose column name (auto-detect if omitted)")
    ap.add_argument("--id-cols", default="SUBJECT_ID,HADM_ID,ICUSTAY_ID", help="Comma-separated ID columns")
    ap.add_argument("--drop-cols", default="is_dead", help="Comma-separated extra columns to drop from features")
    ap.add_argument("--k", type=int, default=4, help="Number of clusters for KMeans")
    ap.add_argument("--margin-pct", type=float, default=0.0, help="Label 1 if dose < (1 - margin)*cluster_median")
    ap.add_argument("--random-state", type=int, default=1337)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--max-iter", type=int, default=200)
    ap.add_argument("--n-init", type=int, default=5)
    args = ap.parse_args()

    pseudo_df, meta = make_pseudolabels(
        input_path=args.input,
        output_csv=args.output_csv,
        meta_json=args.meta_json,
        dose_col=args.dose_col,
        id_cols=args.id_cols,
        drop_cols=args.drop_cols,
        k=args.k,
        margin_pct=args.margin_pct,
        random_state=args.random_state,
        batch_size=args.batch_size,
        max_iter=args.max_iter,
        n_init=args.n_init,
    )
