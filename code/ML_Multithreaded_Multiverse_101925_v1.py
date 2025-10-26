#!/usr/bin/env python3
# ==============================================================================
# Copyright (c) 2025 Aaron Johnson, Drexel University
# Licensed under the MIT License - see LICENSE file for details
# ==============================================================================
"""
CHR ↔ AHA: State‑blocked, parallel multiverse for (a) direct adoption predictors and (b) CHR×(AI/Robotics) interactions on DV21/DV15.

Key features
------------
• Postgres → pandas ingest (no Spark), robust FIPS handling, wide-table cleaning for ~600 CHR variables (3 tables).
• Direct effects: L1-Logit stability selection + RandomForest across N_REPS bootstrap-like state‑blocked splits.
• Interaction effects: OLS with state‑clustered SEs for CHR × (MO11..MO15, MO21) on DV21 & DV15 with controls (log(pop), census division).
• Parallelization: joblib (loky). Default N_REPS=150, N_JOBS=min(8, cpu_count).
• Guardrails: group-blocked splits (anti-leakage), missingness filtering, constant-feature pruning.
• Surprise index: residual-based "unexpected leaders/laggards" per adoption target.
• Fairness lens: residual summaries by rurality quartile (if HRSA IRR present).
• Optional: Knockoffs (knockpy) for FDR-aware discovery; EBM (interpretml) for non-linear shapes & interactions.

Usage example
-------------
python ML_Multithreaded_Multiverse_101925_v1.py [--outdir ./custom_output] [--n-reps 200] [--n-jobs 12]
"""

from __future__ import annotations

import os, re, json, argparse, logging, sys, warnings
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
import pandas as pd

from sqlalchemy import create_engine, text
from joblib import Parallel, delayed, cpu_count

# sklearn / statsmodels
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score
import statsmodels.api as sm

# Optional libraries (safe fallbacks)
try:
    from knockpy.knockoff_filter import KnockoffFilter  # type: ignore
    KNOCKPY_AVAILABLE = True
except Exception:
    KNOCKPY_AVAILABLE = False

try:
    from interpret.glassbox import ExplainableBoostingClassifier  # type: ignore
    INTERPRET_AVAILABLE = True
except Exception:
    INTERPRET_AVAILABLE = False

# ----------------------------- Configuration Settings -----------------------------

# Default configuration values
CONFIG = {
    # Analysis settings
    "YEAR": 2024,
    #"N_REPS": 150,  # for production purposes (reference)
    "N_REPS": 150,  # use 20 for smoke testing
    "N_JOBS": min(8, cpu_count() or 2),
    "CV_MODE": "state",
    "TEST_SIZE": 0.30,
    "ADOPT_THRESHOLD": "zero",  # or "median"
    "INTERACTION_TOP_K": 120,
    "DROP_MISSING_THRESH": 0.40,
    "SEED": 42,
    "LOGIT_C": 0.1,
    "LOGIT_MAX_ITER": 8000,
    "LOGIT_TOL": 1e-3,
    
    # Database settings (will use environment variables with these defaults)
    "POSTGRES_HOST": os.getenv("POSTGRES_HOST", "localhost"),
    "POSTGRES_DB": os.getenv("POSTGRES_DB", "Research_TEST"),
    "POSTGRES_USER": os.getenv("POSTGRES_USER", "postgres"),
    "POSTGRESQL_KEY": os.getenv("POSTGRESQL_KEY")
}

# Default output directory in the same directory as the script
DEFAULT_OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "multiverse_output")

# Targeted moderation configuration (IV31–IV39 × MO11–MO15 on DV15/DV21)
TARGETED_IV_MAP = {
    "IV31": "adult_smoking_raw_value",
    "IV32": "adult_obesity_raw_value",
    "IV33": "food_environment_index_raw_value",
    "IV34": "physical_inactivity_raw_value",
    "IV35": "access_to_exercise_opportunities_raw_value",
    "IV36": "excessive_drinking_raw_value",
    "IV37": "alcohol_impaired_driving_deaths_raw_value",
    "IV38": "sexually_transmitted_infections_raw_value",
    "IV39": "teen_births_raw_value",
}

TARGETED_MODERATORS = ["MO11", "MO12", "MO13", "MO14", "MO15"]
TARGETED_DV_MAP = {
    "DV15": "preventable_hospital_stays_raw_value",
    "DV21": "premature_death_raw_value",
}

# ----------------------------- logging / args -----------------------------

def setup_logging(outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    log_path = os.path.join(outdir, "run_log.txt")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[logging.FileHandler(log_path, mode="w"), logging.StreamHandler(sys.stdout)],
    )
    logging.info("Logging to %s", log_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CHR↔AHA multiverse: state‑blocked ML + interactions")
    p.add_argument("--outdir", default=DEFAULT_OUTDIR, help=f"Output directory (default: {DEFAULT_OUTDIR})")
    p.add_argument("--n-reps", type=int, default=CONFIG["N_REPS"], help=f"Stability/metrics reps (default: {CONFIG['N_REPS']})")
    p.add_argument("--n-jobs", type=int, default=CONFIG["N_JOBS"], help=f"Number of parallel jobs (default: {CONFIG['N_JOBS']})")
    p.add_argument("--seed", type=int, default=CONFIG["SEED"], help=f"Random seed (default: {CONFIG['SEED']})")
    
    # Parse args and update CONFIG with any overrides
    args = p.parse_args()
    CONFIG["N_REPS"] = args.n_reps
    CONFIG["N_JOBS"] = args.n_jobs
    CONFIG["SEED"] = args.seed
    
    return args


# ----------------------------- utilities -----------------------------

ID_COLS = {"county_fips", "_5_digit_fips", "state_fips", "state_abbreviation", "name", "year"}
SAFE_FLOAT = np.float64

def normalize_fips(s: pd.Series) -> pd.Series:
    """
    Normalize any FIPS-like series to 5-digit county codes.

    Handles ints, floats (e.g., 1001.0), string-coded FIPS with padding, and
    mixed artifacts like 'AL01001'. Summary geographies ending in '000' are
    masked to NA.
    """
    ser = pd.Series(s, copy=True)
    numeric = pd.to_numeric(ser, errors="coerce")

    norm = pd.Series(index=ser.index, dtype="object")

    numeric_mask = numeric.notna()
    if numeric_mask.any():
        norm.loc[numeric_mask] = (
            numeric.loc[numeric_mask]
            .round()
            .astype("Int64")
            .astype(str)
            .str.zfill(5)
        )

    non_numeric_mask = ~numeric_mask
    if non_numeric_mask.any():
        cleaned = ser.loc[non_numeric_mask].astype(str).str.strip()
        cleaned = cleaned.str.replace(r"\.0$", "", regex=True)
        cleaned = cleaned.str.replace(r"[^0-9]", "", regex=True)
        norm.loc[non_numeric_mask] = cleaned.str[-5:].str.zfill(5)

    norm = norm.where(norm.str.len() > 0)
    norm = norm.mask(norm.isna())
    norm = norm.mask(norm.str.endswith("000"))
    return norm

def to_numeric_clean(series: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
    if isinstance(series, pd.DataFrame):
        # Apply conversion column-wise and return a DataFrame; callers should select Series where possible.
        return series.apply(to_numeric_clean)
    if series.dtype.kind in "biuf":  # already numeric
        return series.astype(SAFE_FLOAT)
    s = series.astype(str).str.replace(",", "", regex=False)
    s = s.str.replace("%", "", regex=False)
    s = s.str.replace(r"\s+", "", regex=True)
    s = pd.to_numeric(s, errors="coerce").astype(SAFE_FLOAT)
    return s

def pick_numeric_features(df: pd.DataFrame, exclude: List[str], drop_missing_thresh: float) -> List[str]:
    """Select numeric features while logging the audit trail for kept/dropped columns."""
    excluded_by_rule = [c for c in df.columns if c in exclude]
    if excluded_by_rule:
        logging.info(
            "Feature selection: excluded %d columns due to ID/target/control rules: %s",
            len(excluded_by_rule),
            ", ".join(sorted(excluded_by_rule)),
        )

    candidates = [c for c in df.columns if c not in exclude]
    kept_info: List[Tuple[str, str, float, int]] = []
    dropped_info: List[Tuple[str, str, str, float, int]] = []

    for c in candidates:
        col_dtype = str(df[c].dtype)
        missing_pct = float(df[c].isna().mean())
        unique_count = int(df[c].nunique(dropna=True))

        if not pd.api.types.is_numeric_dtype(df[c]):
            dropped_info.append((c, "non-numeric dtype", col_dtype, missing_pct, unique_count))
            continue

        if missing_pct > drop_missing_thresh:
            reason = f"missingness {missing_pct:.1%} > threshold {drop_missing_thresh:.1%}"
            dropped_info.append((c, reason, col_dtype, missing_pct, unique_count))
            continue

        if unique_count <= 1:
            dropped_info.append((c, "constant or single unique value", col_dtype, missing_pct, unique_count))
            continue

        kept_info.append((c, col_dtype, missing_pct, unique_count))

    logging.info(
        "Feature selection audit: %d candidates after exclusions -> %d kept, %d dropped",
        len(candidates),
        len(kept_info),
        len(dropped_info),
    )

    for col, dtype_desc, miss, uniq in kept_info:
        logging.info(
            "  KEEP %-60s dtype=%s missing=%.1f%% unique=%d",
            col,
            dtype_desc,
            miss * 100,
            uniq,
        )

    for col, reason, dtype_desc, miss, uniq in dropped_info:
        logging.info(
            "  DROP %-60s reason=%s dtype=%s missing=%.1f%% unique=%d",
            col,
            reason,
            dtype_desc,
            miss * 100,
            uniq,
        )

    return [col for col, _, _, _ in kept_info]

def add_controls(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    # log(population)
    if "population" in df.columns:
        df["population"] = to_numeric_clean(df["population"])
        df = df[df["population"].notna() & (df["population"] > 0)]
        df["log_population"] = np.log(df["population"])
    else:
        df["log_population"] = np.nan

    # census division dummies
    div_dummies = pd.get_dummies(df.get("census_division", pd.Series(dtype=str)).astype(str),
                                 prefix="div", drop_first=True, dtype=int)
    df = pd.concat([df, div_dummies], axis=1)
    controls = ["log_population"] + list(div_dummies.columns)
    return df, controls

def group_key(df: pd.DataFrame, mode: str) -> pd.Series:
    if mode == "division" and "census_division" in df.columns:
        return df["census_division"].astype(str).fillna("UNK")
    # default to state by FIPS (first two digits)
    if "county_fips" in df.columns:
        return df["county_fips"].astype(str).str[:2]
    if "_5_digit_fips" in df.columns:
        return df["_5_digit_fips"].astype(str).str[:2]
    return pd.Series(["00"] * len(df))

def rng_list(n: int, seed: int) -> List[int]:
    r = np.random.default_rng(seed)
    return list(map(int, r.integers(1, 2_147_483_647, size=n)))


# ----------------------------- DB ingest -----------------------------

def build_engine():
    host = CONFIG["POSTGRES_HOST"]
    db = CONFIG["POSTGRES_DB"]
    user = CONFIG["POSTGRES_USER"]
    pw = CONFIG["POSTGRESQL_KEY"]
    if not pw:
        logging.error("POSTGRESQL_KEY environment variable is not set.")
        sys.exit(1)
    uri = f"postgresql+psycopg2://{user}:{pw}@{host}/{db}"
    eng = create_engine(uri)
    with eng.connect() as c:
        c.execute(text("SELECT 1"))
    logging.info("Connected to Postgres at %s / db=%s", host, db)
    return eng

def read_table(eng, table: str) -> pd.DataFrame:
    try:
        # Try getting column info first to apply casts selectively
        cols_query = f"""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = '{table}' AND table_schema = 'public'
        """
        col_info = pd.read_sql(text(cols_query), eng)
        
        # Build query with casts for numeric-like columns
        numeric_cols = []
        for _, row in col_info.iterrows():
            col_name = row['column_name']
            data_type = row['data_type'].lower()
            if 'int' in data_type or 'numeric' in data_type or 'float' in data_type:
                numeric_cols.append(f'"{col_name}"')
            elif data_type == 'varchar' or data_type == 'text':
                # Try to detect if it could be numeric
                numeric_cols.append(f'NULLIF("{col_name}", \'\')::numeric AS "{col_name}"')
            else:
                numeric_cols.append(f'"{col_name}"')
        
        # Only apply this optimization if we have column info
        if numeric_cols:
            query = f'SELECT {", ".join(numeric_cols)} FROM public."{table}"'
            return pd.read_sql(text(query), eng)
        else:
            return pd.read_sql(text(f'SELECT * FROM public."{table}"'), eng)
    except Exception as e:
        logging.warning(f"Couldn't optimize column types for {table}: {e}")
        # Fallback to original approach
        try:
            return pd.read_sql(text(f'SELECT * FROM public."{table}"'), eng)
        except Exception:
            # try without quotes
            return pd.read_sql(text(f"SELECT * FROM public.{table}"), eng)

def fetch_all_frames(eng, year: int) -> Dict[str, pd.DataFrame]:
    frames: Dict[str, pd.DataFrame] = {}
    # CHR chunks
    for t in ["chr_analytic_chunk_1", "chr_analytic_chunk_2", "chr_analytic_chunk_3"]:
        df = read_table(eng, t)
        if "year" in df.columns:
            # year may be text; keep year==str(year)
            df = df[df["year"].astype(str) == str(year)]
        frames[t] = df
        logging.info("%s: %s rows, %s cols", t, df.shape[0], df.shape[1])

    # Controls
    vcm = read_table(eng, "vw_conceptual_model_adjpd")
    # keep only needed cols to reduce width
    keep_cols = [c for c in ["county_fips", "population", "census_division", "state_abbreviation"] if c in vcm.columns]
    vcm = vcm[keep_cols].copy()
    frames["vw_conceptual_model_adjpd"] = vcm
    logging.info("vw_conceptual_model_adjpd: %s rows", vcm.shape[0])

    # Tech (AHA)
    # prefer *_adjpd; fallback to non‑adjpd
    for tech_table in ["vw_county_tech_summary_adjpd", "vw_county_tech_summary"]:
        try:
            vcts = read_table(eng, tech_table)
            frames["tech"] = vcts
            frames["tech_table_name"] = tech_table
            logging.info("%s: %s rows, %s cols", tech_table, vcts.shape[0], vcts.shape[1])
            break
        except Exception:
            continue
    if "tech" not in frames:
        logging.error("Could not read vw_county_tech_summary_adjpd nor vw_county_tech_summary.")
        sys.exit(1)

    # Optional HRSA rurality
    try:
        hrsa = read_table(eng, "hrsa_health_equity_data")
        frames["hrsa"] = hrsa
        logging.info("hrsa_health_equity_data: %s rows", hrsa.shape[0])
    except Exception:
        logging.info("hrsa_health_equity_data not available; fairness lens will skip.")

    return frames

def resolve_and_merge(fr: Dict[str, pd.DataFrame], year: int) -> pd.DataFrame:
    c1 = fr["chr_analytic_chunk_1"].copy()
    c2 = fr["chr_analytic_chunk_2"].copy()
    c3 = fr["chr_analytic_chunk_3"].copy()

    # Normalize FIPS for CHR
    for d in [c1, c2, c3]:
        fips_col = "_5_digit_fips" if "_5_digit_fips" in d.columns else ("county_fips" if "county_fips" in d.columns else None)
        if fips_col is None:
            logging.error("CHR chunk missing county fips column.")
            sys.exit(1)
        d["county_fips"] = normalize_fips(d[fips_col])
        d.dropna(subset=["county_fips"], inplace=True)

    # Merge CHR chunks on county_fips (inner to keep aligned)
    chr_df = c1.merge(c2, on=["county_fips"], how="inner", suffixes=("", "_c2"))
    chr_df = chr_df.merge(c3, on=["county_fips"], how="inner", suffixes=("", "_c3"))

    # Controls
    vcm = fr["vw_conceptual_model_adjpd"].copy()
    if "county_fips" in vcm.columns:
        vcm["county_fips"] = normalize_fips(vcm["county_fips"])
    chr_df = chr_df.merge(vcm, on="county_fips", how="left")

    # Tech
    tech = fr["tech"].copy()
    fips_col = "county_fips" if "county_fips" in tech.columns else None
    if not fips_col:
        logging.error("Tech table missing county_fips.")
        sys.exit(1)
    tech["county_fips"] = normalize_fips(tech["county_fips"])

    # Robust column resolution for AI/robotics
    def find_col(options: List[str]) -> Optional[str]:
        low = {c.lower(): c for c in tech.columns}
        for o in options:
            if o.lower() in low:
                return low[o.lower()]
        return None

    mo_map: Dict[str, List[str]] = {
        # MO11–MO15 mapping with common variants (+/_adjpd)
        "MO11": ["pct_wfaiart_enabled_adjpd", "pct_wfaiart_enabled", "wfa_art"],  # Automate routine tasks
        "MO12": ["pct_wfaioacw_enabled_adjpd", "pct_wfaioacw_enabled", "wfa_oacw"],  # Optimize workflows
        "MO13": ["pct_wfaippd_enabled_adjpd", "pct_wfaippd_enabled", "wfa_ippd"],  # Predict patient demand
        "MO14": ["pct_wfaipsn_enabled_adjpd", "pct_wfaipsn_enabled", "wfa_ipsn"],  # Predict staffing needs
        "MO15": ["pct_wfaiss_enabled_adjpd",  "pct_wfaiss_enabled",  "wfa_iss"],   # Staff scheduling
        "MO21": ["pct_robohos_enabled_adjpd", "pct_robohos_enabled", "robotics_use"],
    }
    resolved_cols: Dict[str, str] = {}
    for k, opts in mo_map.items():
        col = find_col(opts)
        if col is None:
            logging.warning("Tech column for %s not found (tried %s). It will be missing.", k, opts)
        else:
            resolved_cols[k] = col

    tech_use = tech[["county_fips"] + list(resolved_cols.values())].copy()
    tech_use = tech_use.drop_duplicates(subset=["county_fips"])
    tech_use = tech_use.rename(columns={v: k for k, v in resolved_cols.items()})

    # Optional HRSA rurality (IRR)
    if "hrsa" in fr:
        hrsa = fr["hrsa"].copy()
        code_col = "county_fips_code" if "county_fips_code" in hrsa.columns else None
        if code_col:
            hrsa["county_fips"] = normalize_fips(hrsa[code_col])
            hrsa = hrsa[["county_fips", "irr_county_value"]].copy() if "irr_county_value" in hrsa.columns else hrsa[["county_fips"]]
            chr_df = chr_df.merge(hrsa, on="county_fips", how="left")

    # Merge tech into CHR+controls
    df = chr_df.merge(tech_use, on="county_fips", how="left")

    # Drop duplicated columns that appear after successive merges to keep feature matrix well-defined.
    if df.columns.duplicated().any():
        dup_cols = sorted(set(df.columns[df.columns.duplicated()]))
        logging.warning("Dropping duplicated columns after merge: %s", dup_cols)
        df = df.loc[:, ~df.columns.duplicated()]

    # Keep a simple state key for clustering
    df["state_key"] = df["county_fips"].astype(str).str[:2]
    return df


# ----------------------------- cleaning & feature set -----------------------------

def coerce_numeric_wide(df: pd.DataFrame) -> pd.DataFrame:
    # Coerce all non-ID/object columns to numeric where possible
    non_ids = [c for c in df.columns if c not in ID_COLS and c not in ["census_division", "state_key"]]
    for c in non_ids:
        if df[c].dtype == object:
            df[c] = to_numeric_clean(df[c])
    return df

def resolve_column_with_suffix(df: pd.DataFrame, base_name: str) -> Optional[str]:
    """
    Resolve canonical CHR column names that may appear with merge suffixes (e.g., _c2, _c3).
    Returns the actual dataframe column name or None if not present.
    """
    col_lookup = {c.lower(): c for c in df.columns}
    candidates = [base_name]
    # Allow chunk suffixes applied during merges
    candidates.extend([f"{base_name}_c2", f"{base_name}_c3"])
    base_lower = base_name.lower()
    # Some tables may drop _raw_value suffixes for derived columns
    if base_name.endswith("_raw_value"):
        candidates.append(base_name[:-10])  # remove "_raw_value"
    for cand in candidates:
        actual = col_lookup.get(cand.lower())
        if actual:
            return actual
    # Fallback: partial match across columns if an unexpected suffix was attached
    for c in df.columns:
        if base_lower in c.lower():
            return c
    return None

def assemble_feature_matrix(df: pd.DataFrame, drop_missing_thresh: float,
                            targets: List[str], dv_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    # Candidate features = numeric CHR variables across chunks
    exclude = list(ID_COLS) + ["census_division", "state_key", "population", "log_population"] + targets + dv_cols
    feat_cols = pick_numeric_features(df, exclude=exclude, drop_missing_thresh=drop_missing_thresh)
    return df[feat_cols], feat_cols


# ----------------------------- modeling: direct effects -----------------------------

def make_preproc(numeric_features: List[str]) -> ColumnTransformer:
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler(with_mean=True, with_std=True)),
    ])
    return ColumnTransformer([("num", num_pipe, numeric_features)], remainder="drop")

def one_rep_l1_rf(
    X: pd.DataFrame, y: pd.Series, groups: pd.Series,
    numeric_features: List[str], test_size: float, seed: int
) -> Dict:
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    (tr_idx, te_idx) = next(gss.split(X, y, groups))
    Xtr, Xte = X.iloc[tr_idx], X.iloc[te_idx]
    ytr, yte = y.iloc[tr_idx], y.iloc[te_idx]

    pre = make_preproc(numeric_features)

    # L1-logit (balanced, saga) – we keep n_jobs=1 to avoid nested parallelism
    l1 = Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(
            penalty="l1",
            solver="saga",
            C=CONFIG["LOGIT_C"],
            tol=CONFIG["LOGIT_TOL"],
            class_weight="balanced",
            max_iter=CONFIG["LOGIT_MAX_ITER"],
            n_jobs=1,
            random_state=seed,
        ))
    ])
    l1.fit(Xtr, ytr)
    if hasattr(l1.named_steps["clf"], "predict_proba"):
        p = l1.predict_proba(Xte)[:, 1]
    else:
        p = l1.decision_function(Xte)
        p = (p - p.min()) / (p.max() - p.min() + 1e-9)
    auc = roc_auc_score(yte, p)
    pr = average_precision_score(yte, p)
    bal = balanced_accuracy_score(yte, (p >= 0.5).astype(int))

    # Extract selected features (nonzero coef) in original order
    # After preprocessing, order is preserved for numeric_features
    clf = l1.named_steps["clf"]
    if hasattr(clf, "coef_"):
        coefs = np.asarray(clf.coef_).ravel()
    else:
        coefs = np.zeros(len(numeric_features))
    selected_mask = (np.abs(coefs) > 1e-12)
    signs = np.sign(coefs)

    # RandomForest for complementary importance (n_jobs=1 here)
    rf = Pipeline([
        ("pre", pre),
        ("rf", RandomForestClassifier(n_estimators=500, max_depth=None, min_samples_leaf=2,
                                     class_weight="balanced_subsample", n_jobs=1, random_state=seed))
    ])
    rf.fit(Xtr, ytr)
    rf_imp = rf.named_steps["rf"].feature_importances_

    return {
        "metrics": {"auc": float(auc), "pr_auc": float(pr), "balanced_acc": float(bal)},
        "selected": selected_mask.astype(int).tolist(),
        "signs": signs.tolist(),
        "rf_importance": rf_imp.tolist(),
        "test_index": te_idx.tolist(),
        "preds": p.tolist(),
        "ytest": yte.astype(int).tolist(),
    }

def run_direct_effects(
    X: pd.DataFrame, feat_cols: List[str], df: pd.DataFrame,
    targets: Dict[str, str], outdir: str
):
    groups = group_key(df, CONFIG["CV_MODE"])

    pre = make_preproc(feat_cols)  # for EBM, etc.

    total_targets = len(targets)
    for idx, (target, colname) in enumerate(targets.items(), start=1):
        if colname not in df.columns:
            logging.warning("Skipping %s: column '%s' missing.", target, colname)
            continue

        # build binary label
        y_cont = to_numeric_clean(df[colname]).fillna(0.0)
        if CONFIG["ADOPT_THRESHOLD"] == "zero":
            y = (y_cont > 0.0).astype(int)
            if y.nunique() < 2:
                # fallback: median
                thr = float(np.nanmedian(y_cont.values))
                y = (y_cont > thr).astype(int)
        else:
            thr = float(np.nanmedian(y_cont.values))
            y = (y_cont > thr).astype(int)
        if y.nunique() < 2:
            logging.warning("%s has only one class after thresholding. Skipping.", target)
            continue

        seeds = rng_list(CONFIG["N_REPS"], CONFIG["SEED"] + hash(target) % 10000)

        logging.info("Direct effects %d/%d for %s (%s): %d reps, %d features",
                     idx, total_targets, target, colname, CONFIG["N_REPS"], len(feat_cols))

        reps = Parallel(n_jobs=CONFIG["N_JOBS"], backend="loky", verbose=10)(
            delayed(one_rep_l1_rf)(X[feat_cols], y, groups, feat_cols, CONFIG["TEST_SIZE"], s)
            for s in seeds
        )

        # Aggregate metrics
        mets = pd.DataFrame([r["metrics"] for r in reps])
        mets_summary = mets.agg(["mean", "std"])
        mets_summary.to_csv(os.path.join(outdir, f"metrics_{target}.csv"))

        # Stability selection
        sel_mat = np.vstack([r["selected"] for r in reps])  # n_reps × n_feat
        sign_mat = np.vstack([r["signs"] for r in reps])
        rf_mat = np.vstack([r["rf_importance"] for r in reps])

        freq = sel_mat.mean(axis=0)
        sign_mean = np.nan_to_num(sign_mat, nan=0.0).mean(axis=0)
        rf_imp = rf_mat.mean(axis=0)

        stab = pd.DataFrame({
            "feature": feat_cols,
            "stability_freq": freq,
            "mean_sign": sign_mean,
            "rf_importance_mean": rf_imp,
        }).sort_values(["stability_freq", "rf_importance_mean"], ascending=[False, False])
        stab["stable_core_flag"] = (stab["stability_freq"] >= 0.60).astype(int)
        stab.to_csv(os.path.join(outdir, f"stability_{target}.csv"), index=False)

        # Adoption residuals / surprise (from last rep's test as proxy; or average across reps)
        # We'll average predictions across reps for each county index seen in test folds.
        n = len(df)
        sum_pred = np.zeros(n, dtype=np.float64)
        cnt_pred = np.zeros(n, dtype=np.int32)
        for r in reps:
            te_idx = np.array(r["test_index"], dtype=int)
            pred = np.array(r["preds"], dtype=float)
            sum_pred[te_idx] += pred
            cnt_pred[te_idx] += 1
        avg_pred = np.divide(sum_pred, np.maximum(cnt_pred, 1), out=np.zeros_like(sum_pred), where=np.maximum(cnt_pred, 1) > 0)
        # Re-align y to dataframe index
        y_full = y.reset_index(drop=True).values
        resid = np.where(cnt_pred > 0, y_full - avg_pred, np.nan)
        surprise_q = pd.Series(resid).abs().rank(pct=True)

        resid_df = pd.DataFrame({
            "county_fips": df["county_fips"].values,
            "y_observed": y_full,
            "p_hat": avg_pred,
            "residual": resid,
            "surprise_quantile": surprise_q,
            "population": df.get("population", pd.Series([np.nan]*n)).values,
            "census_division": df.get("census_division", pd.Series([""]*n)).values,
        })
        resid_df.to_csv(os.path.join(outdir, f"adoption_residuals_{target}.csv"), index=False)

        # Fairness lens by rurality (if available)
        if "irr_county_value" in df.columns:
            tmp = resid_df.copy()
            tmp["irr_county_value"] = to_numeric_clean(df["irr_county_value"]).values
            tmp["rural_q"] = pd.qcut(tmp["irr_county_value"], q=4, duplicates="drop")
            fair = tmp.groupby("rural_q")["residual"].agg(["count", "mean", "std"]).reset_index()
            fair.to_csv(os.path.join(outdir, f"fairness_residuals_{target}.csv"), index=False)

        # Optional: Knockoffs
        if KNOCKPY_AVAILABLE:
            try:
                # Knockoffs expects dense numeric X, standardized; we reuse preprocessor to transform
                pre.fit(X[feat_cols])
                Xn = pre.transform(X[feat_cols])
                yv = y.values.astype(int)
                kf = KnockoffFilter(fdr=0.10, knockoff="gaussian")
                kf.fit(X=Xn, y=yv, model="lasso")  # lasso stat by default
                ko = pd.DataFrame({
                    "feature": np.array(feat_cols)[kf.selected],
                    "selected": 1
                })
                ko.to_csv(os.path.join(outdir, f"knockoff_{target}.csv"), index=False)
            except Exception as e:
                logging.warning("Knockoffs failed for %s: %s", target, str(e))

        # Optional: EBM for shapes & pairwise
        if INTERPRET_AVAILABLE:
            try:
                ebm = ExplainableBoostingClassifier(random_state=CONFIG["SEED"], interactions=10, n_jobs=1)
                # Work on simple imputed/scaled X for robustness
                imp = SimpleImputer(strategy="median")
                X_imp = pd.DataFrame(imp.fit_transform(X[feat_cols]), columns=feat_cols, index=X.index)
                ebm.fit(X_imp, y)
                glb = ebm.explain_global()
                names = glb.data()["names"]
                scores = glb.data()["scores"]
                terms = pd.DataFrame({"term": names, "importance": [float(np.abs(np.array(s)).mean()) for s in scores]})
                terms.sort_values("importance", ascending=False).to_csv(os.path.join(outdir, f"ebm_terms_{target}.csv"), index=False)
                # Pairwise terms are already in names with pattern "feat1 x feat2"
                pairs = terms[terms["term"].str.contains(" x ", regex=False)].copy()
                if not pairs.empty:
                    pairs.to_csv(os.path.join(outdir, f"ebm_pairwise_{target}.csv"), index=False)
            except Exception as e:
                logging.warning("EBM failed for %s: %s", target, str(e))


# ----------------------------- modeling: interaction sweep -----------------------------

def rank_chr_for_dv(df: pd.DataFrame, dv_col: str, feat_cols: List[str], top_k: int) -> List[str]:
    # Rank by absolute Pearson r (if numeric) and mutual information proxy (variance filter); simple & fast.
    y = to_numeric_clean(df[dv_col]).values
    ranks = []
    for c in feat_cols:
        x = df[c].values.astype(float)
        # quick guards
        if np.isnan(x).all() or np.nanstd(x) == 0:
            r = 0.0
        else:
            xm = x - np.nanmean(x)
            ym = y - np.nanmean(y)
            num = np.nansum(xm * ym)
            den = np.sqrt(np.nansum(xm * xm) * np.nansum(ym * ym)) + 1e-12
            r = float(abs(num / den))
        ranks.append((c, r))
    ranks.sort(key=lambda z: z[1], reverse=True)
    if top_k and top_k > 0:
        return [c for c, _ in ranks[:top_k]]
    return [c for c, _ in ranks]

def fit_interaction_ols_cluster(
    df: pd.DataFrame, dv: str, x: str, m: str, controls: List[str], cluster_col: str
) -> Tuple[str, float, float, float, float, int, bool]:
    # Build model columns
    data = df[[dv, x, m] + controls + [cluster_col]].copy()
    data[dv] = to_numeric_clean(data[dv])
    data[x]  = to_numeric_clean(data[x])
    data[m]  = to_numeric_clean(data[m])
    data["int"] = data[x] * data[m]

    # Ensure numeric controls
    for c in controls:
        if c not in data.columns:
            data[c] = np.nan
        elif data[c].dtype == object:
            data[c] = to_numeric_clean(data[c])

    # Drop NA rows
    data = data.dropna(subset=[dv, x, m, "int"])
    if data.shape[0] < (len(controls) + 5):
        return x, np.nan, np.nan, np.nan, np.nan, int(data.shape[0]), True

    y = data[dv].astype(float).values
    X = data[["int", x, m] + controls].astype(float)
    X = sm.add_constant(X, has_constant="add")
    clusters = data[cluster_col].astype(str)

    try:
        if clusters.nunique() >= 2:
            res = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": clusters})
        else:
            res = sm.OLS(y, X).fit()
        beta = float(res.params.get("int", np.nan))
        ci_l, ci_u = [float(v) for v in res.conf_int().loc["int"]] if "int" in res.params.index else (np.nan, np.nan)
        p = float(res.pvalues.get("int", np.nan))
        return x, beta, p, ci_l, ci_u, int(data.shape[0]), False
    except Exception:
        return x, np.nan, np.nan, np.nan, np.nan, int(data.shape[0]), True

def run_interactions(
    df: pd.DataFrame, feat_cols: List[str], controls: List[str],
    outdir: str
):
    # Moderators: MO11..MO15, MO21 if present
    moderators = [c for c in ["MO11", "MO12", "MO13", "MO14", "MO15", "MO21"] if c in df.columns]
    if not moderators:
        logging.warning("No moderator columns found; skipping interaction sweep.")
        return

    dv_map = {
        "dv21": "premature_death_raw_value",
        "dv15": "preventable_hospital_stays_raw_value",
    }
    # Keep only present DVs
    dv_map = {k: v for k, v in dv_map.items() if v in df.columns}

    total_dvs = len(dv_map)
    for dv_idx, (dv_key, dv_col) in enumerate(dv_map.items(), start=1):
        cluster_col = "state_key"
        # Choose top‑K CHR candidates
        cand = rank_chr_for_dv(df, dv_col, feat_cols, top_k=CONFIG["INTERACTION_TOP_K"])
        # Remove controls / clustering columns from candidate pool to avoid duplicate-column issues.
        removable = set(controls + [dv_col, cluster_col])
        cand = [c for c in cand if c not in removable]
        if len(cand) < CONFIG["INTERACTION_TOP_K"]:
            logging.info(
                "Interaction sweep %d/%d for %s: %d candidates after dropping controls/duplicates",
                dv_idx, total_dvs, dv_key, len(cand)
            )
        logging.info(
            "Interaction sweep %d/%d for %s (%s): testing %d CHR features × %d moderators",
            dv_idx, total_dvs, dv_key, dv_col, len(cand), len(moderators)
        )
        rows = []
        # Parallelize across (x, m) pairs
        jobs = []
        for m in moderators:
            for x in cand:
                jobs.append((x, m))
        def _run(pair):
            x, m = pair
            return (m,) + fit_interaction_ols_cluster(df, dv_col, x, m, controls, cluster_col)
        outs = Parallel(n_jobs=CONFIG["N_JOBS"], backend="loky", verbose=10)(delayed(_run)(j) for j in jobs)

        for m, x, beta, p, ci_l, ci_u, n, err in outs:
            rows.append({
                "dv": dv_col, "moderator": m, "chr_feature": x,
                "beta_interaction": beta, "p_value": p,
                "ci_low": ci_l, "ci_high": ci_u,
                "n_obs": n, "fit_error": int(err)
            })
        if not rows:
            logging.warning(f"Interaction sweep for {dv_key} produced no results. Skipping CSV output.")
            continue
        res = pd.DataFrame(rows).sort_values(["p_value"], na_position="last")
        res.to_csv(os.path.join(outdir, f"interaction_sweep_{dv_key}.csv"), index=False)

def run_targeted_moderations(
    df: pd.DataFrame,
    controls: List[str],
    outdir: str,
) -> None:
    """
    Evaluate predefined moderation combinations between IV31–IV39, MO11–MO15, and DV15/DV21.
    Persists one CSV per DV containing all tested permutations.
    """
    cluster_col = "state_key"
    if cluster_col not in df.columns:
        logging.warning("Targeted moderation sweep skipped: state_key column missing for clustering.")
        return

    moderators = [m for m in TARGETED_MODERATORS if m in df.columns]
    if not moderators:
        logging.warning("Targeted moderation sweep skipped: none of %s present.", TARGETED_MODERATORS)
        return

    for dv_code, dv_base in TARGETED_DV_MAP.items():
        dv_col = resolve_column_with_suffix(df, dv_base)
        if not dv_col:
            logging.warning("Targeted moderation: DV %s column '%s' not found.", dv_code, dv_base)
            continue

        rows = []
        logging.info(
            "Targeted moderation for %s (%s): testing %d IVs × %d moderators",
            dv_code, dv_col, len(TARGETED_IV_MAP), len(moderators)
        )
        for iv_code, iv_base in TARGETED_IV_MAP.items():
            iv_col = resolve_column_with_suffix(df, iv_base)
            if not iv_col:
                logging.warning("Targeted moderation: IV %s column '%s' not found.", iv_code, iv_base)
                continue
            for mod in moderators:
                _, beta, p, ci_l, ci_u, n_obs, fit_err = fit_interaction_ols_cluster(
                    df, dv_col, iv_col, mod, controls, cluster_col
                )
                rows.append({
                    "dv_code": dv_code,
                    "dv_column": dv_col,
                    "iv_code": iv_code,
                    "iv_column": iv_col,
                    "moderator": mod,
                    "beta_interaction": beta,
                    "p_value": p,
                    "ci_low": ci_l,
                    "ci_high": ci_u,
                    "n_obs": n_obs,
                    "fit_error": int(fit_err),
                })

        if not rows:
            logging.warning("Targeted moderation for %s produced no rows; skipping output.", dv_code)
            continue

        out_df = pd.DataFrame(rows).sort_values(["p_value"], na_position="last")
        out_path = os.path.join(outdir, f"targeted_moderation_{dv_code.lower()}.csv")
        out_df.to_csv(out_path, index=False)
        logging.info("Targeted moderation results written to %s", out_path)

def validate_data_quality(df: pd.DataFrame, critical_cols: List[str]) -> None:
    """Log data quality metrics for critical columns."""
    for col in critical_cols:
        if col not in df.columns:
            logging.warning(f"Critical column '{col}' missing from dataset")
            continue
            
        # Check for missing values
        missing = df[col].isna().sum()
        missing_pct = missing / len(df) * 100 if len(df) > 0 else 0
        
        # For numeric columns
        if pd.api.types.is_numeric_dtype(df[col]):
            unique = df[col].nunique()
            zeros = (df[col] == 0).sum()
            zeros_pct = zeros / len(df) * 100 if len(df) > 0 else 0
            
            logging.info(
                f"Column '{col}': {len(df)-missing}/{len(df)} values ({missing_pct:.1f}% missing), "
                f"{unique} unique values, {zeros} zeros ({zeros_pct:.1f}%)"
            )
        # For string columns (like FIPS)
        elif pd.api.types.is_string_dtype(df[col]):
            unique = df[col].nunique()
            empty = (df[col] == '').sum()
            empty_pct = empty / len(df) * 100 if len(df) > 0 else 0
            
            logging.info(
                f"Column '{col}': {len(df)-missing}/{len(df)} values ({missing_pct:.1f}% missing), "
                f"{unique} unique values, {empty} empty strings ({empty_pct:.1f}%)"
            )

# ----------------------------- main -----------------------------

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    setup_logging(args.outdir)
    warnings.filterwarnings("ignore")

    # Connect & fetch
    eng = build_engine()
    frames = fetch_all_frames(eng, CONFIG["YEAR"])
    df = resolve_and_merge(frames, CONFIG["YEAR"])

    # Coerce numerics and build controls
    df = coerce_numeric_wide(df)
    df, controls = add_controls(df)

    # Validate data quality for critical columns
    validate_data_quality(df, [
        "county_fips", "population", "log_population", 
        "premature_death_raw_value", "preventable_hospital_stays_raw_value"
    ] + [c for c in ["MO11", "MO12", "MO13", "MO14", "MO15", "MO21"] if c in df.columns])

    # Identify targets present
    targets: Dict[str, str] = {k: k for k in ["MO11", "MO12", "MO13", "MO14", "MO15", "MO21"] if k in df.columns}
    if not targets:
        logging.error("No MO11..MO15/MO21 columns present after merge. Check tech table.")
        sys.exit(1)

    # Build feature matrix (exclude targets and DVs)
    dv_cols = [c for c in ["premature_death_raw_value", "preventable_hospital_stays_raw_value"] if c in df.columns]
    X_all, feat_cols = assemble_feature_matrix(df, CONFIG["DROP_MISSING_THRESH"], list(targets.keys()), dv_cols)

    # Persist dataset snapshot
    snap = df[["county_fips"] + feat_cols + list(targets.keys()) + dv_cols + ["population", "census_division"]].copy()
    snap.to_parquet(os.path.join(args.outdir, "dataset_snapshot.parquet"), index=False)
    with open(os.path.join(args.outdir, "run_config.json"), "w") as f:
        json.dump({**CONFIG, "outdir": args.outdir}, f, indent=2)

    logging.info("Final modeling frame: %d counties, %d CHR features", X_all.shape[0], len(feat_cols))

    # ----- Direct effects -----
    run_direct_effects(X_all, feat_cols, df, targets, args.outdir)

    # ----- Interaction sweep -----
    run_interactions(df, feat_cols, controls, args.outdir)

    # ----- Targeted moderation grid (IV31–IV39 × MO11–MO15 on DV15/DV21) -----
    run_targeted_moderations(df, controls, args.outdir)

    logging.info("Done. Outputs written to %s", args.outdir)


if __name__ == "__main__":
    main()
