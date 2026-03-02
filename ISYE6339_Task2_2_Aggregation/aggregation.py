import re
from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# CONFIG
# ============================================================

# 你的数据位置（GitHub / 本地都可以改这里）
INPUT_ROOT = Path("data")

OUT_DIR = Path("_aggregated_one_plan")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 选择规划规则
PLANNING_RULE = "MAX"   # 可改成 "MEAN"

Q_LO, Q_HI = 0.05, 0.95


# ============================================================
# Helpers
# ============================================================

def infer_run_id(fp: Path) -> str:
    parent = fp.parent.name
    if re.match(r"run\d+", parent, flags=re.IGNORECASE):
        return parent.lower()

    m = re.search(r"(run\d+)", fp.name, flags=re.IGNORECASE)
    if m:
        return m.group(1).lower()

    return fp.parent.name.lower()


def read_many(pattern: str) -> pd.DataFrame:
    files = list(INPUT_ROOT.rglob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")

    dfs = []
    for fp in files:
        df = pd.read_csv(fp)
        df["run_id"] = infer_run_id(fp)
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def summarize_across_runs(df: pd.DataFrame, group_keys: list, value_cols: list) -> pd.DataFrame:
    for c in value_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    g = df.groupby(group_keys, dropna=False)

    out = g[value_cols].agg(["mean", "std", "min", "max"]).reset_index()

    out.columns = [
        c[0] if c[1] == "" else f"{c[0]}__{c[1]}"
        for c in out.columns.to_flat_index()
    ]

    qlo = g[value_cols].quantile(Q_LO).reset_index()
    qhi = g[value_cols].quantile(Q_HI).reset_index()

    for c in value_cols:
        qlo.rename(columns={c: f"{c}__p05"}, inplace=True)
        qhi.rename(columns={c: f"{c}__p95"}, inplace=True)

    out = out.merge(qlo, on=group_keys, how="left")
    out = out.merge(qhi, on=group_keys, how="left")

    n_runs = g["run_id"].nunique().reset_index(name="n_runs")
    out = out.merge(n_runs, on=group_keys, how="left")

    return out


def build_planning_table(summary_df: pd.DataFrame, group_keys: list, base_cols: list) -> pd.DataFrame:
    pick = "mean" if PLANNING_RULE.upper() == "MEAN" else "max"

    keep_cols = group_keys + ["n_runs"]

    for c in base_cols:
        col = f"{c}__{pick}"
        if col in summary_df.columns:
            keep_cols.append(col)

    plan = summary_df[keep_cols].copy()

    rename_map = {
        f"{c}__{pick}": f"{c}_PLAN"
        for c in base_cols
        if f"{c}__{pick}" in plan.columns
    }

    plan.rename(columns=rename_map, inplace=True)
    return plan


# ============================================================
# 1) MarketYear
# ============================================================

print("Processing MarketYear...")

my = read_many("*MarketYear_OTDReachable_Summary*.csv")

MY_KEYS = ["Year","Country","RegionType","MarketKey","Scenario","Model"]
VAL_COLS_MY = ["BaseUnits","ReachableUnits","BaseRevenue_EUR","ReachableRevenue_EUR"]

my_summary = summarize_across_runs(my, MY_KEYS, VAL_COLS_MY)
my_plan = build_planning_table(my_summary, MY_KEYS, VAL_COLS_MY)

my_summary.to_csv(OUT_DIR / "MarketYear__AGG_STATS.csv", index=False)
my_plan.to_csv(OUT_DIR / f"MarketYear__ONEPLAN_{PLANNING_RULE}.csv", index=False)


# ============================================================
# 2) Year Summary
# ============================================================

print("Processing Year...")

yy = read_many("*Year_OTDReachable_Summary*.csv")

YY_KEYS = ["Year","Scenario"]
VAL_COLS_YY = ["BaseUnits","ReachableUnits","BaseRevenue_EUR","ReachableRevenue_EUR"]

yy_summary = summarize_across_runs(yy, YY_KEYS, VAL_COLS_YY)
yy_plan = build_planning_table(yy_summary, YY_KEYS, VAL_COLS_YY)

yy_summary.to_csv(OUT_DIR / "Year__AGG_STATS.csv", index=False)
yy_plan.to_csv(OUT_DIR / f"Year__ONEPLAN_{PLANNING_RULE}.csv", index=False)


# ============================================================
# 3) Daily
# ============================================================

print("Processing Daily...")

daily = read_many("*Daily_OTDReachable_*.csv")

if "Date" in daily.columns:
    daily["Date"] = pd.to_datetime(daily["Date"], errors="coerce")

DAILY_KEYS = ["Date","Year","Country","RegionType","MarketKey","Scenario","Model"]
VAL_COLS_D = ["BaseDailyUnits","ReachableDailyUnits","BaseDailyRevenue_EUR","ReachableDailyRevenue_EUR"]

daily_summary = summarize_across_runs(daily, DAILY_KEYS, VAL_COLS_D)
daily_plan = build_planning_table(daily_summary, DAILY_KEYS, VAL_COLS_D)

daily_summary.to_csv(OUT_DIR / "Daily__AGG_STATS.csv", index=False)
daily_plan.to_csv(OUT_DIR / f"Daily__ONEPLAN_{PLANNING_RULE}.csv", index=False)

print("DONE")