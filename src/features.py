"""
Feature engineering for the Sri Lanka dengue outbreak early-warning model.

The SAME functions are used by the training notebook and by the Streamlit
front end, so the features seen at inference time are identical to the ones
the model was trained on (no training/serving skew).

Data: weekly dengue cases reported by the Epidemiology Unit, Ministry of
Health, Sri Lanka, for the 26 RDHS (Regional Director of Health Services)
divisions, Dec 2006 - May 2026 (compiled in the `denguedatahub` R package).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Domain knowledge: administrative province + agro-climatic zone per RDHS.
# Wet zone gets rain from the SW monsoon (May-Sep); dry zone mainly from the
# NE monsoon (Dec-Feb); the intermediate zone receives both.
# Kalmune is a separate RDHS carved out of Ampara district.
# --------------------------------------------------------------------------
DISTRICT_META = {
    "Colombo":      ("Western",       "Wet"),
    "Gampaha":      ("Western",       "Wet"),
    "Kalutara":     ("Western",       "Wet"),
    "Kandy":        ("Central",       "Wet"),
    "Matale":       ("Central",       "Intermediate"),
    "NuwaraEliya":  ("Central",       "Wet"),
    "Galle":        ("Southern",      "Wet"),
    "Matara":       ("Southern",      "Wet"),
    "Hambanthota":  ("Southern",      "Dry"),
    "Jaffna":       ("Northern",      "Dry"),
    "Kilinochchi":  ("Northern",      "Dry"),
    "Mannar":       ("Northern",      "Dry"),
    "Vavuniya":     ("Northern",      "Dry"),
    "Mullaitivu":   ("Northern",      "Dry"),
    "Batticaloa":   ("Eastern",       "Dry"),
    "Ampara":       ("Eastern",       "Dry"),
    "Kalmune":      ("Eastern",       "Dry"),
    "Trincomalee":  ("Eastern",       "Dry"),
    "Kurunegala":   ("North Western", "Intermediate"),
    "Puttalam":     ("North Western", "Dry"),
    "Anuradhapura": ("North Central", "Dry"),
    "Polonnaruwa":  ("North Central", "Dry"),
    "Badulla":      ("Uva",           "Intermediate"),
    "Monaragala":   ("Uva",           "Dry"),
    "Ratnapura":    ("Sabaragamuwa",  "Wet"),
    "Kegalle":      ("Sabaragamuwa",  "Wet"),
}
DISTRICTS = sorted(DISTRICT_META)

CAT_FEATURES = ["district", "province", "climate_zone", "monsoon_season"]
NUM_FEATURES = [
    "week_sin", "week_cos",
    "log_c0", "log_c1", "log_c2", "log_c3", "log_c52",
    "log_roll4", "log_roll8", "std_log4",
    "growth_1w", "growth_4w",
    "ratio_thr", "roll4_ratio_thr", "log_thr",
    "log_nat0", "nat_growth_1w",
]
FEATURES = CAT_FEATURES + NUM_FEATURES
TARGET = "outbreak_next_week"


def monsoon_season(week: int) -> str:
    """Map ISO-like week of year to Sri Lanka's four climatic seasons."""
    w = int(week)
    if w >= 49 or w <= 8:
        return "NE_monsoon"          # Dec - Feb
    if 9 <= w <= 18:
        return "First_inter_monsoon"  # Mar - Apr
    if 19 <= w <= 39:
        return "SW_monsoon"          # May - Sep
    return "Second_inter_monsoon"     # Oct - Nov


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["start.date"], format="%m/%d/%Y")
    df["year"] = df["year"].astype(int)
    df["week"] = df["week"].astype(int)
    df = df.rename(columns={"cases": "cases"})
    return df[["date", "year", "week", "district", "cases"]].sort_values(["district", "date"]).reset_index(drop=True)


def clean_implausible_dips(df: pd.DataFrame, min_neighbour: float = 50, ratio: float = 0.15):
    """Detect isolated, epidemiologically implausible one-week collapses
    (e.g. Colombo 316 -> 9 -> 291), which are almost certainly transcription /
    PDF-scraping errors in the WER (a dropped digit), not real transmission.
    Rule: both neighbouring weeks >= `min_neighbour` cases and the week itself
    < `ratio` x the smaller neighbour. Flagged values are replaced by the
    geometric mean of the two neighbours (interpolation in log space).
    Returns (clean_df, flagged_rows)."""
    df = df.sort_values(["district", "date"]).copy()
    g = df.groupby("district")["cases"]
    prev, nxt = g.shift(1), g.shift(-1)
    mask = (prev >= min_neighbour) & (nxt >= min_neighbour) & (df["cases"] < ratio * np.minimum(prev, nxt))
    flagged = df.loc[mask, ["district", "date", "cases"]].assign(prev_week=prev[mask], next_week=nxt[mask])
    df.loc[mask, "cases"] = np.round(np.expm1((np.log1p(prev[mask]) + np.log1p(nxt[mask])) / 2))
    flagged["imputed"] = df.loc[mask, "cases"]
    return df.reset_index(drop=True), flagged.reset_index(drop=True)


def district_thresholds(df: pd.DataFrame, train_end: str, q: float = 0.75) -> dict:
    """District-specific 'high transmission' threshold = q-quantile of weekly
    cases in the TRAINING period only (prevents target leakage)."""
    tr = df[df["date"] < pd.Timestamp(train_end)]
    return tr.groupby("district")["cases"].quantile(q).clip(lower=1.0).to_dict()


def build_panel_features(df: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    """Vectorised feature construction on the full district x week panel.
    Row t uses only information available at the end of week t; the target is
    whether week t+1 exceeds the district threshold."""
    df = df.sort_values(["district", "date"]).copy()
    nat = df.groupby("date")["cases"].sum().rename("nat0")
    nat = nat.to_frame()
    nat["nat1"] = nat["nat0"].shift(1)
    df = df.merge(nat, left_on="date", right_index=True, how="left")

    g = df.groupby("district")["cases"]
    for k in range(8):
        df[f"c{k}"] = g.shift(k)
    df["c52"] = g.shift(52)
    df["next_cases"] = g.shift(-1)
    df["thr"] = df["district"].map(thresholds)

    out = _derive(df)
    out["next_cases"] = df["next_cases"].values
    out[TARGET] = np.where(df["next_cases"].isna(), np.nan,
                           (df["next_cases"] >= df["thr"]).astype(float))
    for col in ["date", "year", "cases", "thr"]:
        out[col] = df[col].values
    return out


def features_from_inputs(district: str, week: int, last8: list, same_week_last_year,
                         national_this_week, national_last_week, thresholds: dict) -> pd.DataFrame:
    """Build ONE feature row from raw inputs typed into the front end.
    last8 = [cases this week, 1 week ago, ..., 7 weeks ago]."""
    row = {"district": district, "week": week}
    for k in range(8):
        v = last8[k] if k < len(last8) else np.nan
        row[f"c{k}"] = np.nan if v is None else float(v)
    row["c52"] = np.nan if same_week_last_year is None else float(same_week_last_year)
    row["nat0"] = np.nan if national_this_week is None else float(national_this_week)
    row["nat1"] = np.nan if national_last_week is None else float(national_last_week)
    row["thr"] = thresholds[district]
    return _derive(pd.DataFrame([row]))


def _derive(df: pd.DataFrame) -> pd.DataFrame:
    """Shared maths for both panel and single-row feature construction."""
    out = pd.DataFrame(index=df.index)
    out["district"] = df["district"].values
    out["province"] = df["district"].map(lambda d: DISTRICT_META[d][0]).values
    out["climate_zone"] = df["district"].map(lambda d: DISTRICT_META[d][1]).values
    out["monsoon_season"] = df["week"].map(monsoon_season).values
    out["week"] = df["week"].values
    ang = 2 * np.pi * df["week"].astype(float) / 52.0
    out["week_sin"], out["week_cos"] = np.sin(ang), np.cos(ang)

    logs = {k: np.log1p(df[f"c{k}"].astype(float)) for k in range(8)}
    for k in range(4):
        out[f"log_c{k}"] = logs[k]
    out["log_c52"] = np.log1p(df["c52"].astype(float))

    raw4 = df[[f"c{k}" for k in range(4)]].astype(float)
    raw8 = df[[f"c{k}" for k in range(8)]].astype(float)
    prev4 = df[[f"c{k}" for k in range(4, 8)]].astype(float)
    out["log_roll4"] = np.log1p(raw4.mean(axis=1, skipna=False))
    out["log_roll8"] = np.log1p(raw8.mean(axis=1, skipna=False))
    out["std_log4"] = pd.concat([logs[k] for k in range(4)], axis=1).std(axis=1, ddof=0, skipna=False)
    out["growth_1w"] = logs[0] - logs[1]
    out["growth_4w"] = out["log_roll4"] - np.log1p(prev4.mean(axis=1, skipna=False))

    thr = df["thr"].astype(float)
    out["ratio_thr"] = df["c0"].astype(float) / thr
    out["roll4_ratio_thr"] = raw4.mean(axis=1, skipna=False) / thr
    out["log_thr"] = np.log1p(thr)

    out["log_nat0"] = np.log1p(df["nat0"].astype(float))
    out["nat_growth_1w"] = out["log_nat0"] - np.log1p(df["nat1"].astype(float))
    return out
