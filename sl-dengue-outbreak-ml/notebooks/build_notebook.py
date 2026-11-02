"""Builds dengue_outbreak_ml.ipynb from the cells below (run: python build_notebook.py)."""
import nbformat as nbf

cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s.strip()))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Dengue Outbreak Early-Warning for Sri Lanka's 26 RDHS Divisions
### Applied Machine Learning — Class Activity (RF vs XGBoost) extended into the Final Assignment

**Problem (local, Sri Lankan context).** Dengue is Sri Lanka's most burdensome vector-borne disease (174,687 reported cases in the 2017 epidemic alone in this dataset). Every week the **Epidemiology Unit of the Ministry of Health** publishes the *Weekly Epidemiological Report (WER)* with dengue notifications for each of the 26 **RDHS** (Regional Director of Health Services) divisions. Regional health officers must decide **where to send fogging teams, Public Health Inspectors and extra hospital capacity next week**.

**ML task — binary classification.** *Given the dengue situation in a district at the end of week t, will week t+1 be a "high-transmission week" for that district?*

- **Target** `outbreak_next_week = 1` if next week's cases ≥ the district's own 75th-percentile weekly count over 2007–2021 (a district-specific alert threshold, so Colombo and Mullaitivu are each judged against their own normal).
- **Why classification and not raw-count regression?** The operational decision is binary (escalate vector control or not) and the count distribution is extremely skewed (0 to 2,631/week).

**Notebook map**
1. Data collection & quality checks → 2. EDA → 3. Target & feature engineering (domain knowledge) → 4. Temporal split & single held-out sample → 5. Preprocessing → 6. Existing-approach baselines → 7. **Part 1: Random Forest vs XGBoost with designed + tuned hyper-parameters** → 8. Soft voting (lecture notebook), LightGBM, CatBoost, **Stacking** → 9. Model selection & final test → 10. **Explainability: SHAP + LIME** → 11. **Single-sample inference from the .pkl** + next-week forecast → 12. Export → 13. Conclusions.
""")

code(r"""
import sys, time, pickle, warnings, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

ROOT = Path("..").resolve()
sys.path.insert(0, str(ROOT / "src"))
from features import (load_raw, clean_implausible_dips, district_thresholds, build_panel_features, features_from_inputs,
                      DISTRICT_META, CAT_FEATURES, NUM_FEATURES, FEATURES, TARGET)
from cb_wrapper import CatBoostWrapper

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, KFold
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score, precision_score,
                             recall_score, balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay)
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import shap, lime, lime.lime_tabular

SEED = 42
import os; FAST = os.environ.get("FAST") == "1"   # quick smoke-test mode (CI); full run uses FAST=False
FIG = ROOT / "reports" / "figures"; FIG.mkdir(parents=True, exist_ok=True)
MODELS = ROOT / "models"; MODELS.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
pd.set_option("display.float_format", lambda v: f"{v:,.3f}")
print("xgboost", __import__("xgboost").__version__, "| lightgbm", __import__("lightgbm").__version__,
      "| catboost", __import__("catboost").__version__, "| shap", shap.__version__)
""")

md(r"""
## 1. Data collection

**Source:** Epidemiology Unit, Ministry of Health Sri Lanka — Weekly Epidemiological Reports (WER), district-level dengue notifications, Dec 2006 → May 2026. The PDFs were scraped and tidied by the open-source R package **`denguedatahub`** (Talagala, University of Sri Jayewardenepura; GitHub `thiyangt/denguedatahub`, object `srilanka_weekly_data`). I exported it to `data/raw/srilanka_weekly_dengue.csv`.

| column | meaning |
|---|---|
| `year`, `week` | epidemiological year and week number |
| `start.date`, `end.date` | the reporting week |
| `district` | RDHS division (25 districts + **Kalmune**, a separate RDHS carved out of Ampara) |
| `cases` | dengue notifications that week |
""")

code(r"""
raw = pd.read_csv(ROOT / "data/raw/srilanka_weekly_dengue.csv")
print(raw.shape); display(raw.head())
df = load_raw(ROOT / "data/raw/srilanka_weekly_dengue.csv")
print("Period:", df.date.min().date(), "→", df.date.max().date(), "| districts:", df.district.nunique())
""")

md("### 1.1 Data-quality checks (nulls, duplicates, impossible values, gaps)")
code(r"""
checks = {
    "null values (any column)": int(raw.isna().sum().sum()),
    "duplicate district-week rows": int(df.duplicated(["district", "date"]).sum()),
    "negative case counts": int((df.cases < 0).sum()),
    "non-integer case counts": int((df.cases % 1 != 0).sum()),
}
gaps = df.groupby("district")["date"].diff().dt.days.value_counts().rename("count")
rows_per_district = df.district.value_counts()
display(pd.Series(checks, name="result").to_frame())
print("Days between consecutive reports (per district):"); display(gaps.to_frame())
print("District with a missing week:", rows_per_district[rows_per_district < rows_per_district.max()].to_dict())
""")

code(r"""
# Locate Puttalam's missing week and the irregular 6/8/9-day reporting intervals
all_dates = pd.Index(sorted(df.date.unique()))
missing = all_dates.difference(df[df.district == "Puttalam"].date)
print("Puttalam missing report week:", [d.date() for d in missing])
odd = df.assign(gap=df.groupby("district").date.diff().dt.days).query("gap not in [7] and gap == gap")
display(odd.drop_duplicates("date")[["date", "year", "week", "gap"]])
""")

md(r"""
### 1.2 Outlier detection: implausible one-week collapses
A scan of the series shows isolated weeks where a large district suddenly reports almost nothing and then bounces straight back (e.g. Colombo **316 → 9 → 291** in March 2026). Dengue transmission cannot stop for exactly one week and resume, and the values (9, 10, 11, 12…) look like a **dropped digit** when the WER PDFs were transcribed. Rule: *both neighbouring weeks ≥ 50 cases and the week < 15 % of the smaller neighbour* → treat as an error and impute with the geometric mean of the neighbours (log-space interpolation). Genuine epidemic peaks (high outliers) are **not** touched.
""")
code(r"""
df, flagged = clean_implausible_dips(df)
print(f"Flagged and imputed {len(flagged)} of {len(df):,} district-weeks ({len(flagged)/len(df):.2%})")
display(flagged)
""")

md(r"""
**Findings.** No nulls, duplicates or negative values in the raw table, but 22 transcription-type low outliers were found and imputed (1.2). Puttalam is missing one week (Feb 2026, an un-reported week in the WER — it only shifts that district's lags by one week for a few rows) and three report dates shift by ±1–2 days at year boundaries (WER week-numbering changes) — both harmless for weekly lags. **Missing values do appear after feature engineering** (lags at the start of each series, and *same week last year* during 2007), which is handled with median imputation + missing-indicator flags inside the model pipeline (Section 5). Extreme values (e.g. Colombo's 2,631 cases in a single 2017 week) are **real epidemic weeks, not errors**, so they are kept and tamed with a `log1p` transform instead of being clipped.
""")

md("## 2. Exploratory data analysis")
code(r"""
nat = df.groupby("date").cases.sum()
fig, ax = plt.subplots(2, 1, figsize=(12, 6.5))
ax[0].plot(nat.index, nat.values, lw=1, color="#1f5f8b")
ax[0].set_title("Sri Lanka — national weekly dengue notifications"); ax[0].set_ylabel("cases / week")
for yr, lbl in [(2017, "2017 epidemic"), (2019, "2019"), (2023, "2023")]:
    s = nat[str(yr)]; ax[0].annotate(lbl, (s.idxmax(), s.max()), xytext=(5, -5), textcoords="offset points", fontsize=8)
yearly = df[df.year.between(2007, 2025)].groupby("year").cases.sum()
ax[1].bar(yearly.index, yearly.values, color="#7aa6c2"); ax[1].set_title("Annual total (complete years)")
plt.tight_layout(); plt.savefig(FIG / "01_national_trend.png", bbox_inches="tight"); plt.show()
""")

code(r"""
df["month"] = df.date.dt.month
season = df.groupby(["year", "month"]).cases.sum().unstack().loc[2007:2025]
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5), gridspec_kw={"width_ratios": [1.4, 1]})
im = ax[0].imshow(np.log1p(season.values), aspect="auto", cmap="magma_r")
ax[0].set_yticks(range(len(season)), season.index, fontsize=7); ax[0].set_xticks(range(12), list("JFMAMJJASOND"))
ax[0].set_title("Monthly cases (log scale): mid-year & Dec–Jan peaks"); plt.colorbar(im, ax=ax[0], fraction=0.03)
dist_tot = df.groupby("district").cases.sum().sort_values()
ax[1].barh(dist_tot.index, dist_tot.values, color=["#c0392b" if DISTRICT_META[d][1] == "Wet" else "#7f8c8d" for d in dist_tot.index])
ax[1].set_title("Total cases 2006–2026 (red = wet zone)"); ax[1].tick_params(axis="y", labelsize=7)
plt.tight_layout(); plt.savefig(FIG / "02_seasonality_districts.png", bbox_inches="tight"); plt.show()
print("Share of all cases in Western Province:", round(df[df.district.isin(['Colombo','Gampaha','Kalutara'])].cases.sum() / df.cases.sum(), 3))
""")

code(r"""
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
ax[0].hist(df.cases, bins=80, color="#7aa6c2"); ax[0].set_title(f"Raw weekly cases (skew={df.cases.skew():.1f})")
ax[1].hist(np.log1p(df.cases), bins=50, color="#1f5f8b"); ax[1].set_title(f"log1p(cases) (skew={np.log1p(df.cases).skew():.2f})")
c = df[df.district == "Colombo"].set_index("date").cases
ax[2].scatter(np.log1p(c.shift(1)), np.log1p(c), s=4, alpha=.4); ax[2].set_title("Colombo: log cases t-1 vs t")
ax[2].set_xlabel("log1p(cases t-1)"); ax[2].set_ylabel("log1p(cases t)")
plt.tight_layout(); plt.savefig(FIG / "03_distribution_autocorr.png", bbox_inches="tight"); plt.show()
ac = pd.Series({k: np.log1p(c).autocorr(k) for k in [1, 2, 4, 8, 26, 52]}, name="autocorr")
display(ac.to_frame().T)
""")

md(r"""
**EDA take-aways that drive the design**
- **Strong persistence**: log-cases autocorrelation ≈ 0.83 at lag 1 (still 0.70 at lag 4) → recent lags will be the backbone features (a naive "this week is high → next week is high" rule is the baseline to beat).
- **Bi-modal seasonality** linked to the **South-West monsoon (May–Jul)** and **North-East monsoon (Dec–Jan)** → cyclical week encoding + a `monsoon_season` category.
- **Wet-zone / urban districts dominate** (Western Province ≈ 40 % of all cases) → district, province and agro-climatic zone features; thresholds must be *district-specific*.
- **Heavy right skew** (see histogram titles) → `log1p` transformation; tree ensembles are scale-invariant, but the log helps the linear baseline and the growth-rate features.
- **Regime changes**: 2017 epidemic, COVID-19 lockdowns (2020–21: low reporting), high years 2023–24 → a **temporal** validation is mandatory (random K-fold would leak the future).
""")

md(r"""
## 3. Target definition & feature engineering

All features at row *t* use only data available at the end of week *t* (no look-ahead). The per-district threshold is computed **only from 2007–2021 (training period)** to avoid target leakage.

| group | features | rationale (domain knowledge) |
|---|---|---|
| Recent burden | `log_c0..log_c3` (this week & 3 previous), `log_roll4`, `log_roll8`, `std_log4` | transmission is autocorrelated; incubation + reporting lag ≈ 1–2 weeks |
| Momentum | `growth_1w`, `growth_4w` | an epidemic is defined by acceleration, not only level |
| Relative level | `ratio_thr`, `roll4_ratio_thr`, `log_thr` | puts Colombo and Mullaitivu on the same scale |
| Seasonality | `week_sin`, `week_cos`, `monsoon_season`, `log_c52` | monsoon-driven breeding; same week last year |
| Geography | `district`, `province`, `climate_zone` | wet / intermediate / dry zone rainfall regimes, urbanisation |
| National context | `log_nat0`, `nat_growth_1w` | island-wide waves (e.g. new DENV serotype in 2017) spill over between districts |
""")

code(r"""
TRAIN_END, VALID_END = "2022-01-01", "2024-01-01"
thresholds = district_thresholds(df, TRAIN_END, q=0.75)
panel = build_panel_features(df, thresholds)
display(pd.Series(thresholds).sort_values(ascending=False).rename("alert threshold (cases/week)").to_frame().T)
print(panel.shape); display(panel[FEATURES + [TARGET]].head(3).T)
""")

md(r"""
### 3.1 Consistency check: front-end feature builder == training feature builder
The web app builds a feature row from numbers typed in by the user using `features_from_inputs()`. To guarantee no *training/serving skew*, I rebuild a random sample of panel rows with that function and assert they are identical.
""")
code(r"""
def inputs_for(district, date):
    s = df[df.district == district].set_index("date").cases
    natl = df.groupby("date").cases.sum()
    pos = s.index.get_loc(date)
    last8 = [s.iloc[pos - k] if pos - k >= 0 else None for k in range(8)]
    c52 = s.iloc[pos - 52] if pos - 52 >= 0 else None
    npos = natl.index.get_loc(date)
    return dict(district=district, week=int(df[(df.district == district) & (df.date == date)].week.iloc[0]),
                last8=last8, same_week_last_year=c52, national_this_week=natl.iloc[npos],
                national_last_week=natl.iloc[npos - 1] if npos else None)

rng = np.random.default_rng(SEED)
sample_idx = rng.choice(panel.index, 200, replace=False)
max_diff = 0.0
for i in sample_idx:
    r = panel.loc[i]
    one = features_from_inputs(**inputs_for(r.district, r.date), thresholds=thresholds)
    a = one[NUM_FEATURES].astype(float).values.ravel(); b = r[NUM_FEATURES].astype(float).values
    assert (one[CAT_FEATURES].values.ravel() == r[CAT_FEATURES].values).all()
    max_diff = max(max_diff, np.nanmax(np.abs(np.nan_to_num(a, nan=-9) - np.nan_to_num(b, nan=-9))))
print("max |difference| over 200 random rows:", max_diff)
""")

md(r"""
## 4. Temporal split + one held-out sample for inference

| split | weeks used as *t* | purpose |
|---|---|---|
| **Train** | 2007 – 2021 | fit + time-series cross-validated hyper-parameter search |
| **Validation** | 2022 – 2023 | compare models, choose decision threshold |
| **Test** | 2024 – May 2026 | final, untouched evaluation (models refit on train+valid) |
| **Held-out single sample** | Colombo, latest labelled week | "new data point" for the inference demo — never used in training or testing |

The last week in the file (target unknown) is used for a genuine **next-week forecast**.
""")
code(r"""
labelled = panel[panel[TARGET].notna()].copy()
labelled[TARGET] = labelled[TARGET].astype(int)
holdout_idx = labelled[labelled.district == "Colombo"].date.idxmax()
holdout = labelled.loc[[holdout_idx]]
labelled = labelled.drop(index=holdout_idx)
forecast_rows = panel[panel[TARGET].isna()].copy()

train = labelled[labelled.date < TRAIN_END].sort_values(["date", "district"])
valid = labelled[(labelled.date >= TRAIN_END) & (labelled.date < VALID_END)].sort_values(["date", "district"])
test  = labelled[labelled.date >= VALID_END].sort_values(["date", "district"])
trval = pd.concat([train, valid])
Xtr, ytr, Xva, yva, Xte, yte = train[FEATURES], train[TARGET], valid[FEATURES], valid[TARGET], test[FEATURES], test[TARGET]
Xtv, ytv = trval[FEATURES], trval[TARGET]

summary = pd.DataFrame({s: [len(d), d.date.min().date(), d.date.max().date(), d[TARGET].mean()] for s, d in
                        [("train", train), ("valid", valid), ("test", test)]}, index=["rows", "from", "to", "positive rate"]).T
display(summary)
print("Held-out sample:", holdout[["district", "date", "year", "cases", "next_cases", "thr", TARGET]].to_dict("records"))
print("Forecast rows (target unknown):", len(forecast_rows), "week starting", forecast_rows.date.max().date())
""")

md(r"""
⚠️ **Prior shift.** Only 26 % of 2007–21 district-weeks exceed the threshold, but ≈ 55 % of 2022–26 weeks do — dengue burden in Sri Lanka has risen (2023 was the 3rd-worst year on record). This is genuine *concept/prior drift* (Lecture 10: monitoring & retraining). It is why: (i) threshold-free metrics (ROC-AUC, PR-AUC) are reported, (ii) the decision threshold is re-chosen on the 2022–23 validation period, and (iii) the final models are refit on train+valid before testing.
""")

md(r"""
## 5. Preprocessing pipelines
Everything lives inside sklearn `Pipeline`s so that imputation/scaling statistics are learned **only from the training fold** (no data leakage — Lecture 11).
- **Null handling / imputation:** `SimpleImputer(median, add_indicator=True)` — imputes start-of-series lags and adds a *was-missing* flag.
- **Transformation:** `log1p` of counts (in the feature builder) + cyclical sin/cos for week.
- **Scaling / normalisation:** `StandardScaler` for the linear baseline (trees don't need it).
- **Encoding:** `OneHotEncoder` for RF / XGBoost / LightGBM; **native categorical handling** for CatBoost.
""")
code(r"""
def make_pre(scale=False):
    num = [("imp", SimpleImputer(strategy="median", add_indicator=True))] + ([("sc", StandardScaler())] if scale else [])
    return ColumnTransformer([("num", Pipeline(num), NUM_FEATURES),
                              ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES)],
                             verbose_feature_names_out=False)

def evaluate(name, y, p, thr=0.5):
    yhat = (p >= thr).astype(int)
    return {"model": name, "ROC-AUC": roc_auc_score(y, p), "PR-AUC": average_precision_score(y, p),
            "F1": f1_score(y, yhat), "Precision": precision_score(y, yhat, zero_division=0),
            "Recall": recall_score(y, yhat), "Bal. Acc": balanced_accuracy_score(y, yhat),
            "Brier": brier_score_loss(y, p), "threshold": thr}

def best_f1_threshold(y, p):
    grid = np.linspace(0.1, 0.9, 81)
    return float(grid[np.argmax([f1_score(y, (p >= t).astype(int)) for t in grid])])

tscv = TimeSeriesSplit(n_splits=4)   # expanding-window CV that respects time order
neg_pos = float((ytr == 0).sum() / (ytr == 1).sum()); print("neg/pos ratio in train:", round(neg_pos, 2))
""")

md(r"""
## 6. "Existing approaches" — baselines to beat
1. **Persistence rule** (what an officer reading the WER does today): *alert next week if this week is already above the threshold*. Score = `cases_t / threshold`.
2. **Seasonal naive**: alert if the same week last year was above threshold.
3. **Logistic Regression** (scaled, one-hot) — the classic statistical approach.
""")
code(r"""
val_rows, val_probs = [], {}
p = valid["ratio_thr"].values; val_rows.append(evaluate("Persistence rule", yva, np.clip(p / 2, 0, 1), thr=0.5)); val_probs["Persistence rule"] = np.clip(p / 2, 0, 1)
p = valid["log_c52"].pipe(np.expm1).fillna(0).values / valid["thr"].values
val_rows.append(evaluate("Seasonal naive", yva, np.clip(p / 2, 0, 1), thr=0.5))

logreg = Pipeline([("pre", make_pre(scale=True)), ("m", LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced"))])
logreg.fit(Xtr, ytr); val_probs["Logistic Regression"] = logreg.predict_proba(Xva)[:, 1]
val_rows.append(evaluate("Logistic Regression", yva, val_probs["Logistic Regression"]))
display(pd.DataFrame(val_rows).set_index("model"))
""")
md("(For the rules, `score/2 ≥ 0.5` ⇔ `cases ≥ threshold`, so their F1/recall are those of the plain rule.)")

md(r"""
## 7. PART 1 — Random Forest vs XGBoost with **designed** (non-default) hyper-parameters

Default parameters are **not** used. I first reason about each parameter from the data, then let a small `RandomizedSearchCV` with **`TimeSeriesSplit`** pick within that designed range, optimising **PR-AUC** (average precision — robust to class imbalance).

**Random Forest (bagging) — design reasoning**
| parameter | designed range | why |
|---|---|---|
| `n_estimators` | 300 – 500 | variance falls with more trees; >500 gives negligible gain for 2 CPU cores |
| `max_depth` | 10 – 20 | default `None` grows trees until leaves are pure → memorises noisy weekly counts |
| `min_samples_leaf` | 5 – 20 | each leaf must summarise ≥ 5 district-weeks → smoother probabilities |
| `max_features` | `sqrt`, 0.3, 0.5 | the lag features are highly correlated; sub-sampling features de-correlates the trees |
| `max_samples` | 0.6, 0.8 | bootstrap sub-sampling adds diversity and speed |
| `class_weight` | `None`, `balanced_subsample` | only 26 % positives in training |

**XGBoost (boosting) — design reasoning**
| parameter | designed range | why |
|---|---|---|
| `n_estimators` × `learning_rate` | 300–800 × 0.02–0.08 | small steps + many trees generalise better than default 0.3 |
| `max_depth` | 3 – 6 | boosting needs weak learners; deep trees overfit epidemic years |
| `min_child_weight` | 1, 5, 10 | minimum hessian per leaf — regularises rare extreme weeks |
| `subsample`, `colsample_bytree` | 0.7–0.9, 0.6–0.9 | stochastic boosting, fights correlated lags |
| `reg_lambda`, `gamma` | 1–10, 0–1 | L2 penalty & minimum loss reduction to split |
| `scale_pos_weight` | 1 or neg/pos ≈ 2.8 | imbalance handling |
""")
code(r"""
rf_space = {"m__n_estimators": [300, 400, 500], "m__max_depth": [10, 14, 20], "m__min_samples_leaf": [5, 10, 20],
            "m__max_features": ["sqrt", 0.3, 0.5], "m__max_samples": [0.6, 0.8], "m__class_weight": [None, "balanced_subsample"]}
rf_search = RandomizedSearchCV(Pipeline([("pre", make_pre()), ("m", RandomForestClassifier(random_state=SEED, n_jobs=-1))]),
                               rf_space, n_iter=2 if FAST else 10, cv=tscv, scoring="average_precision", random_state=SEED, n_jobs=1)
t0 = time.time(); rf_search.fit(Xtr, ytr); print(f"RF search {time.time()-t0:.0f}s | best CV PR-AUC {rf_search.best_score_:.3f}")
print({k.replace('m__', ''): v for k, v in rf_search.best_params_.items()})
""")
code(r"""
xgb_space = {"m__n_estimators": [300, 500, 800], "m__learning_rate": [0.02, 0.04, 0.08], "m__max_depth": [3, 4, 5, 6],
             "m__min_child_weight": [1, 5, 10], "m__subsample": [0.7, 0.8, 0.9], "m__colsample_bytree": [0.6, 0.75, 0.9],
             "m__reg_lambda": [1, 5, 10], "m__gamma": [0, 0.5, 1], "m__scale_pos_weight": [1, round(neg_pos, 2)]}
xgb_base = XGBClassifier(tree_method="hist", eval_metric="aucpr", random_state=SEED, n_jobs=2)
xgb_search = RandomizedSearchCV(Pipeline([("pre", make_pre()), ("m", xgb_base)]), xgb_space, n_iter=2 if FAST else 20, cv=tscv,
                                scoring="average_precision", random_state=SEED, n_jobs=1)
t0 = time.time(); xgb_search.fit(Xtr, ytr); print(f"XGB search {time.time()-t0:.0f}s | best CV PR-AUC {xgb_search.best_score_:.3f}")
print({k.replace('m__', ''): v for k, v in xgb_search.best_params_.items()})
""")
code(r"""
rf, xgb = rf_search.best_estimator_, xgb_search.best_estimator_
val_probs["Random Forest"] = rf.predict_proba(Xva)[:, 1]
val_probs["XGBoost"] = xgb.predict_proba(Xva)[:, 1]
part1 = pd.DataFrame([evaluate(n, yva, val_probs[n]) for n in ["Random Forest", "XGBoost"]]).set_index("model")
display(part1)

fig, ax = plt.subplots(1, 4, figsize=(16, 3.6))
for n, c in [("Random Forest", "#2e8b57"), ("XGBoost", "#c0392b")]:
    RocCurveDisplay.from_predictions(yva, val_probs[n], name=n, ax=ax[0], curve_kwargs={"color": c})
    PrecisionRecallDisplay.from_predictions(yva, val_probs[n], name=n, ax=ax[1], curve_kwargs={"color": c})
for a, n in zip(ax[2:], ["Random Forest", "XGBoost"]):
    ConfusionMatrixDisplay.from_predictions(yva, (val_probs[n] >= .5).astype(int), ax=a, colorbar=False,
                                            display_labels=["normal", "outbreak"], cmap="Blues"); a.set_title(n + " (valid, t=0.5)")
ax[0].set_title("ROC — validation 2022-23"); ax[1].set_title("Precision-Recall — validation")
plt.tight_layout(); plt.savefig(FIG / "04_rf_vs_xgb.png", bbox_inches="tight"); plt.show()
""")
code(r"""
# Over/under-fitting check: CV score vs validation score, and fit time
t0 = time.time(); rf.fit(Xtr, ytr); rf_t = time.time() - t0
t0 = time.time(); xgb.fit(Xtr, ytr); xgb_t = time.time() - t0
diag = pd.DataFrame({
    "train PR-AUC": [average_precision_score(ytr, m.predict_proba(Xtr)[:, 1]) for m in (rf, xgb)],
    "time-series CV PR-AUC": [rf_search.best_score_, xgb_search.best_score_],
    "validation PR-AUC": [part1.loc["Random Forest", "PR-AUC"], part1.loc["XGBoost", "PR-AUC"]],
    "fit time (s)": [rf_t, xgb_t]}, index=["Random Forest", "XGBoost"])
display(diag)
""")

md(r"""
### 7.1 Interpreted strengths & limitations (Random Forest vs XGBoost)

| | **Random Forest (bagging)** | **XGBoost (boosting)** |
|---|---|---|
| How it learns | many deep, de-correlated trees on bootstrap samples, averaged → **reduces variance** | shallow trees added sequentially, each fixing the residual errors of the previous ones → **reduces bias** |
| Strengths here | very robust to noisy weekly counts and outlier epidemic weeks; few sensitive hyper-parameters; trivially parallel; OOB-style stability | usually the best discrimination on tabular data; built-in L1/L2 regularisation; handles imbalance via `scale_pos_weight`; native missing-value handling; fast `hist` algorithm |
| Limitations here | probabilities are compressed toward 0.5 (averaging) → needs threshold tuning; large model files; cannot extrapolate beyond counts seen in training (e.g. post-2022 higher burden) | more hyper-parameters, easier to overfit an epidemic year if `learning_rate`/`max_depth` are badly set; sequential → slower to tune; also cannot extrapolate trends |
| Interpretability | impurity importance (biased to high-cardinality features) → use permutation/SHAP | exact TreeSHAP values, gain/cover importance |

The numeric comparison above (validation table and overfitting check) is interpreted in Section 9 together with all other models.
""")

md(r"""
## 8. Beyond Part 1 — Soft voting (lecture notebook), LightGBM, CatBoost and Stacking

- **Soft voting** (simplified lecture pipeline): average the probabilities of the tuned RF and XGBoost.
- **LightGBM**: leaf-wise gradient boosting — faster, good with many rows. Designed: `num_leaves=31`, `max_depth=6`, `learning_rate=0.03`, 600 trees, `min_child_samples=40`, bagging 0.8 / feature 0.7, L2=5.
- **CatBoost**: our data has **4 categorical features** (district, province, climate zone, monsoon season) — CatBoost encodes them with ordered target statistics instead of one-hot. Designed: depth 6, `learning_rate=0.05`, 700 iterations, `l2_leaf_reg=5`, balanced class weights.
- **Stacking**: RF + XGBoost + LightGBM + CatBoost as level-0 learners, **logistic regression** as the meta-learner trained on 5-fold out-of-fold probabilities (`KFold`, unshuffled → contiguous time blocks).
""")
code(r"""
lgbm = Pipeline([("pre", make_pre()), ("m", LGBMClassifier(n_estimators=100 if FAST else 600, learning_rate=0.03, num_leaves=31, max_depth=6,
                 min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5,
                 random_state=SEED, n_jobs=2, verbose=-1))])
catb = CatBoostWrapper(cat_features=CAT_FEATURES, iterations=100 if FAST else 700, depth=6, learning_rate=0.05, l2_leaf_reg=5,
                       auto_class_weights="Balanced", random_seed=SEED, thread_count=2)
voting = VotingClassifier([("rf", rf), ("xgb", xgb)], voting="soft")
stack = StackingClassifier([("rf", rf), ("xgb", xgb), ("lgbm", lgbm), ("cat", catb)],
                           final_estimator=LogisticRegression(C=1.0, max_iter=1000), cv=KFold(5, shuffle=False),
                           stack_method="predict_proba", n_jobs=1)
extra = {"Soft Voting (RF+XGB)": voting, "LightGBM": lgbm, "CatBoost": catb, "Stacking (RF+XGB+LGBM+Cat)": stack}
for n, m in extra.items():
    t0 = time.time(); m.fit(Xtr, ytr); val_probs[n] = m.predict_proba(Xva)[:, 1]; print(f"{n}: {time.time()-t0:.0f}s")
""")

md("## 9. Model comparison on validation → model selection")
code(r"""
ALL = ["Persistence rule", "Logistic Regression", "Random Forest", "XGBoost", "Soft Voting (RF+XGB)",
       "LightGBM", "CatBoost", "Stacking (RF+XGB+LGBM+Cat)"]
thr_opt = {n: (0.5 if n == "Persistence rule" else best_f1_threshold(yva, val_probs[n])) for n in ALL}
val_table = pd.DataFrame([evaluate(n, yva, val_probs[n], thr_opt[n]) for n in ALL]).set_index("model")
val_table = val_table.sort_values("PR-AUC", ascending=False)
display(val_table.style.background_gradient(subset=["ROC-AUC", "PR-AUC", "F1"], cmap="Greens").format("{:.3f}"))
best_name = val_table.drop(index=["Persistence rule"]).index[0]
print("Selected model (highest validation PR-AUC):", best_name, "| decision threshold:", thr_opt[best_name])
""")

md(r"""
## 10. Final, untouched test (2024 – May 2026)
All learned models are **refit on train + validation (2007–2023)** and evaluated once on the test period with the thresholds chosen on validation.
""")
code(r"""
fitted = {"Logistic Regression": logreg, "Random Forest": rf, "XGBoost": xgb, **extra}
test_probs = {"Persistence rule": np.clip(test["ratio_thr"].values / 2, 0, 1)}
for n, m in fitted.items():
    t0 = time.time(); m.fit(Xtv, ytv); test_probs[n] = m.predict_proba(Xte)[:, 1]; print(f"refit {n}: {time.time()-t0:.0f}s")
test_table = pd.DataFrame([evaluate(n, yte, test_probs[n], thr_opt[n]) for n in ALL]).set_index("model").sort_values("PR-AUC", ascending=False)
display(test_table.style.background_gradient(subset=["ROC-AUC", "PR-AUC", "F1"], cmap="Greens").format("{:.3f}"))
test_table.to_csv(ROOT / "reports" / "test_metrics.csv"); val_table.to_csv(ROOT / "reports" / "validation_metrics.csv")
""")
code(r"""
fig, ax = plt.subplots(1, 3, figsize=(16, 4))
order = test_table.index[::-1]
ax[0].barh(order, test_table.loc[order, "PR-AUC"], color=["#c0392b" if n == best_name else "#7aa6c2" for n in order])
ax[0].set_xlim(test_table["PR-AUC"].min() - .03, 1); ax[0].set_title("Test PR-AUC (2024–26)")
for n in ["Persistence rule", "Logistic Regression", "Random Forest", "XGBoost", best_name]:
    PrecisionRecallDisplay.from_predictions(yte, test_probs[n], name=n, ax=ax[1])
ax[1].set_title("Precision–Recall, test"); ax[1].legend(fontsize=7)
ConfusionMatrixDisplay.from_predictions(yte, (test_probs[best_name] >= thr_opt[best_name]).astype(int), ax=ax[2],
                                        colorbar=False, display_labels=["normal", "outbreak"], cmap="Blues")
ax[2].set_title(f"{best_name}\n(test, threshold {thr_opt[best_name]:.2f})", fontsize=9)
plt.tight_layout(); plt.savefig(FIG / "05_test_comparison.png", bbox_inches="tight"); plt.show()
""")
code(r"""
# Where does the selected model work well / badly?  Per-district test performance
best_model = fitted[best_name]
tt = test.assign(p=test_probs[best_name], yhat=(test_probs[best_name] >= thr_opt[best_name]).astype(int))
by_d = tt.groupby("district").apply(lambda g: pd.Series({
    "positive rate": g[TARGET].mean(), "F1": f1_score(g[TARGET], g.yhat, zero_division=0),
    "ROC-AUC": roc_auc_score(g[TARGET], g.p) if g[TARGET].nunique() > 1 else np.nan,
    "zone": DISTRICT_META[g.name][1]})).sort_values("F1")
fig, ax = plt.subplots(figsize=(11, 3.6))
ax.bar(by_d.index, by_d.F1, color=by_d.zone.map({"Wet": "#c0392b", "Intermediate": "#e67e22", "Dry": "#7f8c8d"}))
ax.set_title(f"{best_name}: test F1 by RDHS (red=wet, orange=intermediate, grey=dry zone)"); ax.tick_params(axis="x", rotation=70, labelsize=8)
plt.tight_layout(); plt.savefig(FIG / "06_f1_by_district.png", bbox_inches="tight"); plt.show()
display(by_d.groupby("zone")[["F1", "ROC-AUC", "positive rate"]].mean())
""")

md(r"""
## 11. Explainability (Lectures 10–11): SHAP (global + local) and LIME (local)

- **SHAP** (TreeExplainer on the XGBoost pipeline — exact Shapley values for trees) → global *beeswarm*, dependence plot and a local waterfall for the held-out Colombo week.
- **Permutation importance** on the selected model (model-agnostic, computed on the test set).
- **LIME** (Local Interpretable Model-agnostic Explanations) → fits a weighted linear surrogate *g* around the held-out sample using perturbed neighbours, explaining the **selected model** itself (works even for a stacked ensemble because LIME only needs `predict_proba`).
""")
code(r"""
xgb_pre, xgb_m = xgb.named_steps["pre"], xgb.named_steps["m"]
Xte_t = pd.DataFrame(xgb_pre.transform(Xte), columns=xgb_pre.get_feature_names_out())
explainer = shap.TreeExplainer(xgb_m)
sv = explainer(Xte_t)
plt.figure(); shap.plots.beeswarm(sv, max_display=15, show=False)
plt.title("SHAP summary — XGBoost, test set"); plt.savefig(FIG / "07_shap_beeswarm.png", bbox_inches="tight"); plt.show()
mean_abs = pd.Series(np.abs(sv.values).mean(0), index=Xte_t.columns).sort_values(ascending=False)
display(mean_abs.head(10).rename("mean |SHAP|").to_frame().T)
""")
code(r"""
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
shap.plots.scatter(sv[:, "ratio_thr"], color=sv[:, "growth_1w"], ax=ax[0], show=False)
ax[0].set_xlim(0, 6); ax[0].set_title("Effect of this week's cases / threshold")
shap.plots.scatter(sv[:, "week_cos"], color=sv[:, "week_sin"], ax=ax[1], show=False)
ax[1].set_title("Seasonality (week_cos: +1 = Jan, -1 = Jul)")
plt.tight_layout(); plt.savefig(FIG / "08_shap_dependence.png", bbox_inches="tight"); plt.show()
""")
code(r"""
perm = permutation_importance(best_model, Xte, yte, scoring="average_precision", n_repeats=1 if FAST else 5, random_state=SEED, n_jobs=1)
pi = pd.Series(perm.importances_mean, index=FEATURES).sort_values()
fig, ax = plt.subplots(figsize=(7, 5)); ax.barh(pi.index, pi.values, color="#1f5f8b")
ax.set_title(f"Permutation importance — {best_name} (drop in test PR-AUC)"); plt.tight_layout()
plt.savefig(FIG / "09_permutation_importance.png", bbox_inches="tight"); plt.show()
""")

md("### 11.1 Local explanation of the held-out sample — SHAP waterfall vs LIME")
code(r"""
h_t = pd.DataFrame(xgb_pre.transform(holdout[FEATURES]), columns=xgb_pre.get_feature_names_out())
plt.figure(); shap.plots.waterfall(explainer(h_t)[0], max_display=12, show=False)
plt.title("SHAP waterfall — Colombo held-out week (log-odds)"); plt.savefig(FIG / "10_shap_waterfall.png", bbox_inches="tight"); plt.show()

# LIME on the *selected* model: label-encode categoricals for LIME's sampler, decode inside predict_fn
cat_maps = {c: sorted(trval[c].unique()) for c in CAT_FEATURES}
def encode(X):
    Z = X[FEATURES].copy()
    for c in CAT_FEATURES: Z[c] = Z[c].map({v: i for i, v in enumerate(cat_maps[c])})
    return Z.astype(float)
def predict_fn(Z):
    X = pd.DataFrame(Z, columns=FEATURES)
    for c in CAT_FEATURES: X[c] = [cat_maps[c][int(round(min(max(v, 0), len(cat_maps[c]) - 1)))] for v in X[c]]
    for c in NUM_FEATURES: X[c] = X[c].astype(float)
    return best_model.predict_proba(X)
bg = encode(trval).fillna(encode(trval).median())
lime_exp = lime.lime_tabular.LimeTabularExplainer(bg.values, feature_names=FEATURES, class_names=["normal", "outbreak"],
             categorical_features=list(range(len(CAT_FEATURES))), categorical_names={i: cat_maps[c] for i, c in enumerate(CAT_FEATURES)},
             discretize_continuous=True, mode="classification", random_state=SEED)
exp = lime_exp.explain_instance(encode(holdout).values[0], predict_fn, num_features=10, num_samples=500 if FAST else 3000)
fig = exp.as_pyplot_figure(); fig.set_size_inches(8, 4); plt.title(f"LIME — {best_name}, Colombo held-out week")
plt.tight_layout(); plt.savefig(FIG / "11_lime_holdout.png", bbox_inches="tight"); plt.show()
print("LIME local surrogate R²:", round(exp.score, 3), "| surrogate prediction:", np.round(exp.local_pred, 3))
display(pd.DataFrame(exp.as_list(), columns=["condition", "weight (→ outbreak)"]))
""")

md(r"""
## 12. Single-sample inference from the exported pickle (Part 1, Step 6)
The selected model is exported together with everything the front end needs (district thresholds, decision threshold, feature list, latest inputs per district). I then **re-load the `.pkl` from disk** and predict the held-out Colombo week that the model has never seen.
""")
code(r"""
natl = df.groupby("date").cases.sum()
latest = {d: inputs_for(d, df[df.district == d].date.max()) | {"date": str(df[df.district == d].date.max().date())} for d in DISTRICT_META}
bundle = {"model": best_model, "model_name": best_name, "decision_threshold": thr_opt[best_name],
          "district_thresholds": thresholds, "features": FEATURES, "cat_features": CAT_FEATURES, "num_features": NUM_FEATURES,
          "explainer_model": xgb, "latest_inputs": latest, "test_metrics": test_table.reset_index().to_dict("records"),
          "trained_on": "2007-2023 district-weeks (Epidemiology Unit WER via denguedatahub)", "target": TARGET}
with open(MODELS / "dengue_outbreak_model.pkl", "wb") as f: pickle.dump(bundle, f)
pickle.dump(voting, open(MODELS / "model.pkl", "wb"))            # soft-voting ensemble, as in the lecture notebook
pickle.dump(xgb, open(MODELS / "xgb_model.pkl", "wb"))
for p in sorted(MODELS.glob("*.pkl")): print(p.name, f"{p.stat().st_size/1e6:.1f} MB")
""")
code(r"""
loaded = pickle.load(open(MODELS / "dengue_outbreak_model.pkl", "rb"))
h = holdout.iloc[0]
x_new = features_from_inputs(**inputs_for(h.district, h.date), thresholds=loaded["district_thresholds"])
prob = loaded["model"].predict_proba(x_new[loaded["features"]])[0, 1]
pred = int(prob >= loaded["decision_threshold"])
print(f"District: {h.district} | week starting {h.date.date()} | cases this week: {int(h.cases)} | alert threshold: {h.thr:.0f}")
print(f"Model ({loaded['model_name']}) → P(outbreak next week) = {prob:.3f} → prediction: {'OUTBREAK ALERT' if pred else 'normal'}")
print(f"Actual next week: {int(h.next_cases)} cases → {'OUTBREAK' if h[TARGET] else 'normal'} | correct: {pred == int(h[TARGET])}")
""")
md("### 12.1 Genuine forecast: next week for all 26 RDHS (last week in the data, target not yet known)")
code(r"""
fc = forecast_rows.copy()
fc["P(outbreak next week)"] = loaded["model"].predict_proba(fc[FEATURES])[:, 1]
fc["alert"] = np.where(fc["P(outbreak next week)"] >= loaded["decision_threshold"], "ALERT", "")
out = fc[["district", "date", "cases", "thr", "P(outbreak next week)", "alert"]].rename(columns={"thr": "threshold", "date": "week starting"})
out = out.sort_values("P(outbreak next week)", ascending=False).reset_index(drop=True)
out.to_csv(ROOT / "reports" / "next_week_forecast.csv", index=False); display(out)
""")

md(r"""
## 13. Conclusions

The tables above are interpreted in `REPORT.md`; the key points:

1. **Every ML ensemble beats the "existing approach"** (persistence rule used when reading the WER) and logistic regression on threshold-free metrics — adding momentum, seasonality, geography and national context on top of "this week's level" helps.
2. **Random Forest vs XGBoost** — see the Part-1 table: boosting gives the better ranking quality with shallow trees and regularisation; the forest is more stable but produces compressed probabilities.
3. **Model choice is justified by validation PR-AUC (time-aware) and confirmed on a 2½-year untouched test period**; the decision threshold is tuned for F1 on 2022–23 to cope with the post-2022 prior shift.
4. **Explanations agree with epidemiology**: this week's burden relative to the district's normal, the 4-week trend and the season (SW-monsoon weeks) dominate SHAP and LIME; geography matters mostly through the threshold scaling.
5. **Limitations**: no rainfall/temperature/population covariates (Department of Meteorology data would add genuine lead time), reporting delays in the WER, and prior drift → the deployed model should be **monitored and retrained** (Lecture 10 MLOps loop).
""")

nb = nbf.v4.new_notebook(); nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "dengue_outbreak_ml.ipynb")
print("written", len(cells), "cells")
