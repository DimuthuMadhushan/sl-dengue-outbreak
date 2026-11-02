"""
Sri Lanka Dengue Outbreak Early-Warning — Streamlit front end.

Loads the pickled model bundle produced by notebooks/dengue_outbreak_ml.ipynb and
lets a user type in the latest Weekly Epidemiological Report (WER) numbers for an
RDHS division to get P(outbreak next week) plus a SHAP explanation.

Run:  streamlit run app/app.py
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from features import DISTRICT_META, features_from_inputs, monsoon_season  # noqa: E402
import cb_wrapper  # noqa: E402,F401  (needed to unpickle the CatBoost wrapper)

MODEL_PATH = ROOT / "models" / "dengue_outbreak_model.pkl"

st.set_page_config(page_title="Dengue Early Warning — Sri Lanka", page_icon="🦟", layout="wide")


@st.cache_resource
def load_bundle():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_explainer(_xgb_pipeline):
    import shap
    return shap.TreeExplainer(_xgb_pipeline.named_steps["m"])


bundle = load_bundle()
model, thr_decision = bundle["model"], bundle["decision_threshold"]
thresholds, FEATURES = bundle["district_thresholds"], bundle["features"]
latest = bundle["latest_inputs"]

PRETTY = {
    "ratio_thr": "This week ÷ district threshold", "roll4_ratio_thr": "4-week avg ÷ threshold",
    "log_c0": "Cases this week (log)", "log_c1": "Cases 1 wk ago (log)", "log_c2": "Cases 2 wks ago (log)",
    "log_c3": "Cases 3 wks ago (log)", "log_c52": "Same week last year (log)", "log_roll4": "4-week average (log)",
    "log_roll8": "8-week average (log)", "std_log4": "4-week volatility", "growth_1w": "1-week growth",
    "growth_4w": "4-week growth", "log_thr": "District baseline level", "log_nat0": "National cases (log)",
    "nat_growth_1w": "National 1-week growth", "week_sin": "Season (sin)", "week_cos": "Season (cos)",
}


def pretty(name: str) -> str:
    if name.startswith("missingindicator_"):
        return "Missing: " + PRETTY.get(name.replace("missingindicator_", ""), name)
    for cat in ["district", "province", "climate_zone", "monsoon_season"]:
        if name.startswith(cat + "_"):
            return f"{cat.replace('_', ' ').title()} = {name[len(cat) + 1:]}"
    return PRETTY.get(name, name)


def predict(inputs: dict):
    x = features_from_inputs(**inputs, thresholds=thresholds)
    return float(model.predict_proba(x[FEATURES])[0, 1]), x


# ----------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("🦟 Model")
    st.write(f"**Deployed model:** {bundle['model_name']}")
    st.write(f"**Decision threshold:** {thr_decision:.2f}")
    st.caption(bundle["trained_on"])
    m = pd.DataFrame(bundle["test_metrics"]).set_index("model")[["PR-AUC", "Recall"]]
    st.write("**Test performance (2024 – May 2026)**")
    st.dataframe(m.style.format("{:.3f}"), width="stretch")
    st.caption("Target: next week's cases ≥ the district's 75th-percentile weekly count (2007–2021).")

st.title("Dengue Outbreak Early-Warning — Sri Lanka RDHS divisions")
st.write("Enter the latest **Weekly Epidemiological Report** figures for a division to estimate the "
         "probability that **next week** will be a high-transmission (outbreak-alert) week.")

tab_pred, tab_outlook, tab_about = st.tabs(["🔮 Predict", "🗺️ Next-week outlook (all 26 RDHS)", "ℹ️ About"])

# ----------------------------------------------------------------- predict tab
with tab_pred:
    c1, c2 = st.columns([1, 2])
    with c1:
        district = st.selectbox("RDHS division", sorted(DISTRICT_META), index=sorted(DISTRICT_META).index("Colombo"))
        prefill = st.toggle("Pre-fill with the latest WER data", value=True)
        d = latest[district] if prefill else {"week": 1, "last8": [0] * 8, "same_week_last_year": 0,
                                              "national_this_week": 1000, "national_last_week": 1000}
        if prefill:
            st.caption(f"Latest report in dataset: week starting {d['date']}")
        week = st.number_input("Epidemiological week of the *current* week (1–53)", 1, 53, int(d["week"]), key=f"wk{district}{prefill}")
        prov, zone = DISTRICT_META[district]
        st.info(f"**{district}** — {prov} Province, {zone} zone\n\nAlert threshold: **{thresholds[district]:.0f} cases/week**\n\n"
                f"Season: {monsoon_season(week).replace('_', ' ')}")
    with c2:
        st.subheader("Weekly dengue cases in this division")
        labels = ["This week", "1 week ago", "2 weeks ago", "3 weeks ago", "4 weeks ago", "5 weeks ago", "6 weeks ago", "7 weeks ago"]
        cols = st.columns(4)
        last8 = []
        for k, lab in enumerate(labels):
            v = d["last8"][k]
            last8.append(cols[k % 4].number_input(lab, min_value=0, value=int(v if v is not None and v == v else 0),
                                                  step=1, key=f"c{k}{district}{prefill}"))
        cc = st.columns(3)
        c52 = cc[0].number_input("Same week last year", min_value=0, value=int(d["same_week_last_year"] or 0), key=f"c52{district}{prefill}")
        n0 = cc[1].number_input("National total this week", min_value=0, value=int(d["national_this_week"] or 0), key=f"n0{district}{prefill}")
        n1 = cc[2].number_input("National total last week", min_value=0, value=int(d["national_last_week"] or 0), key=f"n1{district}{prefill}")

    if st.button("Predict next week", type="primary", width="stretch"):
        inputs = dict(district=district, week=int(week), last8=last8, same_week_last_year=c52,
                      national_this_week=n0, national_last_week=n1)
        prob, x = predict(inputs)
        alert = prob >= thr_decision
        r1, r2, r3 = st.columns(3)
        r1.metric("P(outbreak next week)", f"{prob:.1%}")
        r2.metric("Decision", "🚨 ALERT" if alert else "✅ Normal")
        r3.metric("This week vs threshold", f"{last8[0] / thresholds[district]:.2f}×")
        st.progress(min(prob, 1.0))
        if alert:
            st.error("Recommend escalating vector control (fogging, PHI inspections, source reduction) and "
                     "preparing hospital capacity in this division.")
        else:
            st.success("Transmission expected to stay below the alert threshold — continue routine surveillance.")

        # SHAP explanation from the XGBoost member
        try:
            xp = bundle["explainer_model"]
            pre = xp.named_steps["pre"]
            xt = pd.DataFrame(pre.transform(x[FEATURES]), columns=pre.get_feature_names_out())
            sv = load_explainer(xp).shap_values(xt)[0]
            contrib = pd.Series(sv, index=[pretty(c) for c in xt.columns])
            top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(8)
            st.subheader("Why? — top SHAP contributions (XGBoost, log-odds)")
            st.caption("Positive bars push toward an outbreak alert, negative bars toward normal.")
            st.bar_chart(top.rename("SHAP value").to_frame(), horizontal=True)
        except Exception as e:  # explanation is optional
            st.caption(f"Explanation unavailable: {e}")
        with st.expander("Engineered features sent to the model"):
            st.dataframe(x[FEATURES].T.rename(columns={0: "value"}).astype(str))

# ----------------------------------------------------------------- outlook tab
with tab_outlook:
    rows = []
    for dname, d in latest.items():
        p, _ = predict({k: d[k] for k in ["district", "week", "last8", "same_week_last_year",
                                          "national_this_week", "national_last_week"]})
        rows.append({"RDHS": dname, "zone": DISTRICT_META[dname][1], "cases this week": d["last8"][0],
                     "threshold": round(thresholds[dname]), "P(outbreak next week)": p,
                     "alert": "🚨" if p >= thr_decision else ""})
    out = pd.DataFrame(rows).sort_values("P(outbreak next week)", ascending=False).reset_index(drop=True)
    st.write(f"Forecast for the week after **{next(iter(latest.values()))['date']}** using the latest data bundled with the model.")
    st.bar_chart(out.set_index("RDHS")["P(outbreak next week)"])
    st.dataframe(out.style.format({"P(outbreak next week)": "{:.1%}", "cases this week": "{:.0f}"}), width="stretch")

# ----------------------------------------------------------------- about tab
with tab_about:
    st.markdown("""
**Data:** Epidemiology Unit, Ministry of Health Sri Lanka — Weekly Epidemiological Reports, district-level dengue
notifications Dec 2006 – May 2026 (tidied by the open-source `denguedatahub` R package).

**Models compared:** persistence rule, logistic regression, Random Forest, XGBoost, soft voting (RF+XGB), LightGBM,
CatBoost and a stacked ensemble. Hyper-parameters were designed and tuned with time-series cross-validation.

**Limitations:** the model only sees reported cases (no rainfall/temperature yet); reporting delays and the higher
dengue burden after 2022 mean the model should be monitored and retrained regularly. This tool supports — it does not
replace — the judgement of Medical Officers of Health and regional epidemiologists.
""")
