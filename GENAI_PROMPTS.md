# GenAI prompts used for the front end (Part 1 – Step 7)

The guideline asks for the pickle loader + UI to be generated with a GenAI assistant (Gemini / ChatGPT / Claude).
These are the prompts used; the result is `app/app.py`, which I then reviewed and adjusted (feature builder reuse,
SHAP explanation, outlook tab).

---

### Prompt 1 — pickle loader + input form

> I have trained a scikit-learn machine-learning model that predicts whether a Sri Lankan RDHS health division
> will have a **dengue outbreak-alert week next week**. It is saved with `pickle` as
> `models/dengue_outbreak_model.pkl`. The pickle is a Python **dict** with keys:
> `model` (a fitted sklearn pipeline / ensemble with `predict_proba`), `decision_threshold` (float),
> `district_thresholds` (dict district → cases/week), `features` (list of column names),
> `explainer_model` (an XGBoost sklearn Pipeline with steps `pre` and `m`), `latest_inputs`
> (dict district → latest raw inputs) and `test_metrics` (list of dicts).
>
> Feature engineering is done by my function
> `features_from_inputs(district, week, last8, same_week_last_year, national_this_week, national_last_week, thresholds)`
> in `src/features.py`, which returns a one-row DataFrame.
>
> Write a **Streamlit** app (`app/app.py`) that:
> 1. loads the pickle once with `st.cache_resource` (add `src/` to `sys.path` first, and import `cb_wrapper` so the pickle can be loaded);
> 2. has input fields for: RDHS division (select box with the 26 divisions), epidemiological week (1–53),
>    weekly dengue cases for this week and the previous 7 weeks, cases in the same week last year,
>    and the national total for this week and last week;
> 3. has a toggle to pre-fill the fields with `latest_inputs[district]`;
> 4. on a **Predict** button, builds the features with `features_from_inputs`, calls `model.predict_proba`,
>    and shows the probability, an "OUTBREAK ALERT / Normal" decision using `decision_threshold`, and a short
>    public-health recommendation;
> 5. shows the model name and test metrics in the sidebar.

### Prompt 2 — explanation + outlook

> Extend the app: after predicting, compute SHAP values with `shap.TreeExplainer` on
> `explainer_model.named_steps["m"]` using the row transformed by `explainer_model.named_steps["pre"]`
> (column names from `get_feature_names_out()`), and show the 8 largest contributions as a horizontal bar chart
> with human-readable names. Add a second tab that runs the model for all 26 divisions using `latest_inputs`
> and shows a ranked table + bar chart of next-week outbreak probability, and an "About" tab describing the
> data source (Epidemiology Unit WER via `denguedatahub`) and limitations.

### Prompt 3 — deployment (bonus)

> Write a minimal Dockerfile (python:3.13-slim) that installs `requirements-app.txt`, copies `src/`, `app/` and the
> pickle, and starts Streamlit on port 8501 so it can be deployed to Azure Container Apps or Streamlit Community Cloud.
