---
title: Credit Risk Modelling API
emoji: 📊
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Credit Risk Modelling

Predicts the probability that a loan applicant defaults, and converts it into a CIBIL-style **300–900
credit score** with a rating band a loan officer can act on.

**[Live demo](https://<your-subdomain>.streamlit.app)** ·
**[API docs](https://<your-hf-username>-credit-risk-api.hf.space/docs)** ·
**[Notebook](https://nbviewer.org/github/johnwilfredd-curimo/credit-risk-modelling/blob/main/notebooks/credit_risk_modelling.ipynb)**

---

## The problem

Lauki Finance needs a default probability per applicant, and the statement of work makes
**explainability a contract deliverable**: the lender has to be able to say *why* an applicant was
scored the way they were.

The modelling difficulty is **class imbalance**: 4,297 defaults in 50,000 loans, about 8.6%. A model that
predicts "no default" every time scores 91% accuracy and is worthless.

## Result

| Metric | Value | Reading |
|---|---|---|
| ROC-AUC | **0.9837** | separates defaulters from non-defaulters almost cleanly |
| Gini | **0.967** | `2 × AUC − 1` |
| KS statistic | **85.98%** at decile 8 | well above the 40 rule-of-thumb, and in the top 3 deciles |
| Rank ordering | **holds** | no decile breaks monotonicity |

**Default-class performance** on a test set of 12,497 applicants, 1,074 actual defaults:

| Class | Precision | Recall | F1 |
|---|---|---|---|
| 0 (repaid) | 0.99 | 0.93 | 0.96 |
| **1 (default)** | **0.56** | **0.94** | **0.70** |

Recall 0.94 against precision 0.56 is a deliberate trade, not a weakness: the model catches 94% of real
defaulters and accepts a high false-positive rate to do it, because a missed default costs a lender far
more than a declined good applicant. That balance came from SMOTE-Tomek resampling plus
`class_weight='balanced'`, tuned with Optuna on macro-F1 (best CV score 0.9459).

Accuracy is deliberately not the headline. With 8.6% positives it's the least informative number
available, which is why this project is evaluated on the credit-risk metrics instead: WOE/IV for feature
selection, ROC-AUC and Gini for separation, and rank ordering plus KS for whether the score is usable
decile by decile.

### Why Logistic Regression, not XGBoost

XGBoost was trained and evaluated alongside it. Logistic Regression won on two grounds:

1. **The SOW requires explainability.** A linear model in log-odds gives a per-feature contribution you
   can show a regulator or a declined applicant.
2. **It's what makes the credit score possible.** The 300–900 scale is computed directly from the
   model's linear term: `base 300 + (1 − P(default)) × 600`. That formula exists because the model has
   coefficients and an intercept. A tree ensemble would have needed a separate scaling layer bolted on.

Class imbalance was handled three ways across four training attempts (under-sampling, over-sampling,
SMOTE-Tomek), with hyperparameters tuned via Optuna and RandomizedSearchCV scored on **f1** rather than
accuracy.

The notebook also carries two appendices. **A**: Optuna vs. GridSearchCV. **B**: rank ordering and the
KS statistic worked by hand on 16 borrowers.

## Architecture

One model, two interfaces:

```
src/prediction.py   ← loads the artifact bundle, builds the feature vector, scores
   ├── streamlit_app.py   human-facing demo
   └── api/main.py        machine-facing JSON service
```

Both import the same `predict()`. There is no second copy of the preprocessing logic, so the demo and the
API cannot drift apart. The Postman suite asserts they agree.

`artifacts/model_data.joblib` is a single bundle holding the fitted model, the scaler, the feature order
and the columns to scale, so the serving code can't disagree with training about any of them.

```
credit-risk-modelling/
├─ streamlit_app.py          Streamlit entrypoint
├─ src/prediction.py         shared prediction module
├─ api/main.py               FastAPI service
├─ postman/                  collection + environment
├─ artifacts/                model + scaler + feature schema (one .joblib)
├─ dataset/                  customers, loans, bureau data, + appendix B files
├─ notebooks/                the full end-to-end notebook
└─ Dockerfile                serves the API on port 7860
```

## Run it locally

```bash
python -m venv .venv
source .venv/bin/activate                 # macOS / Linux
# Windows Git Bash:    source .venv/Scripts/activate
# Windows PowerShell:  .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The API, from the repo root:

```bash
pip install -r requirements-api.txt
uvicorn api.main:app --reload --port 8000
```

Then open <http://127.0.0.1:8000/docs>.

```bash
curl -X POST http://127.0.0.1:8000/predict_credit_risk \
  -H "Content-Type: application/json" \
  -d '{"age":28,"income":1200000,"loan_amount":2560000,"loan_tenure_months":36,
       "avg_dpd_per_delinquency":20,"delinquency_ratio":30,"credit_utilization_ratio":30,
       "num_open_accounts":2,"residence_type":"Owned","loan_purpose":"Education",
       "loan_type":"Unsecured"}'
```

```json
{ "probability": 0.0412, "credit_score": 875, "rating": "Excellent" }
```

Categorical fields are typed as enums, so an unknown category returns **422** rather than a confident
wrong answer.

To re-run the notebook: `pip install -r requirements-dev.txt`, then run it from the `notebooks/` folder.

## API tests

`postman/` holds a Postman collection covering the health check, the happy path, and two validation
failures. Beyond checking the response shape, it asserts an **invariant of the scoring logic**: the
returned `rating` must always match the band its `credit_score` falls into, whatever the model predicts.

`base_url` is a collection variable defaulting to `http://127.0.0.1:8000`, so it runs with no
setup:

```bash
npx newman run postman/credit-risk-modelling.postman_collection.json
```

Against a deployed instance, override that one variable:

```bash
npx newman run postman/credit-risk-modelling.postman_collection.json \
    --env-var base_url=https://<host>
```

## Attribution

Project brief, dataset and baseline approach from the Codebasics course *Master Machine Learning for Data
Science & AI* (codebasics.io). This repository is my own end-to-end rebuild: the lifecycle structure,
analysis write-up, serving layer, API, test suite and deployment are mine.

Self-directed learning project; not commissioned client work. "Lauki Finance" and "AtliQ AI" are the
course's fictional companies.
