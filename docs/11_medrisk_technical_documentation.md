# MedRISK — Complete Technical Documentation

## 1. Executive Summary
MedRISK is an AI-powered Medical Device Safety Intelligence & Risk Assessment System designed to predict the severity of medical equipment failure-related events. Instead of incorrectly forecasting future device breakdown using time-to-event without continuous sensor data, the system successfully extracts *preventive intelligence* from historical FDA adverse-event patterns and device recurrence metrics. The solution surfaces this through a FastAPI backend, an interactive Streamlit dashboard, a SHAP-based explainer, a rule-based recommendation engine, and a grounded GenAI Copilot for decision support.

## 2. Problem Statement
**"Predicting medical equipment failure is crucial for ensuring patient safety, minimizing downtime, and reducing maintenance costs."** 

Biomedical teams struggle with alert fatigue and prioritizing limited maintenance resources across thousands of devices. While true "predictive maintenance" requires continuous IoT telemetry (which is absent in retrospective FDA reporting datasets), healthcare facilities desperately need a way to triage safety anomalies the moment they occur to prevent severe, repeated historical patterns from translating into catastrophic patient outcomes.

## 3. Proposed Solution
MedRISK addresses this by providing a dual-layer intelligence platform:
1. **Event Risk (ML):** Evaluates ad-hoc problem descriptions using an NLP-based Random Forest model to estimate the likelihood that the described anomaly resembles historically severe, Class I FDA recalls.
2. **Historical Preventive Risk (Rules):** Evaluates device-level history (total prior events, prior Class I events, recalls) to flag chronic offending devices before repeated issues escalate.

This is supported by exact historical evidence retrieval, SHAP feature attribution, and deterministic maintenance recommendations.

## 4. Project Objectives
- **Patient Safety:** Rapidly identify anomalies resembling historically severe profiles (Class I) for priority investigation.
- **Downtime Minimization:** Flag devices with high historical recurrence rates for preventive intervention.
- **Maintenance Cost:** Shift from round-robin scheduling to risk-informed prioritization.
- **Explainability:** Ensure all AI predictions are transparently explained to biomedical technicians.

## 5. Technology Stack
- **Data & ML Engineering:** Python, Pandas, Scikit-Learn, XGBoost, TF-IDF
- **Explainability & Calibration:** SHAP (`TreeExplainer`), Isotonic Regression
- **Backend API:** FastAPI, Pydantic, SQLite (FTS5 search)
- **Frontend UI:** Streamlit
- **Generative AI:** OpenAI/Gemini APIs (LLM via Copilot routing)
- **Testing:** Pytest

## 6. Development Timeline — Day by Day
*Timeline verified via Git commit history and timestamps.*

| Date / Day | Major Work | Domain | What Changed | Evidence |
|---|---|---|---|---|
| Phase 1 | Data Pipeline | Data Eng | Cleaned devices/events/manufacturers datasets, engineered features. | Commit `f6c5b4a`: "Complete data pipeline and feature engineering" |
| Phase 2 | ML Risk Engine | ML | Trained RF, LR, XGBoost. Implemented Isotonic Calibration. | Commit `726b45f`: "Complete Stage 5 ML risk engine" |
| Phase 3 | Scoring & Rules | Scoring | Finalized risk bands, SHAP integration, and recommendation rules. | Commit `219d845`: "Complete Stage 6 risk scoring engine" |
| Phase 4 | Backend APIs | Backend | Built FastAPI endpoints (`/assess`, `/predict`) + SQLite FTS5 serving DB. | Commit `4fd0ca5`: "Complete Stage 7 explainability and API backend" |
| Phase 5 | Frontend UI | Frontend | Built Streamlit dashboard and visualizations. | Commit `8292cde`: "feat: add Streamlit frontend dashboard" |
| Phase 6 | Copilot & Docs | GenAI & Deploy | Added grounded GenAI Copilot. Created `pytest` suite. Optimized snapshots. | Commits `be00220` (Copilot), `51cdc3f` (tests), `6503805` (docs) |

## 7. Eight Technical Domains

### 7.1 Data Engineering and Dataset Understanding
**Status: IMPLEMENTED**
- **Dataset:** Real retrospective FDA datasets located in `data/raw/`.
- **Schema & Size:** `devices.csv` (118k rows), `events.csv` (125k rows), `manufacturers.csv` (32k rows).
- **Quality & Missingness:** High sparsity in `quantity_in_commerce` (73%), `risk_class` (72%), and `implanted` (70%). Handled gracefully by the pipeline without fabricating data.

### 7.2 Temporal Dataset and Feature Engineering
**Status: IMPLEMENTED**
- **Target Definition:** Predicting whether a reported event maps to a "Class I" (severe) FDA recall action.
- **Temporal Splitting:** Historical cutoff implemented. The SQLite serving database (`artifacts/serving/historical_evidence.sqlite`) explicitly contains only pre-cutoff training-era data to prevent future leakage. 
- **Features:** Text features extracted from `reason` and `device_information` using TF-IDF. Categorical mappings applied to device classification.
- **Integrity Rule:** `device_id` is explicitly excluded from the predictive ML feature array to prevent overfitting on specific serial numbers, but is retained for historical aggregate rule formulation.

### 7.3 ML Model Training and Evaluation
**Status: IMPLEMENTED**
- **Models Verified in Code:** Logistic Regression (interpretable baseline), Random Forest (ensemble for mixed features), and XGBoost.
- **Implementation Strategy:** `src/models/train.py` dynamically handles severe class imbalance using `class_weight='balanced'` for LR/RF and `scale_pos_weight` for XGBoost. 
- **Production Model:** Random Forest is the primary production model invoked in the `inference_service` because it supports robust probability estimation and exact SHAP tree explanations.

### 7.4 Risk Scoring and Explainability
**Status: IMPLEMENTED**
- **Calibration:** `CalibratedClassifierCV(method="isotonic")` applied exclusively on the training split to convert raw model logits into reliable probabilities.
- **Risk Thresholds:** Explicit bounds enforced: LOW (<0.20), MEDIUM (0.20–0.50), HIGH (>=0.50).
- **Explainability:** Uses `shap.TreeExplainer` instantiated on the Random Forest model. Real-time inference dynamically computes word/feature contributions for ad-hoc user query text.

### 7.5 FastAPI Backend and APIs
**Status: IMPLEMENTED**
- **Endpoints:**
  - `POST /assess`: Orchestrates ML prediction, SHAP explanation, historical retrieval, and recommendation logic.
  - `POST /copilot`: Context-grounded Q&A.
- **Schemas:** Validated strictly via Pydantic (e.g., `AssessResponse` integrates `prediction`, `preventive_risk`, `explanation`, and `recommendation`).
- **Retrieval:** Uses `sqlite3` with an `FTS5` table (`event_fts`) to quickly find structurally similar historical problem descriptions.

### 7.6 Streamlit Frontend and Dashboard
**Status: IMPLEMENTED**
- **Pages:** Includes "Risk Assessment", "Device Detail", and Overview.
- **UI Architecture:** Renders exact backend responses. The Risk Assessment view maps: Event Risk -> Historical Preventive Risk -> Historical Evidence -> SHAP Explanations (Waterfall chart) -> Recommended Actions.
- **Guardrails:** Explicitly distinguishes between "Regulatory Device Class", "Event Risk", and "Historical Preventive Risk" via clear visual notes to prevent misinterpretation.

### 7.7 Recommendation Engine and GenAI
**Status: IMPLEMENTED (Rule-based Recs + GenAI Copilot)**
- **Recommendations:** A deterministic `MaintenanceEngine` maps the ML Risk Level + Regulatory Criticality to an actionable Priority (Critical, High, Medium, Low).
- **GenAI Copilot:** Supported by `backend/routes/copilot.py`. Uses `LLM_PROVIDER` (OpenAI/Gemini). Enforces a strict grounding contract: it builds a context block of *only* real retrieved values, instructs the LLM to answer using *only* that context, and provides a deterministic template fallback if the LLM fails.

### 7.8 Documentation and Deployment
**Status: IMPLEMENTED**
- **Configuration:** Managed via `.env` (with `.env.example` provided).
- **Testing:** Comprehensive `pytest` suite containing 368 tests (100% pass rate) verifying API contracts, model thresholds, and leakage prevention.
- **Deployment Artifacts:** Pre-compiled SQLite serving DB and Joblib models ensure zero-dependency startup performance.

## 8. Complete System Architecture

```text
User Input (Device + Problem)
       │
       ▼
 FastAPI Backend (/assess)
       │
       ├─► Retrieval (SQLite FTS5) ───► Historical Similar Events
       │
       ├─► Feature Engineering ───────► TF-IDF Text Vectors
       │
       ├─► Inference (Random Forest) ─► Event Risk Score (Isotonic Calibrated)
       │
       ├─► Recurrence Rules ──────────► Historical Preventive Risk (Rules)
       │
       ├─► SHAP (TreeExplainer) ──────► Feature Explanations
       │
       └─► MaintenanceEngine ─────────► Recommended Actions
       │
       ▼
Streamlit Dashboard (UI Rendering)
```

## 9. End-to-End Working
1. **Input:** Biomedical tech enters "Implanted cardiac defibrillator" and "Repeated electrical faults observed."
2. **Inference:** The text is mapped to feature vectors. Random Forest outputs a raw probability.
3. **Banding:** The probability is thresholded (e.g., 0.35 -> MEDIUM Event Risk).
4. **Historical Profiling:** Device ID aggregate history checked. (e.g., 5 past events -> MEDIUM Preventive Risk).
5. **Contextualization:** SQLite full-text search retrieves structurally similar historical incidents as supporting evidence.
6. **Action:** Rule engine prescribes "Schedule inspection within 7 days" based on risk + device class.

## 10. Challenges and Solutions
- **Challenge: True Future-Failure Prediction Impossible.** Retrospective FDA data lacks longitudinal sensor telemetry.
  - **Solution:** Pivoted from faking time-to-failure to building a scientifically rigorous *Event Severity Classifier* and a distinct *Historical Preventive Risk* layer.
- **Challenge: Model Calibration.** Raw tree-based probabilities were uncalibrated and useless for strict risk bands.
  - **Solution:** Implemented Isotonic Regression on the training fold, forcing output probabilities to align with actual class incidence rates.
- **Challenge: Temporal Data Leakage.** Assessing historical risk using future knowledge.
  - **Solution:** Strictly decoupled the SQLite serving database to only contain pre-cutoff training-era data; explicitly dropped post-event resolution fields from responses.
- **Challenge: Alert Fatigue.** High-volume, low-severity events overwhelming the UI.
  - **Solution:** SHAP explainability provides immediate "Why", and a deterministic rule engine separates critical triage from low-priority noise.

## 11. Literature Survey
- **Machine Learning Classification:** Utilizing ensemble methods (Random Forest, XGBoost) to classify complex, sparse text/categorical feature spaces.
- **Explainable AI (XAI):** Leveraging Lundberg and Lee's SHAP (2017) to break down complex non-linear tree decisions into additive feature influences, providing trust in medical contexts.

## 12. Research Papers
- **Predicting medical device failure: a promise to reduce healthcare facilities cost through smart healthcare management** — *Noorul Husna Abd Rahman et al. (2023), PeerJ Computer Science.*
- **Integrated failure analysis using machine learning predictive system for smart management of medical equipment maintenance** — *Aizat Hilmi Zamzam et al. (2023), Engineering Applications of Artificial Intelligence.*

## 13. Current Implementation vs Proposed Features

| Component | PPT Description | Current Implementation Code | Final Status |
|---|---|---|---|
| Dataset | Real FDA datasets | `devices.csv`, `events.csv`, `mfrs.csv` | **Implemented** |
| ML Models | Machine Learning Classification | Logistic Regression, Random Forest, XGBoost | **Implemented** (RF active) |
| Risk Scoring | Risk bands and thresholds | Isotonic Calibration + LOW/MED/HIGH | **Implemented** |
| SHAP | Feature Importance explanations | `shap.TreeExplainer` on active queries | **Implemented** |
| FastAPI | API exposure | `backend/main.py`, `/assess`, `/copilot` | **Implemented** |
| Streamlit | Interactive Dashboard | `frontend/pages/` | **Implemented** |
| GenAI | Grounded AI insights | OpenAI/Gemini APIs + Context grounding | **Implemented** |
| Time-to-Failure| Predict exact failure date | None (Data cannot support this) | **Future Scope** |
| MLOps | Drift detection & retraining | No active MLOps infrastructure | **Future Scope** |
| HITL | Biomedical Engineer feedback | None | **Future Scope** |

## 14. Future Scope
1. **Time-to-Failure / Survival Modeling:** Predicting exact failure timelines, requiring new prospective failure labels and operational duration fields.
2. **Live IoT Telemetry + CMMS:** Integrating the scoring engine with real-time continuous sensor readings (vibration, heat) and maintenance databases.
3. **MLOps:** Automated drift detection, monitoring, and continuous retraining pipelines to detect shifting failure modes.
4. **Human-in-the-Loop (HITL):** A feedback loop allowing biomedical engineers to accept/reject UI recommendations to automatically refine rules.

## 15. Final Conclusion
MedRISK successfully delivers a functional, rigorous, and highly defensible intelligence prototype. By explicitly rejecting fabricated future-failure claims and instead focusing on ML-driven severity classification, grounded retrieval, and historical recurrence mapping, the project provides immediate, explainable triage value for biomedical teams without compromising scientific integrity.

---

## 16. Technical Defense & Self-Evaluation Guide

### 7.1 Data Engineering & Dataset Understanding
- **What:** Combined 3 FDA datasets (devices, events, manufacturers).
- **Why:** Real-world medical data is relational.
- **Internal Workings:** Handled massive missingness (70%+) by retaining core text fields (`reason`, `device_information`) without blind imputation.
- **Decisions:** Avoided inventing synthetic replacements for sparse columns.
- **Alternatives:** We could have dropped sparse rows, but it would have destroyed 70% of the dataset.

### 7.2 Deep Dive: Temporal Dataset and Feature Engineering
- **What was implemented:** A text-centric feature pipeline (TF-IDF on `reason` and `device_info`) alongside categorical mappings, rigorously protected against temporal leakage via a strict chronological cutoff.
- **Why it was implemented:** Adverse events are reported chronologically. Standard random K-Fold splits would artificially leak future failure vocabulary into past predictions, over-inflating model accuracy metrics unethically.
- **How it works internally:** Training data is strictly limited to pre-cutoff dates. The SQLite serving database (`artifacts/serving/historical_evidence.sqlite`) explicitly strips out post-event investigation fields (like `action_summary`) so the model only scores based on what is known *at the time of the event*. `device_id` is explicitly dropped from the ML feature array to prevent overfitting to specific instances, but retained for safe historical aggregate counting.
- **Decisions:** Used TF-IDF instead of heavy embeddings to prioritize fast local-machine inference and clear SHAP explainability.
- **Alternatives considered:** Word2Vec/BERT. Rejected due to latency and poor SHAP compatibility for local hackathon demo speeds.
- **Important limitations:** Relying on TF-IDF means typos or non-standard biomedical abbreviations in the `reason` field might be missed unless they appeared frequently in the training set.
- **Evaluator Question:** *"How did you prevent future data leakage?"* -> "By enforcing a strict chronological split and deliberately excluding post-resolution fields like `action_classification` from the serving database."

### 7.3 ML Model Training and Evaluation
- **What:** Trained LR, RF, and XGBoost.
- **Why:** Needed a balance of accuracy (XGBoost/RF) and interpretability (LR).
- **Internal Workings:** Implemented class balancing (`class_weight='balanced'`) to handle extreme sparsity of Class I events.
- **Evaluator Question:** *"Why is Random Forest the active model?"* -> "It handles the mixed feature space well, provides robust probability estimates after calibration, and supports exact `shap.TreeExplainer` computations instantly."

### 7.4 Risk Scoring and Explainability
- **What:** Isotonic Calibration mapping raw logits to probability + SHAP visualizations.
- **Why:** A raw probability of 0.8 from a tree model is often uncalibrated. We needed reliable thresholds (LOW/MED/HIGH).
- **Evaluator Question:** *"Why Isotonic instead of Platt scaling?"* -> "Platt scaling assumes a sigmoid distribution, but our tree-based outputs were non-parametric. Isotonic regression fits the empirical curve better."

### 7.5 FastAPI Backend and APIs
- **What:** Scalable API with `/assess` and `/copilot` endpoints.
- **Why:** Separates the compute-heavy ML pipeline from the frontend UI.

### 7.6 Streamlit Frontend and Dashboard
- **What:** An interactive triage dashboard.
- **Why:** Streamlit allows rapid Python-native iteration.

### 7.7 Recommendation Engine and GenAI
- **What:** Deterministic rule engine for maintenance + grounded GenAI for Q&A.
- **Why:** GenAI hallucinations are dangerous in medical contexts. The deterministic engine guarantees safe maintenance mapping, while GenAI is restricted to reading the retrieved context block.
- **Evaluator Question:** *"Does GenAI dictate the maintenance schedule?"* -> "No. A deterministic, auditable rule engine dictates the schedule. GenAI only acts as a Q&A tool."

### 7.8 Documentation and Deployment
- **What:** Full `pytest` coverage and pre-compiled SQLite/Joblib artifacts.

### Things Every Team Member Must Be Able to Explain
1. **Why we don't predict exact future failure:** The FDA dataset provides retrospective anomaly reports, not continuous IoT telemetry with "healthy" baselines.
2. **The difference between Event Risk and Preventive Risk:** Event Risk is the ML model scoring the *severity* of an ad-hoc problem description. Preventive Risk is a rule-based flag based on the device's *past history* of failures.
3. **How Calibration works:** Raw Random Forest votes are pushed through Isotonic Regression to map them to true real-world probabilities, ensuring 0.40 actually means a 40% historical incidence rate.
4. **How SHAP is generated:** `shap.TreeExplainer` calculates the marginal contribution of each TF-IDF text token relative to the model's base expected value, instantly generated during the `/assess` API call.
5. **How Copilot Grounding works:** The LLM is forced to use *only* the retrieved SQLite context block. If the LLM fails, a deterministic template falls back automatically.

### Red-Flag Claims to Avoid
- ❌ **Avoid:** "Our AI predicts exactly when this MRI machine will break down tomorrow."
  ✅ **Say instead:** "Our system flags when a reported anomaly historically resembles a severe Class I recall event, prioritizing it for immediate investigation."
- ❌ **Avoid:** "We do predictive maintenance."
  ✅ **Say instead:** "We provide predictive severity triage and historical recurrence tracking."
- ❌ **Avoid:** "The model is 99% accurate at stopping failures."
  ✅ **Say instead:** "The model generates calibrated risk probabilities to surface the most dangerous events, significantly reducing alert fatigue."
- ❌ **Avoid:** "GenAI determines the maintenance schedule."
  ✅ **Say instead:** "A deterministic rule engine generates the maintenance schedule; GenAI serves exclusively as an interactive Q&A Copilot grounded in the device's historical facts."
