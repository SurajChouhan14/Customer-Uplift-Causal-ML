# Customer Uplift Modeling & Causal ML Engine
> **Enterprise Double Machine Learning (DML) & Doubly Robust AIPW for Conditional Average Treatment Effect (CATE) Estimation**  
> *Targeted Intervention Policy Optimization on the Criteo AI Uplift Benchmark*

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/tests-9%20passed-brightgreen.svg)]()
[![Dataset](https://img.shields.io/badge/Criteo%20AI%20Uplift-100k%20RCT-orange.svg)]()

---

## 🎯 Executive Overview & Causal Architecture
Traditional machine learning predicts customer response probabilities ($P(Y=1|X)$), which results in wasteful marketing spend on "Sure Things" (who convert anyway) and "Lost Causes" (who never convert). 

This repository implements **Individual Treatment Effect (ITE)** and **Conditional Average Treatment Effect (CATE)** estimation using **Chernozhukov Double Machine Learning (DML)** and **Doubly Robust Augmented Inverse Probability Weighting (AIPW)** with 5-fold cross-fitting on 100,000 randomized trial records from the canonical **Criteo AI Uplift Benchmark (v2.1)**.

```
                  ┌────────────────────────────────────────────────────────┐
                  │   Criteo AI Uplift Benchmark (100,000 RCT Records)     │
                  │   85% Treated / 15% Control across 12 Continuous X     │
                  └───────────────────────────┬────────────────────────────┘
                                              │
                     ┌────────────────────────┴────────────────────────┐
                     ▼                                                 ▼
        ┌─────────────────────────┐                       ┌─────────────────────────┐
        │ Propensity Forest e(X)  │                       │ Outcome Regressors μ(X) │
        │ P(Treatment | Features) │                       │ E(Visit | X, T)         │
        └────────────┬────────────┘                       └────────────┬────────────┘
                     │                                                 │
                     └────────────────────────┬────────────────────────┘
                                              ▼
                  ┌────────────────────────────────────────────────────────┐
                  │     Doubly Robust AIPW Cross-Fitting (5 Folds)         │
                  │     Γ_i = μ1(X) - μ0(X) + (T*(Y-μ1)/e) - ((1-T)*(Y-μ0)/(1-e)) │
                  └───────────────────────────┬────────────────────────────┘
                                              ▼
                  ┌────────────────────────────────────────────────────────┐
                  │    Radcliffe Centered Qini Curve & Bootstrap CIs       │
                  │    Dynamic Decile Cohorts (Persuadables vs Dogs)       │
                  └───────────────────────────┬────────────────────────────┘
                                              ▼
                  ┌────────────────────────────────────────────────────────┐
                  │    Production FastAPI Microservice (0.42ms Latency)    │
                  └────────────────────────────────────────────────────────┘
```

---

## 📊 Empirical Tournament Results (Radcliffe Centered Qini)
Evaluated on an unseen **25,000-record holdout test set** using 10 policy evaluation deciles with 500-sample non-parametric bootstrap confidence intervals:

| Candidate Model Architecture | Train Qini | Test Qini (Area Between Curves) | 95% Bootstrap CI | Top-Decile Gain |
|---|:---:|:---:|:---:|:---:|
| **Single-Model S-Learner (Baseline)** | -8.50 | -20.10 | [-32.3, +33.7] | -22.1 visits |
| **Two-Model T-Learner (Dual Surface)** | 399.90 | 18.03 | [-33.3, +65.7] | -14.7 visits |
| **True Doubly Robust AIPW (5-Fold DML)** 🏆 | **279.30** | **+22.70** | **[-24.7, +71.7]** | **+2.6 incremental visits** |

*Champion model artifact is automatically exported to `models/champion_uplift_model.joblib` and served via FastAPI.*

> **Lineage Note:** For documentation of the initial development-phase relative Qini benchmark (+28.4%), see [`docs/initial_evaluation.md`](docs/initial_evaluation.md).

---

## 🔬 Customer Segmentation Strategy (Dynamic Decile Thresholds)
Using empirical holdout CATE percentiles, the engine segments customers into 4 actionable targeting quadrants:
1. **Persuadables ($CATE > p_{75}$):** Positive incremental responders $	o$ **Target with promotional campaigns.**
2. **Sure Things ($p_{25} \le CATE \le p_{75}$):** Organic converters $	o$ **Do not subsidize.**
3. **Lost Causes ($p_{10} \le CATE < p_{25}$):** Non-responsive users $	o$ **Zero ad spend.**
4. **Sleeping Dogs ($CATE < p_{10}$):** Negative treatment effects $	o$ **Suppress all outreach.**

---

## ⚡ Quickstart (2-Minute Execution)

### 1. Installation
```bash
git clone https://github.com/SurajChouhan14/Customer-Uplift-Causal-ML.git
cd Customer-Uplift-Causal-ML
pip install -r requirements.txt
```

### 2. Run Tournament Pipeline
```bash
python run_pipeline.py
```

### 3. Run Unit Tests (9/9 Passing)
```bash
python test_customer_uplift.py
```

### 4. Launch FastAPI Microservice
```bash
uvicorn serve_api:app --host 0.0.0.0 --port 8000
```

---

## 🧪 Real-Time REST API Endpoints

### Health & Champion Metadata
```bash
curl -X GET http://localhost:8000/health
```

### Single Customer CATE Prediction
```bash
curl -X POST http://localhost:8000/predict_uplift   -H "Content-Type: application/json"   -d '{"features": [21.5, 10.1, 8.3, 4.3, 10.3, 4.0, -4.4, 5.1, 3.9, 14.4, 5.3, -0.17]}'
```

**Response:**
```json
{
  "predicted_cate": 0.00341,
  "segment": "Persuadable",
  "action": "Target with promotional campaign",
  "model_version": "1.2.0",
  "inference_time_ms": 0.42
}
```
