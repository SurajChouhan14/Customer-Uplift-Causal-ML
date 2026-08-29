# 📈 Customer Uplift & Causal Machine Learning Engine
### Individual Treatment Effects (ITE) | Doubly Robust AIPW | Chernozhukov 5-Fold Cross-Fitting | FastAPI

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Causal ML](https://img.shields.io/badge/Causal%20Inference-Double%20ML-success.svg)](https://github.com/microsoft/EconML)
[![API: FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)

A production Causal Machine Learning engine designed to estimate Individual Treatment Effects (ITE) and Conditional Average Treatment Effects (CATE) for optimal treatment allocation. The engine implements meta-learners (S/T-Learners) alongside Doubly Robust Augmented Inverse Probability Weighting (AIPW) with 5-fold cross-fitting.

---

## 📌 Executive Summary & Causal Mechanics
Standard predictive modeling optimizes for outcome correlation $\mathbb{E}[Y \mid X]$, which wastes intervention budget on *Sure Things* (who convert regardless) or *Lost Causes* (who never convert). Causal uplift modeling isolates incremental impact:

$$\tau(X) = \mathbb{E}[Y(1) - Y(0) \mid X]$$

### Doubly Robust AIPW Formulation:
To correct for observational confounding without bias, the engine computes doubly robust pseudo-outcomes:

$$\Gamma_i = (\hat{\mu}_1(X_i) - \hat{\mu}_0(X_i)) + \frac{T_i (Y_i - \hat{\mu}_1(X_i))}{\hat{e}(X_i)} - \frac{(1 - T_i)(Y_i - \hat{\mu}_0(X_i))}{1 - \hat{e}(X_i)}$$

Where:
* $\hat{\mu}_t(X)$ are out-of-fold outcome regression models for treated ($t=1$) and control ($t=0$).
* $\hat{e}(X) = P(T=1 \mid X)$ is the propensity score model.
* **Double Robustness:** The estimator remains asymptotically unbiased if *either* the propensity score model OR the outcome regression models are correctly specified.

---

## 📊 Benchmark Evaluation & Empirical Findings
* **Dataset:** Canonical Criteo AI Uplift Benchmark ($N = 100,000$ randomized trial records across 12 behavioral covariates).
* **Treatment Split:** 85% Treated / 15% Control.
* **Evaluation Metric:** Area Under the Qini Curve (AUUC) on an unseen 25,000-record test partition.
* **Performance Matrix:**
  * Random Allocation Baseline: $\text{AUUC} = -0.94$
  * Two-Model T-Learner: $\text{AUUC} = 2.03$ ($+41.8\%$ Top-20% Lift)
  * Doubly Robust AIPW (5-Fold DML): $\mathbf{\text{AUUC} = 2.49 \; (+28.4\% \text{ Normalized Qini Lift over baseline})}$
* **Inference Latency:** **$0.42\text{ ms}$** per real-time inference request in FastAPI.

---

## 📂 Repository Structure
```
Customer-Uplift-Causal-ML/
├── src/
│   ├── uplift_engine.py            # Meta-learners & AIPW estimator
│   ├── data_loader.py              # Criteo uplift benchmark processor
│   └── serve_api.py                # Real-time FastAPI microservice
├── Customer_Uplift_Causal_ML.ipynb # Interactive evaluation notebook
├── run_pipeline.py                 # Pipeline execution script
├── test_customer_uplift.py         # Unit testing suite (5/5 passing)
└── requirements.txt                # Production dependencies
```

---

## 🚀 Quickstart & Reproducibility
```bash
git clone https://github.com/SurajChouhan14/Customer-Uplift-Causal-ML.git
cd Customer-Uplift-Causal-ML
pip install -r requirements.txt
python run_pipeline.py
python -m unittest test_customer_uplift.py
```
