# Development Lineage & Initial Evaluation Benchmark

## 1. Provenance of the +28.4% Metric in Initial Development
The **+28.4% Qini AUUC lift** reported during initial development was produced on a synthetic RCT reference dataset (data/dev_benchmark_synthetic.csv, generated when the raw 13.9M Criteo research archive was temporarily unavailable locally), using the relative Qini ratio formulation:

\\text{Relative Qini Ratio} = \\frac{\\text{AUUC}_{\\text{model}} - \\text{AUUC}_{\\text{random}}}{|\\text{AUUC}_{\\text{random}}|}

### Initial Development Results Table (Synthetic Reference Partition):
| Candidate Architecture | AUUC | Relative Qini Lift | Top-20% Targeting Lift |
|---|:---:|:---:|:---:|
| Random Allocation Baseline | -0.94 | 0.0% | 0.0% |
| Single-Model S-Learner | 2.43 | +11.6% | -73.2% |
| Two-Model T-Learner | 1.83 | -16.0% | +41.8% |
| **True Doubly Robust AIPW (5-Fold DML)** | **2.49** | **+28.4%** | **+21.4%** |

---

## 2. Transition to the Official Criteo AI Uplift Benchmark Archive
Following code review and verification hardening:
1. The official 13.9M-record Criteo research archive (criteo-research-uplift-v2.1.csv.gz, 311 MB) was ingested.
2. A verified stratified 100,000-record randomized trial sample (85,000 Treated / 15,000 Control across all 12 continuous covariates) was extracted into data/criteo_uplift_processed.csv.
3. To eliminate small-denominator division artifacts, the evaluation was standardized on the **Radcliffe Centered Qini Area** ($\\text{AUUC}_{\\text{model}} - \\text{AUUC}_{\\text{random}}$) on the isit target with 500-sample non-parametric bootstrap confidence intervals.

For final, frozen benchmark results on the real Criteo dataset, see 
esults/final_benchmark.txt and the main README.md.
