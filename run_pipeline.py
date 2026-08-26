"""
Main Execution Pipeline for Customer Uplift Modeling & Causal ML Tournament.
Trains S-Learner, T-Learner, and Doubly Robust AIPW on Criteo AI Uplift Benchmark.
"""

import os
import sys

# Ensure directory is on python search path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')

# Handle both flat-folder and src-packaged executions
try:
    from src.data_loader import CriteoDataLoader
    from src.causal_engine import SingleModelSLearner, TwoModelTLearner, TrueDoublyRobustAIPW
    from src.evaluate_metrics import compute_qini_curve
except ImportError:
    from data_loader import CriteoDataLoader
    from causal_engine import SingleModelSLearner, TwoModelTLearner, TrueDoublyRobustAIPW
    from evaluate_metrics import compute_qini_curve

from sklearn.model_selection import train_test_split


def main():
    print("=" * 95)
    print("MATHEMATICALLY EXACT CAUSAL MODEL SELECTION TOURNAMENT (CHERNOZHUKOV DML / AIPW)")
    print("Dataset: Criteo AI Uplift Benchmark (100,000 Records) | Framework: Double Machine Learning")
    print("=" * 95)

    loader = CriteoDataLoader(data_dir="data", sample_size=100000, random_state=42)
    print("\n[1/4] Loading and standardizing preprocessed covariate matrices...")
    df = loader.load_processed_data()

    feature_cols = [f'f{i}' for i in range(12)]
    X = df[feature_cols].values
    T = df['treatment'].values
    Y = df['conversion'].values

    X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(
        X, T, Y, test_size=0.25, random_state=42, stratify=T
    )
    print(f"      Training partition: {len(X_train):,} | Holdout test partition: {len(X_test):,}")
    print(f"      Treatment distribution: {T_train.mean()*100:.1f}% Treated / {(1-T_train.mean())*100:.1f}% Control.")

    print("\n[2/4] Training candidate causal architectures...")
    models = {
        "Single-Model S-Learner (Baseline)": SingleModelSLearner(n_estimators=100, random_state=42),
        "Two-Model T-Learner (Dual Surface)": TwoModelTLearner(n_estimators=100, random_state=42),
        "True Doubly Robust AIPW (5-Fold DML)": TrueDoublyRobustAIPW(n_folds=5, n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)
    }

    predictions = {}
    for name, model in models.items():
        print(f"      -> Training {name}...")
        model.fit(X_train, T_train, Y_train)
        predictions[name] = model.predict_uplift(X_test)
        print(f"         {name} convergence reached.")

    print("\n[3/4] Evaluating exact population-adjusted Qini Curves and AUUC on unseen test set...")
    results = []
    
    # Baseline random curve
    _, rand_curve, rand_auuc, _ = compute_qini_curve(Y_test, T_test, np.zeros(len(Y_test)))
    results.append({"Model": "Random Allocation Policy (Baseline)", "AUUC": rand_auuc, "Lift": "0.0%", "Top20": "0.0%"})

    for name, preds in predictions.items():
        q_curve, _, auuc, lift_ratio = compute_qini_curve(Y_test, T_test, preds)
        
        # Calculate Top 20% targeting efficiency
        n_top20 = len(preds) // 5
        top20_idx = np.argsort(preds)[::-1][:n_top20]
        y_top20_t = Y_test[top20_idx][T_test[top20_idx] == 1].sum()
        y_top20_c = Y_test[top20_idx][T_test[top20_idx] == 0].sum()
        scale = (T_test.sum() / (len(T_test) - T_test.sum()))
        empirical_lift_top20 = y_top20_t - (y_top20_c * scale)
        random_lift_top20 = (Y_test[T_test == 1].sum() - Y_test[T_test == 0].sum() * scale) * 0.20
        top20_gain = ((empirical_lift_top20 - random_lift_top20) / (abs(random_lift_top20) + 1e-8)) * 100

        results.append({
            "Model": name,
            "AUUC": auuc,
            "Lift": f"{lift_ratio * 100:+.1f}%",
            "Top20": f"{top20_gain:+.1f}%"
        })

    print("\n" + "=" * 95)
    print("EMPIRICAL BENCHMARK TOURNAMENT RESULTS TABLE (POPULATION-ADJUSTED QINI METRICS)")
    print("=" * 95)
    print(f"{'Candidate Model Architecture':<42} | {'AUUC':<8} | {'Qini Lift':<12} | {'Top-20% Lift':<14}")
    print("-" * 95)
    for r in results:
        print(f"{r['Model']:<42} | {r['AUUC']:<8.2f} | {r['Lift']:<12} | {r['Top20']:<14}")
    print("=" * 95)

    best_model = max([r for r in results if "Random" not in r["Model"]], key=lambda x: x["AUUC"])
    print(f"\n AUTOMATED DECISION MATRIX: CHAMPION MODEL SELECTED -> '{best_model['Model']}'")
    print(f"   Optimal AUUC: {best_model['AUUC']:.2f} | Normalized Qini Lift: {best_model['Lift']}")
    print("=" * 95 + "\n")


if __name__ == '__main__':
    import numpy as np
    main()
