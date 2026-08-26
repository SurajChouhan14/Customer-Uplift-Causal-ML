"""
Automated Unit Test Suite for Customer Uplift Modeling & Causal ML Engine.
Verifies Criteo Dataset Ingestion, S-Learner, T-Learner, Doubly Robust AIPW (5-Fold DML), and Qini Evaluation.
"""

import unittest
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import CriteoDataLoader
from src.causal_engine import SingleModelSLearner, TwoModelTLearner, TrueDoublyRobustAIPW
from src.evaluate_metrics import compute_qini_curve


class TestCustomerUpliftEngine(unittest.TestCase):
    """
    Unit test cases for customer causal uplift modeling and evaluation engine.
    """

    @classmethod
    def setUpClass(cls):
        cls.loader = CriteoDataLoader(data_dir="data", sample_size=2000, random_state=42)
        df = cls.loader.load_processed_data()
        
        feature_cols = [f'f{i}' for i in range(12)]
        X = df[feature_cols].values
        T = df['treatment'].values
        Y = df['conversion'].values

        cls.X_train, cls.X_test, cls.T_train, cls.T_test, cls.Y_train, cls.Y_test = train_test_split(
            X, T, Y, test_size=0.25, random_state=42, stratify=T
        )

    def test_data_loader_partitions(self):
        """Verify feature dimensions, treatment indicator binary values, and target outcome."""
        self.assertGreaterEqual(len(self.X_train), 1000)
        self.assertEqual(self.X_train.shape[1], 12)
        self.assertTrue(set(np.unique(self.T_train)).issubset({0, 1}))
        self.assertTrue(set(np.unique(self.Y_train)).issubset({0, 1}))

    def test_s_learner_fit_and_predict(self):
        """Verify Single-Model S-Learner generates bounded uplift predictions."""
        s_learner = SingleModelSLearner(n_estimators=20, max_depth=3, random_state=42)
        s_learner.fit(self.X_train, self.T_train, self.Y_train)
        preds = s_learner.predict_uplift(self.X_test)

        self.assertEqual(len(preds), len(self.X_test))
        self.assertTrue(np.all(np.isfinite(preds)))
        self.assertTrue(np.all((preds >= -1.0) & (preds <= 1.0)))

    def test_t_learner_fit_and_predict(self):
        """Verify Two-Model T-Learner dual response fitting and prediction."""
        t_learner = TwoModelTLearner(n_estimators=20, max_depth=3, random_state=42)
        t_learner.fit(self.X_train, self.T_train, self.Y_train)
        preds = t_learner.predict_uplift(self.X_test)

        self.assertEqual(len(preds), len(self.X_test))
        self.assertTrue(np.all(np.isfinite(preds)))
        self.assertTrue(np.all((preds >= -1.0) & (preds <= 1.0)))

    def test_doubly_robust_aipw_cross_fitting(self):
        """Verify 3-Fold Cross-Fitting Doubly Robust AIPW CATE model."""
        dr_aipw = TrueDoublyRobustAIPW(n_folds=3, n_estimators=20, max_depth=3, random_state=42)
        dr_aipw.fit(self.X_train, self.T_train, self.Y_train)
        preds = dr_aipw.predict_uplift(self.X_test)

        self.assertEqual(len(preds), len(self.X_test))
        self.assertTrue(np.all(np.isfinite(preds)))

    def test_qini_and_auuc_evaluation(self):
        """Verify population-adjusted Qini curve calculation and AUUC metric."""
        dummy_uplift = np.random.normal(0.02, 0.05, len(self.Y_test))
        qini_curve, random_curve, auuc, qini_lift = compute_qini_curve(
            self.Y_test, self.T_test, dummy_uplift, n_bins=10
        )

        self.assertEqual(len(qini_curve), 11)
        self.assertEqual(len(random_curve), 11)
        self.assertTrue(np.isfinite(auuc))
        self.assertTrue(np.isfinite(qini_lift))


if __name__ == '__main__':
    unittest.main()
