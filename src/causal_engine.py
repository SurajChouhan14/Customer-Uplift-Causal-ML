"""
Causal Inference Estimation Engine.

Implements Meta-Learners (S-Learner, T-Learner) and
True Doubly Robust Augmented Inverse Probability Weighting (AIPW) with
Chernozhukov K-Fold Cross-Fitting (Double Machine Learning) for
Individual Treatment Effect (ITE) and Conditional Average Treatment Effect (CATE).
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import StratifiedKFold

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


class SingleModelSLearner:
    """
    Single-Model S-Learner Baseline.
    
    Includes the treatment indicator T as an additional feature in a single predictive model:
        mu(x, t) = E[Y | X=x, T=t]
        tau(x)   = mu(x, 1) - mu(x, 0)
    """

    def __init__(self, n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42):
        self.random_state = random_state
        if HAS_LIGHTGBM:
            self.model = lgb.LGBMClassifier(
                n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
                random_state=random_state, verbose=-1
            )
        else:
            self.model = GradientBoostingClassifier(
                n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
                random_state=random_state
            )

    def fit(self, X, T, Y):
        """
        Fits single response model on concatenated feature and treatment matrix [X, T].
        """
        X_arr = np.asarray(X)
        T_arr = np.asarray(T).reshape(-1, 1)
        Y_arr = np.asarray(Y)

        XT = np.hstack([X_arr, T_arr])
        self.model.fit(XT, Y_arr)
        return self

    def predict_uplift(self, X):
        """
        Estimates uplift by contrasting counterfactual predictions: mu(x, 1) - mu(x, 0).
        """
        X_arr = np.asarray(X)
        n = len(X_arr)

        ones = np.ones((n, 1))
        zeros = np.zeros((n, 1))

        X_treated = np.hstack([X_arr, ones])
        X_control = np.hstack([X_arr, zeros])

        p1 = self.model.predict_proba(X_treated)[:, 1]
        p0 = self.model.predict_proba(X_control)[:, 1]
        return p1 - p0


class TwoModelTLearner:
    """
    Two-Model T-Learner for Heterogeneous Treatment Effect Estimation.
    
    Fits separate response surfaces for the treated and control populations:
        mu_1(x) = E[Y | X=x, T=1]
        mu_0(x) = E[Y | X=x, T=0]
        tau(x)  = mu_1(x) - mu_0(x)
    """

    def __init__(self, n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42):
        self.random_state = random_state
        if HAS_LIGHTGBM:
            self.model_treated = lgb.LGBMClassifier(
                n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
                random_state=random_state, verbose=-1
            )
            self.model_control = lgb.LGBMClassifier(
                n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
                random_state=random_state, verbose=-1
            )
        else:
            self.model_treated = GradientBoostingClassifier(
                n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
                random_state=random_state
            )
            self.model_control = GradientBoostingClassifier(
                n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
                random_state=random_state
            )

    def fit(self, X, T, Y):
        """
        Fits separate conditional models on treated and control cohorts.
        """
        X_arr = np.asarray(X)
        T_arr = np.asarray(T)
        Y_arr = np.asarray(Y)

        mask_treated = (T_arr == 1)
        mask_control = (T_arr == 0)

        self.model_treated.fit(X_arr[mask_treated], Y_arr[mask_treated])
        self.model_control.fit(X_arr[mask_control], Y_arr[mask_control])
        return self

    def predict_uplift(self, X):
        """
        Estimates individual treatment uplift tau_hat(x).
        """
        X_arr = np.asarray(X)
        p1 = self.model_treated.predict_proba(X_arr)[:, 1]
        p0 = self.model_control.predict_proba(X_arr)[:, 1]
        return p1 - p0


class TrueDoublyRobustAIPW:
    """
    True Doubly Robust Augmented Inverse Probability Weighting (AIPW) with
    Chernozhukov K-Fold Cross-Fitting (Double Machine Learning).
    
    1. Cross-Fitting (Sample Splitting):
       Splits data into K folds. Fits nuisance models (propensity e(X) and conditional outcomes mu_0(X), mu_1(X))
       on K-1 folds, predicting out-of-fold nuisance estimates on the holdout fold to prevent overfitting.
       
    2. Pseudo-Outcome Construction:
       Gamma_i = (mu_1(X_i) - mu_0(X_i)) + [T_i * (Y_i - mu_1(X_i)) / e(X_i)] - [(1 - T_i) * (Y_i - mu_0(X_i)) / (1 - e(X_i))]
       
    3. Final Stage-2 Causal Effect Regressor:
       Fits a final Gradient Boosted Tree model on (X, Gamma) to predict pointwise uplift tau_hat(X).
    """

    def __init__(self, n_folds=5, n_estimators=100, learning_rate=0.05, max_depth=4, clip_propensity=0.01, random_state=42):
        self.n_folds = n_folds
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.clip_propensity = clip_propensity
        self.random_state = random_state
        
        # Final Stage-2 Causal Regressor targeting Gamma
        if HAS_LIGHTGBM:
            self.final_causal_regressor = lgb.LGBMRegressor(
                n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
                random_state=random_state, verbose=-1
            )
        else:
            self.final_causal_regressor = GradientBoostingRegressor(
                n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
                random_state=random_state
            )

    def _get_nuisance_models(self):
        prop_model = LogisticRegression(max_iter=1000, random_state=self.random_state)
        if HAS_LIGHTGBM:
            out_t = lgb.LGBMRegressor(n_estimators=self.n_estimators, learning_rate=self.learning_rate, max_depth=self.max_depth, random_state=self.random_state, verbose=-1)
            out_c = lgb.LGBMRegressor(n_estimators=self.n_estimators, learning_rate=self.learning_rate, max_depth=self.max_depth, random_state=self.random_state, verbose=-1)
        else:
            out_t = GradientBoostingRegressor(n_estimators=self.n_estimators, learning_rate=self.learning_rate, max_depth=self.max_depth, random_state=self.random_state)
            out_c = GradientBoostingRegressor(n_estimators=self.n_estimators, learning_rate=self.learning_rate, max_depth=self.max_depth, random_state=self.random_state)
        return prop_model, out_t, out_c

    def fit(self, X, T, Y):
        """
        Executes K-Fold Cross-Fitting to compute un-biased pseudo-outcomes Gamma_i,
        then trains the final Stage-2 Causal Regressor on (X, Gamma).
        """
        X_arr = np.asarray(X)
        T_arr = np.asarray(T)
        Y_arr = np.asarray(Y)
        n = len(X_arr)

        # Out-of-fold nuisance predictions
        oof_propensity = np.zeros(n)
        oof_mu1 = np.zeros(n)
        oof_mu0 = np.zeros(n)

        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        for train_idx, val_idx in skf.split(X_arr, T_arr):
            X_tr, X_val = X_arr[train_idx], X_arr[val_idx]
            T_tr, T_val = T_arr[train_idx], T_arr[val_idx]
            Y_tr, Y_val = Y_arr[train_idx], Y_arr[val_idx]

            prop_model, out_t, out_c = self._get_nuisance_models()

            # 1. Fit Propensity on training fold
            prop_model.fit(X_tr, T_tr)
            p_val = prop_model.predict_proba(X_val)[:, 1]
            oof_propensity[val_idx] = np.clip(p_val, self.clip_propensity, 1.0 - self.clip_propensity)

            # 2. Fit Outcome Models on treated and control subsets of training fold
            mask_t = (T_tr == 1)
            mask_c = (T_tr == 0)

            out_t.fit(X_tr[mask_t], Y_tr[mask_t])
            out_c.fit(X_tr[mask_c], Y_tr[mask_c])

            oof_mu1[val_idx] = out_t.predict(X_val)
            oof_mu0[val_idx] = out_c.predict(X_val)

        # 3. Construct Exact Doubly Robust Pseudo-Outcomes Gamma_i
        gamma = (
            (oof_mu1 - oof_mu0)
            + (T_arr * (Y_arr - oof_mu1) / oof_propensity)
            - ((1 - T_arr) * (Y_arr - oof_mu0) / (1.0 - oof_propensity))
        )

        # 4. Train Final Stage-2 Causal Effect Model on (X, Gamma)
        self.final_causal_regressor.fit(X_arr, gamma)
        return self

    def predict_uplift(self, X):
        """
        Predicts Conditional Average Treatment Effect (CATE) tau_hat(X) using the Stage-2 model.
        """
        X_arr = np.asarray(X)
        return self.final_causal_regressor.predict(X_arr)
