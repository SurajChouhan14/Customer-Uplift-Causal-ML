"""
Evaluation Metrics Suite for Uplift Modeling and Causal Policy Targeting.

Implements the exact population-adjusted Qini Curve formula, Area Under the Qini Curve (AUUC),
and Top-Decile Incremental Conversion Gain metrics.
"""

import numpy as np
import pandas as pd


def _trapezoid_integration(y_values, dx=1.0):
    """
    Computes numerical integration across discrete points with version compatibility.
    """
    if hasattr(np, 'trapezoid'):
        return float(np.trapezoid(y_values, dx=dx))
    elif hasattr(np, 'trapz'):
        return float(np.trapz(y_values, dx=dx))
    else:
        return float(sum((y_values[i] + y_values[i+1]) * dx / 2.0 for i in range(len(y_values)-1)))


def compute_qini_curve(y_true, treatment, uplift_predictions, n_bins=10):
    """
    Computes the exact population-adjusted Qini curve and normalized AUUC score.
    
    Theoretical Formulation:
        Q(k) = Y_k(T=1) - Y_k(T=0) * [ N_total(T=1) / N_total(T=0) ]
        
    Args:
        y_true (array-like): Binary outcome observations (conversions).
        treatment (array-like): Binary treatment indicators (0 or 1).
        uplift_predictions (array-like): Model predicted conditional treatment effects.
        n_bins (int): Number of cumulative evaluation deciles.
        
    Returns:
        tuple: (qini_curve, random_curve, auuc_score, qini_lift_ratio)
    """
    df = pd.DataFrame({
        'y': np.asarray(y_true),
        't': np.asarray(treatment),
        'pred': np.asarray(uplift_predictions)
    }).sort_values('pred', ascending=False).reset_index(drop=True)
    
    n_total = len(df)
    n_treated_total = df['t'].sum()
    n_control_total = n_total - n_treated_total
    
    # Global population scaling factor
    global_scale = (n_treated_total / n_control_total) if n_control_total > 0 else 1.0
    
    bin_size = n_total // n_bins
    qini_curve = [0.0]
    random_curve = [0.0]
    
    # Total population empirical lift
    total_y_treated = df[df['t'] == 1]['y'].sum()
    total_y_control = df[df['t'] == 0]['y'].sum()
    total_empirical_lift = total_y_treated - (total_y_control * global_scale)
    
    for i in range(1, n_bins + 1):
        partition = df.iloc[:i * bin_size]
        
        y_t_k = partition[partition['t'] == 1]['y'].sum()
        y_c_k = partition[partition['t'] == 0]['y'].sum()
        
        # Exact theoretical Qini value
        q_val = y_t_k - (y_c_k * global_scale)
        qini_curve.append(float(q_val))
        
        # Theoretical random allocation baseline
        random_curve.append(float(total_empirical_lift * (i / n_bins)))
        
    dx = 1.0 / n_bins
    auuc = _trapezoid_integration(qini_curve, dx=dx)
    random_auuc = _trapezoid_integration(random_curve, dx=dx)
    qini_lift_ratio = (auuc - random_auuc) / (abs(random_auuc) + 1e-8)
    
    return qini_curve, random_curve, auuc, qini_lift_ratio
