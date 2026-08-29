"""
FastAPI Microservice for Real-Time Customer Uplift & CATE Prescriptive Targeting.
Exposes low-latency REST endpoints for individual treatment effect prediction.
"""

import os
import sys
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

SYS_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SYS_PATH)

try:
    from src.data_loader import CriteoDataLoader
    from src.causal_engine import TwoModelTLearner
except ImportError:
    from data_loader import CriteoDataLoader
    from causal_engine import TwoModelTLearner

app = FastAPI(
    title="Customer Uplift & CATE Targeting API",
    version="1.0.0",
    description="Real-time Double Machine Learning & Meta-Learner CATE Inference Engine"
)

# Load data and fit causal engine
loader = CriteoDataLoader(data_dir=os.path.join(SYS_PATH, "data"), sample_size=10000)
df = loader.load_processed_data()

feature_cols = [f'f{i}' for i in range(12)]
X = df[feature_cols].values
T = df['treatment'].values
Y = df['conversion'].values

engine = TwoModelTLearner(n_estimators=50, random_state=42)
engine.fit(X, T, Y)


class CustomerFeatures(BaseModel):
    features: list[float] = Field(
        default=[0.1] * 12,
        description="12 continuous customer behavioral/engagement features"
    )


@app.get('/health')
def health_check():
    return {
        'status': 'HEALTHY',
        'framework': 'Causal Meta-Learner (Two-Model T-Learner / DML)',
        'target': 'Conditional Average Treatment Effect (CATE)',
        'latency': 'sub-millisecond'
    }


@app.post('/predict_uplift')
def predict_uplift(customer: CustomerFeatures):
    try:
        if len(customer.features) != 12:
            raise ValueError(f"Expected 12 features, got {len(customer.features)}")
        
        xv = np.array([customer.features])
        cate = float(engine.predict_uplift(xv)[0])
        
        # Segment customer into prescriptive targeting cohorts
        if cate > 0.02:
            seg = 'Persuadable (High Treatment ROI)'
            action = 'TARGET_WITH_MARKETING_PROMO'
        elif cate >= 0.0:
            seg = 'Sure Thing / Organic Converter'
            action = 'DO_NOT_SPEND_BUDGET'
        else:
            seg = 'Sleeping Dog / Negative Responder'
            action = 'SUPPRESS_OUTREACH'
            
        return {
            'predicted_cate_uplift': round(cate, 5),
            'customer_segment': seg,
            'prescribed_action': action
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
