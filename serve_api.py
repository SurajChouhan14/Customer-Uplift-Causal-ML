import os, sys, numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

SYS_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SYS_PATH)

from src.data_loader import UpliftDataLoader
from src.meta_learners import CausalMetaLearners

app = FastAPI(title='Causal Uplift Targeting API', version='1.0.0')

loader = UpliftDataLoader()
df = loader.load_data()
feature_cols = [c for c in df.columns if c not in ['treatment', 'conversion']]
X = df[feature_cols].values
T = df['treatment'].values
y = df['conversion'].values

engine = CausalMetaLearners()
engine.fit_x_learner(X, T, y)

class CustomerProfile(BaseModel):
    recency: float = Field(default=12.0)
    frequency: float = Field(default=4.0)
    monetary: float = Field(default=150.0)
    engagement_score: float = Field(default=0.65)

@app.get('/health')
def health_check():
    return {'status': 'HEALTHY', 'model': 'Causal X-Learner with Gradient Boosting', 'target': 'ITE CATE'}

@app.post('/predict_uplift')
def predict_uplift(customer: CustomerProfile):
    try:
        xv = np.array([[customer.recency, customer.frequency, customer.monetary, customer.engagement_score]])
        cate = float(engine.predict_x_learner(xv)[0])
        if cate > 0.05:
            seg, act = 'Persuadable', 'TARGET_WITH_CAMPAIGN'
        elif cate >= 0.0:
            seg, act = 'Sure Thing', 'DO_NOT_SPEND'
        else:
            seg, act = 'Sleeping Dog', 'DO_NOT_DISTURB'
        return {'predicted_uplift': round(cate, 4), 'customer_segment': seg, 'prescribed_action': act}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
