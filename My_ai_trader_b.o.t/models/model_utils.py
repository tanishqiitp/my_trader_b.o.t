# models/model_utils.py
import joblib, os
import numpy as np
from sklearn.preprocessing import StandardScaler

MODEL_PATH = os.path.join(os.path.dirname(__file__), "news_trader_lgb.pkl")

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not trained. Run models/trainer.py")
    model = joblib.load(MODEL_PATH)
    return model

def predict_single(model, feat_dict):
    # feat_dict keys must match training features
    features = ['close_ret_1d','close_ret_3d','sma_diff','news_score','news_pred',
                'war','sanction','opec','contract','earnings','volume']
    X = np.array([feat_dict.get(f,0.0) for f in features]).reshape(1,-1)
    prob = model.predict_proba(X)[0,1]
    pred = int(prob >= 0.5)
    return {"prob_up": float(prob), "pred": pred}
