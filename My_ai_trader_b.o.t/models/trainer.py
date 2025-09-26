# models/trainer.py
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import lightgbm as lgb
import pandas as pd
from user_data.data.feature_engineering import build_feature_dataset

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "news_trader_lgb.pkl")

def train_model(ticker_map):
    print("Fetching features...")
    df = build_feature_dataset(ticker_map)
    if df.empty:
        raise RuntimeError("Empty dataset")
    # choose features
    features = ['close_ret_1d','close_ret_3d','sma_diff','news_score','news_pred',
                'war','sanction','opec','contract','earnings','volume']
    X = df[features].fillna(0)
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)
    lgb_clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05)
    lgb_clf.fit(X_train, y_train)
    y_pred = lgb_clf.predict(X_test)
    y_prob = lgb_clf.predict_proba(X_test)[:,1]
    print(classification_report(y_test, y_pred))
    print("AUC:", roc_auc_score(y_test, y_prob))
    joblib.dump(lgb_clf, MODEL_PATH)
    print("Saved model to", MODEL_PATH)
    return MODEL_PATH

if __name__ == "__main__":
    # small example map (update with accurate tickers)
    ticker_map = {
        "BTC":"BTC-USD",
        "GOLD":"GC=F",
        "CRUDE":"CL=F",
        "RELIANCE":"RELIANCE.NS",
    }
    train_model(ticker_map)
