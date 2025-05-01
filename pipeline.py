# pipeline.py
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

# Load trained objects (to be saved separately)
model = joblib.load("gbm_model.pkl")
selected_features = joblib.load("lasso_selected_features.pkl")
all_columns = joblib.load("all_training_columns.pkl")

# Prediction pipeline
def predict_price(df_raw):
    df = df_raw.copy()

    # Basic cleaning (match what was done in training)
    df['log_SalePrice'] = 0  # placeholder so we can drop it later if needed
    df.fillna("Unknown", inplace=True)

    # One-hot encode (no drop_first)
    df = pd.get_dummies(df, drop_first=False)

    # Add missing columns (ones present in training but not in test input)
    for col in all_columns:
        if col not in df.columns:
            df[col] = 0

    # Drop extra columns (ones present in test but not in training)
    df = df[all_columns]

    # Filter to LASSO-selected columns
    df_lasso = df[selected_features]

    # Predict log price, then revert to dollar price
    log_preds = model.predict(df_lasso)
    preds = np.exp(log_preds)

    return preds
