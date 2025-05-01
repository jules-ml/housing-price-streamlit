import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# --- SETUP ---
np.random.seed(123)

# --- LOAD DATA (update these paths if needed) ---
train_raw = pd.read_csv("C:/Users/jules/OneDrive/Documents/house-prices/train.csv")
test_raw = pd.read_csv("C:/Users/jules/OneDrive/Documents/house-prices/test.csv")

# Add log-transformed SalePrice
train_raw['log_SalePrice'] = np.log(train_raw['SalePrice'])

# --- PREPROCESSING FUNCTION ---
def preprocess_data(data, train_medians=None):
    data = data.copy()
    
    # Handle categoricals
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in categorical_cols:
        data[col] = data[col].astype('category')
        if 'Unknown' not in data[col].cat.categories:
            data[col] = data[col].cat.add_categories('Unknown')
        data[col] = data[col].fillna('Unknown')
    
    # Handle numerics
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if train_medians is None:
        train_medians = {col: data[col].median() for col in numeric_cols}
    for col in numeric_cols:
        data[col] = data[col].fillna(train_medians.get(col, 0))
    
    return data, train_medians

# --- RUN PREPROCESSING ---
train_cleaned, train_medians = preprocess_data(train_raw)
train_cleaned.drop(columns=['SalePrice'], inplace=True)

# --- REMOVE HIGHLY CORRELATED NUMERIC FEATURES ---
numeric_cols = train_cleaned.select_dtypes(include=[np.number]).columns.tolist()
numeric_predictors = [col for col in numeric_cols if col != 'log_SalePrice']
corr_matrix = train_cleaned[numeric_predictors].corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > 0.9)]
print("Removing highly correlated columns:", to_drop)
train_decorrelated = train_cleaned.drop(columns=to_drop)

# --- ONE-HOT ENCODING + TRAIN/TEST SPLIT ---
y = train_decorrelated['log_SalePrice']
X = train_decorrelated.drop(columns=['log_SalePrice'])
X_encoded = pd.get_dummies(X, drop_first=True)

print("Number of columns after dummy encoding:", X_encoded.shape[1])

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.3, random_state=123
)

print("✅ Train shape:", X_train.shape)
print("✅ Test shape:", X_test.shape)

# --- RE-ENCODE WITHOUT DROPPING DUMMIES ---
X = train_decorrelated.drop(columns=['log_SalePrice'])
y = train_decorrelated['log_SalePrice']

X_encoded = pd.get_dummies(X, drop_first=False)
print("✅ After encoding:", X_encoded.shape)

# Drop columns that are constant (equivalent to nearZeroVar)
X_encoded = X_encoded.loc[:, X_encoded.nunique() > 1]
print("✅ After dropping constant cols:", X_encoded.shape)

# Train-test split again with this version
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.3, random_state=123
)

# --- LASSO WITH STANDARDIZATION ---
print("Running LASSO with feature scaling...")

# Use pipeline to scale and fit LASSO
pipeline = make_pipeline(StandardScaler(), LassoCV(cv=5, random_state=123))
pipeline.fit(X_train, y_train)

# Extract fitted LassoCV model
lasso_cv = pipeline.named_steps['lassocv']

# Get selected features
coef = lasso_cv.coef_
selected_features = X_train.columns[coef != 0]
print(f"✅ LASSO selected {len(selected_features)} features.")

# Reduce train/test to LASSO-selected features
X_train_lasso = X_train[selected_features]
X_test_lasso = X_test[selected_features]

print("🔢 Final shape for modeling:")
print("X_train_lasso:", X_train_lasso.shape)
print("X_test_lasso:", X_test_lasso.shape)

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Train GBM on LASSO-selected features
gbm = GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=3, random_state=123)
gbm.fit(X_train_lasso, y_train)

# Predict and evaluate
y_pred_gbm = gbm.predict(X_test_lasso)
rmse_gbm = np.sqrt(mean_squared_error(y_test, y_pred_gbm))
print("✅ Gradient Boosting RMSE:", rmse_gbm)

joblib.dump(gbm, "gbm_model.pkl")
joblib.dump(selected_features.tolist(), "lasso_selected_features.pkl")
joblib.dump(X_encoded.columns.tolist(), "all_training_columns.pkl")
