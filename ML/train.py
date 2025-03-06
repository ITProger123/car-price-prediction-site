import joblib
import warnings
import pandas as pd
from sklearn.model_selection import cross_validate
import numpy as np
from preprocess import extract_fetures_from_engine, preprocess_transmission, extract_speed, compress_similar_gradation
from preprocess import add_is_luxuary_model, add_is_luxuary_brand, preprocess, clean_columns
import re
from sklearn.metrics import root_mean_squared_error
from lightgbm import LGBMRegressor
warnings.filterwarnings('ignore')

pipeline = joblib.load('pipeline.pkl')
df = pd.read_csv("data/train.csv")
X = df.drop(columns=['price'])
y = np.log(df['price'])
X = pipeline.transform(X)

lgb_params = {
        'n_estimators': 2000,
        'num_leaves': 426,
        'learning_rate': 0.011353178352988012,
        'subsample': 0.5772552201954328,
        'colsample_bytree': 0.9164865430101521,
        'reg_alpha': 1.48699088003429e-06,
        'reg_lambda': 0.41539458543414265,
        'min_data_in_leaf': 73,
        'feature_fraction': 0.751673655170548,
        'bagging_fraction': 0.5120415391590843,
        'bagging_freq': 2,
        'min_child_weight': 0.017236362383443497,
        'cat_smooth': 54.81317407769262}


model_lgb = LGBMRegressor(**lgb_params)
results = cross_validate(model_lgb, X, y, cv=5, 
                         return_estimator=True, scoring='r2', return_indices=True)


scores = np.zeros(5)
for i, (idx, model) in enumerate(zip(results['indices']['test'], results['estimator']), start=1):
    X_test = X[idx]
    y_true = np.exp(y[idx])
    y_pred = np.exp(model.predict(X_test))
    score = root_mean_squared_error(y_true, y_pred)
    print(f"Ford_{i}:", f"oof_score = {score}")
    scores[i-1] = score
print(f"Mean oof = {scores.mean()}")


final_model = LGBMRegressor(**lgb_params)
final_model.fit(X, y)
joblib.dump(model, 'model.pkl')