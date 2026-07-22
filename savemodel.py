"""
Train and Save Model — VS Code / Local Version
AutoML & Hyperparameter Optimization Project

This is a STANDALONE script (no Colab needed). It loads the raw CSV,
runs the full preprocessing pipeline, trains the final Random Forest
model using your best hyperparameters (found earlier via Optuna in Colab),
and saves it for use by app.py.

BEFORE RUNNING:
1. Place 'bengaluru_house_prices.csv' in the same folder as this script
2. Install dependencies:
   pip install pandas numpy scikit-learn
3. Update BEST_PARAMS below with the actual values Optuna gave you in Colab
   (printed as "Optuna Best Params : {...}" in your notebook output)

RUN WITH:
   python save_model.py
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# =====================================================================
# >>> UPDATE THIS with your actual Optuna best_params from Colab <<<
# Example — replace with your real printed values:
# Optuna Best Params : {'n_estimators': 250, 'max_depth': 18,
#                        'min_samples_split': 4, 'min_samples_leaf': 2}
# =====================================================================
BEST_PARAMS = {
    'n_estimators': 200,
    'max_depth': 15,
    'min_samples_split': 4,
    'min_samples_leaf': 2
}

# =====================================================================
# STEP 1: LOAD DATA
# =====================================================================
df = pd.read_csv("bengaluru_house_prices.csv")
print("Initial shape:", df.shape)

# =====================================================================
# STEP 2: PREPROCESSING (same pipeline used in Colab)
# =====================================================================

# 2a. Drop unnecessary column
df = df.drop(['society'], axis=1, errors='ignore')

# 2b. Handle missing values
df['location'] = df['location'].fillna('Sarjapur Road')
df['bath'] = df['bath'].fillna(df['bath'].median())
df['balcony'] = df['balcony'].fillna(df['balcony'].median())
df = df.dropna(subset=['size', 'total_sqft'])

# 2c. Extract BHK from 'size'
df['bhk'] = df['size'].apply(lambda x: int(str(x).split(' ')[0]))
df = df.drop('size', axis=1)

# 2d. Clean 'total_sqft'
def convert_sqft(x):
    tokens = str(x).split('-')
    if len(tokens) == 2:
        try:
            return (float(tokens[0]) + float(tokens[1])) / 2
        except ValueError:
            return None
    try:
        return float(x)
    except ValueError:
        return None

df['total_sqft'] = df['total_sqft'].apply(convert_sqft)
df = df.dropna(subset=['total_sqft'])

# 2e. Create price_per_sqft (temporary, for outlier removal)
df['price_per_sqft'] = (df['price'] * 100000) / df['total_sqft']

# 2f. Reduce location cardinality (threshold raised to 50 to avoid feature explosion)
df['location'] = df['location'].apply(lambda x: x.strip())
location_counts = df['location'].value_counts()
rare_locations = location_counts[location_counts <= 50]
df['location'] = df['location'].apply(lambda x: 'other' if x in rare_locations else x)
print("Unique locations after grouping:", df['location'].nunique())

# 2g. Remove outliers
df = df[~(df['total_sqft'] / df['bhk'] < 300)]

def remove_pps_outliers(data):
    out = pd.DataFrame()
    for key, subdf in data.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced = subdf[(subdf.price_per_sqft > (m - st)) & (subdf.price_per_sqft <= (m + st))]
        out = pd.concat([out, reduced], ignore_index=True)
    return out

df = remove_pps_outliers(df)
df = df[df.bath < df.bhk + 2]

# 2h. Drop helper column
df = df.drop('price_per_sqft', axis=1)

# 2i. One-hot encode categorical columns
df = pd.get_dummies(df, columns=['location', 'area_type', 'availability'], drop_first=True)
print("Final shape after preprocessing:", df.shape)

# =====================================================================
# STEP 3: DEFINE FEATURES / TARGET AND SPLIT
# =====================================================================
X = df.drop('price', axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("X_train:", X_train.shape, " X_test:", X_test.shape)

# =====================================================================
# STEP 4: TRAIN FINAL MODEL USING BEST HYPERPARAMETERS
# =====================================================================
final_model = RandomForestRegressor(**BEST_PARAMS, random_state=42, n_jobs=-1)
final_model.fit(X_train, y_train)

from sklearn.metrics import r2_score, mean_squared_error
y_pred = final_model.predict(X_test)
print("Final Model R2  :", r2_score(y_test, y_pred))
print("Final Model RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# =====================================================================
# STEP 5: SAVE MODEL AND COLUMN REFERENCE
# =====================================================================
with open("best_model.pkl", "wb") as f:
    pickle.dump(final_model, f)

model_columns = X_train.columns.tolist()
with open("model_columns.pkl", "wb") as f:
    pickle.dump(model_columns, f)

print("\nSaved best_model.pkl and model_columns.pkl in the current folder.")
print("You can now run: streamlit run app.py")
