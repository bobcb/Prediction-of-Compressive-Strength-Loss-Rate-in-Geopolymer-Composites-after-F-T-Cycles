#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor
from mapie.regression import MapieRegressor
from joblib import dump
import os

# Define paths relative to the script location
data_path = 'Dataset.csv'  # Assume data file is in the same directory or provide full path if needed
output_dir = 'models'  # Directory to save models, relative to script

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# 1. Load data
df = pd.read_csv(data_path)

# 2. Separate features and target variable
X = df.drop(['Compressive strength loss rate (%)'], axis=1)
y = df['Compressive strength loss rate (%)']

# 3. Split into train and test sets
X_train_unss, X_test_unss, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Standardization
ss = StandardScaler()
X_train = ss.fit_transform(X_train_unss)
X_test = ss.transform(X_test_unss)

# 5. Train model: CatBoostRegressor
CB = CatBoostRegressor(
    iterations=200,
    learning_rate=0.1,
    depth=7,
    l2_leaf_reg=7.0,
    random_seed=42,
)
CB.fit(X_train, y_train)

# 6. Train Mapie interval estimator (based on the same trained model)
mapie_model = MapieRegressor(estimator=CB, cv=5, method="plus")
mapie_model.fit(X_train, y_train)

# 7. Save models and scaler
dump(CB, os.path.join(output_dir, 'CB_model.joblib'))
dump(mapie_model, os.path.join(output_dir, 'Mapie_Interval.joblib'))
dump(ss, os.path.join(output_dir, 'StandardScaler.joblib'))

print(f"Models saved to {output_dir}")


# In[ ]:




