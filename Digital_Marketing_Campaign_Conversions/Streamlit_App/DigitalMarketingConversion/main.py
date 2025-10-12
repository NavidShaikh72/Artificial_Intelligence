import pickle
import pandas as pd
import xgboost as xgb


# Step 1: Load your saved model
with open("models/xgboost_model.pkl", "rb") as file:
    model = pickle.load(file)

# Step 2: Load new data for prediction
# (Replace this with your actual dataset or input values)
data = pd.read_csv("data/new_data.csv")

# Step 3: Prepare the data (same preprocessing as training)
# Ensure same columns, encoding, scaling, etc.
X_new = data.drop("target_column", axis=1, errors="ignore")

# Step 4: Make predictions
predictions = model.predict(X_new)

# Step 5: Display results
print("Predictions:", predictions)