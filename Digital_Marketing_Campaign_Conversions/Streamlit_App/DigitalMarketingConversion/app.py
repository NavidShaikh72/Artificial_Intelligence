import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os

# ------------------------------------------------------------
# 1️⃣ Load dataset to auto-generate feature inputs
# ------------------------------------------------------------
@st.cache_data
def load_data():
    base_path = os.path.dirname(__file__)  # location of app.py
    data_path = os.path.join(base_path, "data", "new_data.csv")
    df = pd.read_csv(data_path)
    return df

df = load_data()

# Define target column
target_col = "Conversion"

# Exclude target from feature list
features = [col for col in df.columns if col != target_col]

st.title("🎯 Digital Marketing Conversion Prediction App")
st.markdown("""
This app predicts whether a customer will **convert** (make a purchase or take a desired action)
based on their marketing and demographic features using an **XGBoost model**.
""")

# ------------------------------------------------------------
# 2️⃣ Load trained model (.pkl)
# ------------------------------------------------------------
base_path = os.path.dirname(__file__)  # location of app.py
model_path = os.path.join(base_path,"models", "xgboost_model.pkl")

try:
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    st.success("✅ Model loaded successfully from models/xgboost_model.pkl")
except FileNotFoundError:
    st.error("❌ Could not find model file in 'models/xgboost_model.pkl'. Please check the path.")
    st.stop()

# ------------------------------------------------------------
# 3️⃣ Dynamic input form for all features
# ------------------------------------------------------------
st.subheader("🧮 Input Feature Values")

input_data = {}
for col in features:
    # Detect numeric or categorical column
    if np.issubdtype(df[col].dtype, np.number):
        # numeric input with min/max/default for better UX
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        default_val = float(df[col].mean())
        input_data[col] = st.number_input(
            f"{col}",
            min_value=min_val,
            max_value=max_val,
            value=default_val,
            step=(max_val - min_val) / 100 if max_val != min_val else 1.0
        )
    else:
        # categorical dropdown
        unique_vals = df[col].dropna().unique().tolist()
        input_data[col] = st.selectbox(f"{col}", options=unique_vals)

# ------------------------------------------------------------
# 4️⃣ Predict button
# ------------------------------------------------------------

if st.button("🔍 Predict Conversion"):
    input_df = pd.DataFrame([input_data])

    # -------------------------------
    # Handle categorical columns
    # -------------------------------
    # Identify categorical columns
    cat_cols = input_df.select_dtypes(include=['object']).columns.tolist()

    # Encode categorical columns using the same mapping as training (Label Encoding fallback)
    for col in cat_cols:
        # Fit temporary label encoding from training data unique values
        mapping = {v: i for i, v in enumerate(df[col].dropna().unique())}
        input_df[col] = input_df[col].map(mapping)

    # Fill any missing values
    input_df = input_df.fillna(0)

    # -------------------------------
    # Predict using model
    # -------------------------------
    try:
        prediction = model.predict(input_df)
    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")
        st.stop()

    # -------------------------------
    # Display result
    # -------------------------------
    if len(np.unique(df[target_col])) <= 2:
        pred_label = "✅ Converted" if prediction[0] == 1 else "❌ Not Converted"
        st.subheader("📊 Prediction Result:")
        st.success(pred_label)

        # Optional probability display
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_df)[0][1]
            st.write(f"**Conversion Probability:** {proba*100:.2f}%")
    else:
        st.subheader("📊 Predicted Value:")
        st.info(prediction[0])


# ------------------------------------------------------------
# 5️⃣ Optional: Show dataset preview
# ------------------------------------------------------------
with st.expander("📂 View Sample of Dataset"):
    st.dataframe(df.head())
