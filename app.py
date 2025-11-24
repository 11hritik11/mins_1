import pickle
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

# --------------------------
# LOAD MODEL + SCALER
# --------------------------

@st.cache_resource
def load_model_and_scaler():
    with open("menstrual_cycle_length_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


# --------------------------
# LOAD DATA & DISCOVER COLUMNS
# --------------------------

@st.cache_resource
def load_data_info():
    df = pd.read_csv("FedCycleData071012 (2).csv")

    # Replace blank spaces with NaN
    df = df.replace(" ", np.nan)

    # Drop ≥50% missing
    null = df.isnull().sum()
    null_per = (null / df.shape[0]) * 100
    cols_to_drop = null_per[null_per >= 50].index.tolist()
    df = df.drop(columns=cols_to_drop, errors="ignore")

    # Drop ClientID
    df = df.drop(columns=["ClientID"], errors="ignore")

    # Target + Features
    y_col = "LengthofCycle"
    feature_cols = [c for c in df.columns if c != y_col]

    # Detect categorical columns BEFORE numeric conversion
    categorical_cols = df[feature_cols].select_dtypes(include="object").columns.tolist()
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    # Build LabelEncoders for each categorical column based on ORIGINAL strings
    encoder_dict = {}
    category_options = {}

    for col in categorical_cols:
        le = LabelEncoder()
        non_null_vals = df[col].dropna().astype(str)
        le.fit(non_null_vals)

        encoder_dict[col] = le
        category_options[col] = list(le.classes_)

    return feature_cols, categorical_cols, numeric_cols, encoder_dict, category_options


# --------------------------
# UI THEME
# --------------------------

def set_pink_theme():
    st.markdown("""
        <style>
        .stApp {
            background-color: #ffe4f0;
        }
        h1, h2, h3, h4, p, label, span {
            color: #2b2b2b !important;
        }
        .stButton>button {
            background-color: #ffb6c1;
            color: #2b2b2b;
            border-radius: 10px;
            padding: 8px 18px;
            font-weight: 600;
        }
        .stButton>button:hover {
            background-color: #ff9bb0;
        }
        </style>
    """, unsafe_allow_html=True)


# --------------------------
# STREAMLIT APP
# --------------------------

def main():
    set_pink_theme()

    st.title("🌸 Menstrual Cycle Length Predictor")
    st.write("Enter the model features below to predict cycle length.")

    # Load
    model, scaler = load_model_and_scaler()
    feature_cols, categorical_cols, numeric_cols, encoder_dict, category_options = load_data_info()

    st.subheader("Enter Inputs:")
    input_data = {}

    with st.form("form"):
        col1, col2 = st.columns(2)
        side = 0

        for col in feature_cols:
            container = col1 if side % 2 == 0 else col2

            with container:
                if col in categorical_cols:
                    opt = category_options[col]
                    value = st.selectbox(f"{col}", opt)
                    input_data[col] = value
                else:
                    value = st.number_input(f"{col}", value=0.0)
                    input_data[col] = value

            side += 1

        submit = st.form_submit_button("Predict ✨")

    if submit:
        try:
            # Convert to DF
            df_input = pd.DataFrame([input_data])

            # Encode categorical
            for col in categorical_cols:
                le = encoder_dict[col]
                df_input[col] = le.transform(df_input[col])

            # Order columns
            df_input = df_input[feature_cols]

            # Scale
            x_scaled = scaler.transform(df_input.values)

            # Predict
            pred = model.predict(x_scaled)[0]

            st.success(f"Predicted Cycle Length: **{pred:.2f} days**")

        except Exception as e:
            st.error(f"Error during prediction: {e}")


if __name__ == "__main__":
    main()
