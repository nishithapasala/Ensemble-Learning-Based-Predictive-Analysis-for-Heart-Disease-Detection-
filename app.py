# # ===============================
# # Heart Disease Prediction App
# # ===============================

# import warnings
# warnings.filterwarnings('ignore')

# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# # -------------------------------
# # Page Configuration
# # -------------------------------
# st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

# st.title("❤️ Heart Disease Prediction Web App")
# st.write("This application predicts whether a person has heart disease using **Random Forest Classifier**.")

# # -------------------------------
# # Load Dataset
# # -------------------------------
# @st.cache_data
# def load_data():
#     return pd.read_csv("heart_dataset_complete_MODIFIED.csv")

# df = load_data()

# st.subheader("📊 Dataset Preview")
# st.dataframe(df.head())

# # ======================================================
# # 🔥 IMPORTANT FIX — ENCODING STARTS HERE 🔥
# # ======================================================

# df_encoded = df.copy()
# label_encoders = {}

# for col in df_encoded.columns:
#     if col != "target":
#         if df_encoded[col].dtype == "object":
#             le = LabelEncoder()
#             df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
#             label_encoders[col] = le
#         else:
#             df_encoded[col] = pd.to_numeric(df_encoded[col], errors="coerce")

# df_encoded.dropna(inplace=True)

# # ======================================================
# # 🔥 ENCODING ENDS HERE 🔥
# # ======================================================

# # -------------------------------
# # Data Preprocessing
# # -------------------------------
# X = df_encoded.drop(columns=["target"])
# y = df_encoded["target"]

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# # -------------------------------
# # Train Random Forest Model
# # -------------------------------
# model = RandomForestClassifier(
#     n_estimators=200,
#     random_state=42,
#     max_depth=10
# )
# model.fit(X_train, y_train)

# # -------------------------------
# # Model Evaluation
# # -------------------------------
# y_pred = model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)

# st.subheader("📈 Model Performance")
# st.write(f"**Accuracy:** {accuracy * 100:.2f}%")

# if st.checkbox("Show Classification Report"):
#     st.text(classification_report(y_test, y_pred))

# if st.checkbox("Show Confusion Matrix"):
#     cm = confusion_matrix(y_test, y_pred)
#     fig, ax = plt.subplots()
#     sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
#     st.pyplot(fig)

# # -------------------------------
# # Correlation Heatmap
# # -------------------------------
# if st.checkbox("Show Correlation Heatmap"):
#     corr = df_encoded.corr()
#     fig, ax = plt.subplots(figsize=(10, 6))
#     sns.heatmap(corr, cmap="viridis", ax=ax)
#     st.pyplot(fig)

# # -------------------------------
# # User Input
# # -------------------------------
# st.subheader("🧑‍⚕️ Enter Patient Details")

# user_input = {}

# for col in X.columns:
#     min_val = float(X[col].min())
#     max_val = float(X[col].max())
#     mean_val = float(X[col].mean())

#     user_input[col] = st.slider(
#         col,
#         min_value=min_val,
#         max_value=max_val,
#         value=mean_val
#     )

# input_df = pd.DataFrame([user_input])

# # -------------------------------
# # Prediction
# # -------------------------------
# if st.button("🔍 Predict Heart Disease"):
#     prediction = model.predict(input_df)[0]
#     probability = model.predict_proba(input_df)[0][1]

#     if prediction == 1:
#         st.error(f"⚠️ High Risk of Heart Disease\nProbability: {probability*100:.2f}%")
#     else:
#         st.success(f"✅ Low Risk of Heart Disease\nProbability: {probability*100:.2f}%")

# st.markdown("---")
# st.caption("Developed using Random Forest & Streamlit")


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# --- 1. DATA LOADING & PREPROCESSING ---
@st.cache_resource
def get_trained_model():
    # Load the specific dataset mentioned
    df = pd.read_csv('Dataset Heart Disease (1).csv')
    
    # Preprocessing to handle missing values and categorical 'sex'
    df = df.fillna(df.median(numeric_only=True))
    if 'sex' in df.columns and df['sex'].dtype == 'object':
        df['sex'] = df['sex'].map({'M': 1, 'F': 0}).fillna(1)
    
    # Define features and target based on the CSV columns
    X = df.drop(columns=["target"])
    y = df["target"]
    
    # Train the model
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    return rf, X.columns.tolist()

# Initialize model and get the exact column names/order
model, trained_feature_names = get_trained_model()

# --- 2. STREAMLIT UI ---
st.title("❤️ Heart Disease Prediction")

st.sidebar.header("Enter Patient Details")

def get_user_input():
    # We must create a dictionary where keys EXACTLY match 'trained_feature_names'
    # Features: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
    
    age = st.sidebar.slider("Age", 1, 100, 50)
    sex_choice = st.sidebar.selectbox("Sex", ["Male", "Female"])
    sex = 1 if sex_choice == "Male" else 0
    
    cp = st.sidebar.selectbox("Chest Pain Type (cp: 0-3)", [0, 1, 2, 3])
    trestbps = st.sidebar.number_input("Resting Blood Pressure (trestbps)", 80, 200, 120)
    chol = st.sidebar.number_input("Serum Cholestoral (chol)", 100, 600, 200)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", [0, 1])
    restecg = st.sidebar.selectbox("Resting ECG Results (restecg: 0-2)", [0, 1, 2])
    thalach = st.sidebar.slider("Max Heart Rate (thalach)", 60, 220, 150)
    exang = st.sidebar.selectbox("Exercise Induced Angina (exang)", [0, 1])
    oldpeak = st.sidebar.slider("ST Depression (oldpeak)", 0.0, 6.0, 1.0)
    slope = st.sidebar.selectbox("Slope of ST Segment (slope: 0-2)", [0, 1, 2])
    ca = st.sidebar.selectbox("Major Vessels (ca: 0-4)", [0, 1, 2, 3, 4])
    thal = st.sidebar.selectbox("Thalassemia (thal: 1-3)", [1, 2, 3])

    # Create a dictionary
    data = {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
        'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
        'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }
    
    # Convert to DataFrame and ENSURE COLUMN ORDER matches training data
    features = pd.DataFrame([data])
    features = features[trained_feature_names] # Reorder columns to match model
    return features

# Capture input
input_df = get_user_input()

st.subheader("Patient Summary")
st.write(input_df)

# --- 3. PREDICTION ---
if st.button("Predict Result"):
    # This will now work because input_df columns == trained_feature_names
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0]

    if prediction == 1:
        st.error(f"Prediction: **Heart Disease Detected** (Probability: {prediction_proba[1]:.2%})")
    else:
        st.success(f"Prediction: **No Heart Disease** (Probability: {prediction_proba[0]:.2%})")