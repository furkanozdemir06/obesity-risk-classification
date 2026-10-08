import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Obesity Risk Prediction App", layout="wide")

st.title("⚖️ Obesity Risk Prediction App")
st.write("Enter your health and lifestyle metrics below to predict your obesity risk level using Machine Learning.")

# 1. LOAD DATA & TRAIN MODEL (Cached)
@st.cache_resource
def train_model():
    df = pd.read_csv('train.csv')
    
    # Drop ID column if exists
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    
    # Encode categorical variables
    label_encoders = {}
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        
    X = df.drop(columns=['NObeyesdad'])
    y = df['NObeyesdad']
    
    # Train test split and model training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    acc = model.score(X_test, y_test)
    return model, label_encoders, acc, X.columns

model, label_encoders, accuracy, feature_names = train_model()

st.sidebar.success(f"Model Trained Successfully! Accuracy: {accuracy*100:.2f}%")

# 2. USER INPUT FORM
st.header("📋 Input Your Information")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=10, max_value=100, value=25)
    height = st.number_input("Height (Meters)", min_value=1.0, max_value=2.5, value=1.70, step=0.01)
    weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.5)

with col2:
    family_history = st.selectbox("Family History with Overweight?", ["yes", "no"])
    favc = st.selectbox("Frequent High Caloric Food Intake (FAVC)?", ["yes", "no"])
    fcvc = st.slider("Vegetable Consumption Frequency (FCVC)", 1, 3, 2)
    ncp = st.slider("Number of Main Meals (NCP)", 1, 4, 3)

with col3:
    caec = st.selectbox("Consumption of Food Between Meals (CAEC)", ["no", "Sometimes", "Frequently", "Always"])
    smoke = st.selectbox("Smoker?", ["yes", "no"])
    ch2o = st.slider("Daily Water Intake (Liters - CH2O)", 1, 3, 2)
    scc = st.selectbox("Calories Consumption Monitoring (SCC)?", ["yes", "no"])

col4, col5 = st.columns(2)

with col4:
    faf = st.slider("Physical Activity Frequency (FAF)", 0, 3, 1)
    tue = st.slider("Time Using Technology Devices (TUE)", 0, 2, 1)

with col5:
    calc = st.selectbox("Alcohol Consumption (CALC)", ["no", "Sometimes", "Frequently", "Always"])
    mtrans = st.selectbox("Transportation Used (MTRANS)", ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"])

# 3. PREPROCESS INPUT & PREDICT
input_dict = {
    'Gender': gender,
    'Age': age,
    'Height': height,
    'Weight': weight,
    'family_history_with_overweight': family_history,
    'FAVC': favc,
    'FCVC': fcvc,
    'NCP': ncp,
    'CAEC': caec,
    'SMOKE': smoke,
    'CH2O': ch2o,
    'SCC': scc,
    'FAF': faf,
    'TUE': tue,
    'CALC': calc,
    'MTRANS': mtrans
}

input_df = pd.DataFrame([input_dict])

# Encode input categories
for col in input_df.select_dtypes(include=['object']).columns:
    if col in label_encoders:
        input_df[col] = label_encoders[col].transform(input_df[col])

st.markdown("---")

if st.button("🚀 Predict Obesity Risk", type="primary"):
    prediction = model.predict(input_df)
    target_encoder = label_encoders['NObeyesdad']
    result_text = target_encoder.inverse_transform(prediction)[0]
    
    st.subheader("🎯 Prediction Result:")
    
    if "Normal" in result_text:
        st.success(f"Predicted Category: **{result_text}**")
    elif "Insufficient" in result_text:
        st.warning(f"Predicted Category: **{result_text}**")
    elif "Overweight" in result_text:
        st.warning(f"Predicted Category: **{result_text}**")
    else:
        st.error(f"Predicted Category: **{result_text}**")