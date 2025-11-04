import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- Load trained model ---
@st.cache_resource
def load_model():
    with open('logistic_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

st.set_page_config(page_title="Titanic Survival Prediction", layout="wide")
st.title("🚢 Titanic Survival Prediction (Deployed Model)")
st.write("This app loads a pre-trained Logistic Regression model to predict survival based on user input.")

# --- User input form ---
st.sidebar.header("🔹 Enter Passenger Details")

pclass = st.sidebar.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
sex = st.sidebar.selectbox("Sex", ["male", "female"])
age = st.sidebar.slider("Age", 1, 80, 30)
sibsp = st.sidebar.slider("Siblings/Spouses Aboard", 0, 8, 0)
parch = st.sidebar.slider("Parents/Children Aboard", 0, 6, 0)
fare = st.sidebar.slider("Fare", 0.0, 520.0, 32.0)
embarked = st.sidebar.selectbox("Port of Embarkation", ["C", "Q", "S"])

# --- Encode input data ---
sex_encoded = 1 if sex == "male" else 0
embarked_map = {"C": 0, "Q": 1, "S": 2}
embarked_encoded = embarked_map[embarked]

user_input = pd.DataFrame({
    'Pclass': [pclass],
    'Sex': [sex_encoded],
    'Age': [age],
    'SibSp': [sibsp],
    'Parch': [parch],
    'Fare': [fare],
    'Embarked': [embarked_encoded]
})

st.write("### 🧾 User Input Summary")
st.dataframe(user_input)

# --- Predict ---
prediction = model.predict(user_input)[0]
probability = model.predict_proba(user_input)[0][1]

result = "✅ Survived" if prediction == 1 else "❌ Did Not Survive"

st.subheader("🎯 Prediction Result")
st.write(f"**Prediction:** {result}")
st.write(f"**Survival Probability:** {probability:.2f}")

st.markdown("---")
st.info("This app demonstrates professional Streamlit deployment using a pre-trained Logistic Regression model.")
