import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("KNN_heart.pkl")
    scaler = joblib.load("scaler.pkl")
    columns = joblib.load("columns.pkl")
    return model, scaler, columns

model, scaler, columns = load_artifacts()

# ---------------- CUSTOM UI ----------------
st.markdown("""
<style>
/* Adaptive card based on Streamlit theme */
div[data-testid="stForm"] {
    background-color: var(--secondary-background-color);
    padding: 25px;
    border-radius: 16px;
    border: 1px solid var(--border-color);
}

/* Input fields */
input, select, textarea {
    background-color: var(--background-color) !important;
    color: var(--text-color) !important;
}

/* Labels */
label {
    color: var(--text-color) !important;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)


# ---------------- HEADER ----------------
st.markdown(
    """
    <h1 style="text-align:center;color:crimson;">❤️ Heart Disease Prediction System</h1>
    <p style="text-align:center;font-size:17px;">
    Machine Learning based clinical risk assessment
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# ---------------- SIDEBAR ----------------
st.sidebar.title("🩺 Navigation")
menu = st.sidebar.radio("Go to", ["Home", "Predict", "About"])

# ---------------- HOME ----------------
if menu == "Home":
    st.subheader("🏥 Project Overview")
    st.write("""
    This application predicts the **risk of heart disease** using
    a machine learning model trained on clinical health data.

    **Key Inputs**
    - Age, Blood Pressure, Cholesterol
    - ECG results
    - Chest pain & exercise indicators

    ⚠️ *Educational use only. Not a medical diagnosis.*
    """)

# ---------------- PREDICT ----------------
elif menu == "Predict":
    st.subheader("🧪 Patient Clinical Information")

    with st.form("heart_form"):
        col1, col2 = st.columns(2)

        # -------- LEFT COLUMN --------
        with col1:
            age = st.number_input("Age", 20, 100, 45)
            resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
            cholesterol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 200)

            fasting_bs = st.selectbox(
                "Fasting Blood Sugar > 120 mg/dl",
                ["No", "Yes"]
            )

            max_hr = st.number_input("Maximum Heart Rate Achieved", 60, 220, 150)
            oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 6.0, 1.0)

        # -------- RIGHT COLUMN --------
        with col2:
            sex = st.selectbox("Sex", ["Female", "Male"])

            chest_pain = st.selectbox(
                "Chest Pain Type",
                ["ASY", "ATA", "NAP", "TA"],
                help="ASY: Asymptomatic, ATA: Atypical Angina"
            )

            resting_ecg = st.selectbox(
                "Resting ECG",
                ["Normal", "ST", "LVH"]
            )

            exercise_angina = st.selectbox(
                "Exercise Induced Angina",
                ["No", "Yes"]
            )

            st_slope = st.selectbox(
                "ST Slope",
                ["Down", "Flat", "Up"]
            )

        submit = st.form_submit_button("🔍 Predict Risk")

    # ---------------- PREDICTION ----------------
    if submit:
        # -------- BASE FEATURES --------
        input_data = {
            "Age": age,
            "RestingBP": resting_bp,
            "Cholesterol": cholesterol,
            "FastingBS": 1 if fasting_bs == "Yes" else 0,
            "MaxHR": max_hr,
            "Oldpeak": oldpeak,
        }

        # -------- ONE-HOT ENCODING --------
        input_data["Sex_M"] = 1 if sex == "Male" else 0

        input_data["ChestPainType_ATA"] = 1 if chest_pain == "ATA" else 0
        input_data["ChestPainType_NAP"] = 1 if chest_pain == "NAP" else 0
        input_data["ChestPainType_TA"] = 1 if chest_pain == "TA" else 0

        input_data["RestingECG_Normal"] = 1 if resting_ecg == "Normal" else 0
        input_data["RestingECG_ST"] = 1 if resting_ecg == "ST" else 0

        input_data["ExerciseAngina_Y"] = 1 if exercise_angina == "Yes" else 0

        input_data["ST_Slope_Flat"] = 1 if st_slope == "Flat" else 0
        input_data["ST_Slope_Up"] = 1 if st_slope == "Up" else 0

        # -------- DATAFRAME ALIGNMENT --------
        df = pd.DataFrame([input_data])
        df = df.reindex(columns=columns, fill_value=0)

        # -------- SCALE & PREDICT --------
        scaled = scaler.transform(df)
        prediction = model.predict(scaled)[0]
        probability = model.predict_proba(scaled)[0][prediction]

        st.divider()

        if prediction == 1:
            st.error(
                f"⚠️ **High Risk of Heart Disease**\n\n"
                f"Confidence: **{probability:.2%}**"
            )
        else:
            st.success(
                f"✅ **Low Risk of Heart Disease**\n\n"
                f"Confidence: **{probability:.2%}**"
            )


    st.divider()
    st.markdown("### 📘 Medical Terms & Abbreviations Used")

    with st.expander("Click to view full forms & explanations"):
        st.markdown("""
        **Age**  
        - Patient age in years  

        **RestingBP (Resting Blood Pressure)**  
        - Blood pressure measured at rest (mm Hg)  

        **Cholesterol (Serum Cholesterol)**  
        - Cholesterol level in blood (mg/dl)  

        **FastingBS (Fasting Blood Sugar)**  
        - Blood sugar after fasting  
        - *Yes = >120 mg/dl, No = ≤120 mg/dl*  

        **MaxHR (Maximum Heart Rate Achieved)**  
        - Highest heart rate during exercise  

        **Oldpeak (ST Depression)**  
        - ST segment depression induced by exercise  

        **Sex**  
        - Biological sex of patient  

        **Chest Pain Type**
        - **ASY** – Asymptomatic  
        - **ATA** – Atypical Angina  
        - **NAP** – Non-Anginal Pain  
        - **TA** – Typical Angina  

        **RestingECG (Electrocardiographic Results)**
        - **Normal** – Normal ECG  
        - **ST** – ST-T wave abnormality  
        - **LVH** – Left Ventricular Hypertrophy  

        **Exercise Angina**
        - Chest pain induced by exercise  

        **ST Slope**
        - **Up** – Upsloping  
        - **Flat** – Flat  
        - **Down** – Downsloping  

        ⚠️ *For educational purposes only.*
        """)


# ---------------- ABOUT ----------------
else:
    st.subheader("📌 About This Project")
    st.write("""
    **Domain:** Healthcare + Machine Learning  
    **Algorithms:** K-Nearest Neighbors (KNN)  
    **Tools:** Python, Scikit-learn, Streamlit  

    👨‍🎓 Designed for academic & portfolio use.
    """)

    st.warning("This application does not replace professional medical advice.")
