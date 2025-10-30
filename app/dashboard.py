# app/streamlit_app.py
import streamlit as st
import joblib
import numpy as np

# -----------------------------------------------------
# 🎨 Page Configuration
# -----------------------------------------------------
st.set_page_config(
    page_title="Fake Internship Detector",
    page_icon="🕵️‍♂️",
    layout="wide",
)

# -----------------------------------------------------
# 🎯 Load Model & Vectorizer
# -----------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("model/random_forest_model.joblib")
    vectorizer = joblib.load("model/tfidf_vectorizer.joblib")
    return model, vectorizer

model, vectorizer = load_model()

# -----------------------------------------------------
# 🧠 Header Section
# -----------------------------------------------------
st.markdown(
    """
    <style>
    .main-title {
        font-size: 38px;
        font-weight: 700;
        text-align: center;
        color: #2E86C1;
    }
    .sub-title {
        text-align: center;
        color: #616A6B;
        font-size: 18px;
        margin-bottom: 30px;
    }
    .result-box {
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        font-size: 22px;
        font-weight: 600;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<div class='main-title'>Fake Internship Detection App</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Analyze job/internship postings and detect if they are fraudulent or legitimate.</div>", unsafe_allow_html=True)

# -----------------------------------------------------
# 📥 Input Section
# -----------------------------------------------------
st.write("### ✍️ Enter Job Posting Details")

col1, col2 = st.columns(2)

with col1:
    title = st.text_input("Job Title")
    company_profile = st.text_area("Company Profile", height=100)

with col2:
    description = st.text_area("Job Description", height=150)
    requirements = st.text_area("Requirements", height=100)
    benefits = st.text_area("Benefits (optional)", height=80)

telecommuting = st.selectbox("Is this a remote job?", ["No", "Yes"])
has_company_logo = st.selectbox("Does it have a company logo?", ["Yes", "No"])
has_questions = st.selectbox("Does it include screening questions?", ["Yes", "No"])

# -----------------------------------------------------
# 🧩 Prepare Input Function
# -----------------------------------------------------
def prepare_input():
    combined_text = " ".join([
        str(title),
        str(company_profile),
        str(description),
        str(requirements),
        str(benefits)
    ])

    # TF-IDF features
    text_features = vectorizer.transform([combined_text])

    # Known binary features
    extra_features = np.array([
        1 if telecommuting == "Yes" else 0,
        1 if has_company_logo == "Yes" else 0,
        1 if has_questions == "Yes" else 0,
        0,  # Placeholder for 4th missing feature
        0,  # Placeholder for 5th missing feature
        0   # Placeholder for 6th missing feature
    ]).reshape(1, -1)

    from scipy.sparse import hstack
    full_features = hstack([text_features, extra_features])

    return full_features


# -----------------------------------------------------
# 🚀 Prediction Button
# -----------------------------------------------------
if st.button("🔍 Predict"):
    input_features = prepare_input()
    prediction = model.predict(input_features)[0]
    proba = model.predict_proba(input_features)[0][1] * 100

    if prediction == 1:
        st.markdown(
            f"<div class='result-box' style='background-color:#27AE60;'>✅ Legitimate Posting<br>Confidence: {proba:.2f}%</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div class='result-box' style='background-color:#E74C3C;'>🚨 Fake/Scam Posting<br>Confidence: {100 - proba:.2f}%</div>",
            unsafe_allow_html=True
        )

# -----------------------------------------------------
# 📊 Footer
# -----------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align:center; color:#7D7D7D;'>
    Built with ❤️ by <b>Sameer Chauhan</b> | ML Internship Detection Project
    </div>
    """,
    unsafe_allow_html=True
)
