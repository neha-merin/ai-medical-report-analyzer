import streamlit as st
import pickle
import re
import os
from groq import Groq

#set up browser tab
st.set_page_config(
    page_title="AI Medical Report Analyzer",
    page_icon="🏥",
    layout="wide"
)

#load model and vectorizer for once
@st.cache_resource
def load_model():
    with open("model.pkl","rb") as f:
        model=pickle.load(f)
    with open("vectorizer.pkl","rb") as f:
        vectorizer=pickle.load(f)
    return model, vectorizer
model, vectorizer =load_model()

#setup groq client
api_key=os.environ.get("GROQ_API_KEY")
if not api_key:
    st.error("GROQ_API_KEY not found. Set it in your terminal and restart.")
    st.stop()

client = Groq(api_key=api_key)

#clean and predict fn same as generate summary and save_model
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'_+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def classify_report(report_text):
    cleaned = clean_text(report_text)
    features = vectorizer.transform([cleaned])
    prediction = model.predict(features)[0]
    confidence = model.predict_proba(features).max()
    return prediction, confidence

def generate_summary(report_text, specialty):
    prompt = f"""You are a compassionate medical assistant helping a patient 
understand their medical report. The patient has no medical background.

Your task:
- Explain what the report says in simple, clear language
- Avoid ALL medical jargon — if you must use a medical term, explain it immediately
- Use a calm, reassuring tone
- Mention what medical specialty this falls under and what that specialty does
- Explain what the findings mean for the patient's daily life
- Suggest what questions the patient might want to ask their doctor
- Keep your response under 200 words

Medical Specialty: {specialty}

Medical Report:
{report_text[:1000]}

Write your patient-friendly explanation below:"""

    message = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.choices[0].message.content

#ui layout-what user sees
st.title("🏥 AI Medical Report Analyzer")
st.markdown(
    "Paste a medical report below to receive an **automatic specialty"
    "classification** and a **patient-friendly explanation**."
)
st.divider()
#using a two cololumn layout input on left and result on right
col1,col2=st.columns([1,1],gap="large")

with col1:
    st.subheader("📋 Medical Report Input")
    report_text=st.text_area(
        label="Paste your medical report here:",
        height=350,
        placeholder="e.g. SUBJECTIVE: This 45-year-old male presents with "
                    "chest pain radiating to the left arm..."
    )

    analyze_button=st.button(
        "🔍 Analyze Report",
        type="primary",
        use_container_width=True
    )

with col2:
    st.subheader("📊 Analysis Results")
    if analyze_button:
        if not report_text.strip():
            st.warning("Please paste a medical report before analyzing.")
        else:
            with st.spinner("Classifying report..."):
                specialty,confidence=classify_report(report_text)
            
            st.success(f"**Predicted Specialty:**{specialty}")

            if confidence>=0.5:
                st.metric("Model Confidence",f"{confidence:.1%}",delta="High")
            elif confidence>=0.3:
                st.metric("Model Confidence",f"{confidence:.1%}",delta="Medium")
            else:
                st.metric(
                    "Model Confidence",
                    f"{confidence:.1%}",
                    delta="Low - Treat with caution",
                    delta_color="inverse"
                )   
            st.divider()
             # Generate and display patient summary
            with st.spinner("Generating patient-friendly explanation..."):
                summary = generate_summary(report_text, specialty)

            st.subheader("🗣️ Patient-Friendly Explanation")
            st.markdown(summary)

            # Low confidence warning
            if confidence < 0.3:
                st.warning(
                    "⚠️ The model has low confidence in this prediction. "
                    "Please consult a qualified medical professional."
                )

    else:
        # Shown before the user clicks anything
        st.info("Your analysis results will appear here.")