import os 
from groq import Groq
import pandas as pd
import pickle
import re

#load and save model
with open("model.pkl","rb") as f:
    model=pickle.load(f)
with open("vectorizer.pkl","rb") as f:
    vectorizer=pickle.load(f)

#setting up anthropic client
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    raise ValueError(
        "GROQ_API_KEY not found. "
        "Run: set GROQ_API_KEY=your_key_here (Windows) "
        "or export GROQ_API_KEY=your_key_here (Mac/Linux)"
    )

client = Groq(api_key=api_key)

#classification fn


def clean_text(text):
    text=text.lower()
    text=re.sub(r'\d+','',text) #remove numbers
    text=re.sub(r'[^\w\s]',' ',text)#removes punctuation
    text = re.sub(r'_+', ' ', text)        
    text=re.sub(r'\s+',' ',text).strip()#removes multiple spaces
    return text

def classify_report(report_text):
    cleaned=clean_text(report_text)
    features=vectorizer.transform([cleaned])
    prediciton =model.predict(features)[0]
    confidence=model.predict_proba(features).max()
    return prediciton, confidence


#summary generation fn
def generate_patient_summary(report_text, specialty):
    prompt=f"""You are a compassionate medical assistant helping a patient 
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
    message=client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=1024,
        messages=[
            {"role":"user","content":prompt}
        ]

    )

    return message.choices[0].message.content

#combined pipeline
def analyze_report(report_text):
    print("=" * 60)
    print("ANALYZING REPORT...")
    print("=" * 60)
    
    #  Classify
    specialty, confidence = classify_report(report_text)
    print(f"Predicted Specialty: {specialty}")
    print(f"Model Confidence:    {confidence:.1%}")
    
    #  Generate explanation
    print("\nGenerating patient-friendly summary...")
    summary = generate_patient_summary(report_text, specialty)
    
    print("\n" + "=" * 60)
    print("PATIENT-FRIENDLY EXPLANATION")
    print("=" * 60)
    print(summary)

    return specialty,confidence,summary

#test on real report

df_original = pd.read_csv("mtsamples.csv").dropna(subset=['transcription'])

sample_report = df_original['transcription'].iloc[42]
print("ORIGINAL REPORT PREVIEW:")
print(sample_report[:300])
print("...\n")

analyze_report(sample_report)