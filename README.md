# 🏥 AI Medical Report Analyzer

A machine learning web application that classifies medical reports into specialties and generates patient-friendly explanations using a large language model.

**Live Demo:** [ai-medical-report-analyzer-nee.streamlit.app](https://ai-medical-report-analyzer-nee.streamlit.app/)

---

## What It Does

Paste any medical transcription report and the system will:

1. **Classify** the report into one of 15 medical specialties using a trained ML model
2. **Explain** the report in plain, jargon-free language that any patient can understand
3. **Flag** low-confidence predictions with a caution warning

---

## Demo

| Input | Predicted Specialty | Confidence |
|---|---|---|
| Neurology report with MRI findings | Neurology | 26.6% |
| Testicular ultrasound report | Urology | 27.6% |

---

## How It Works

```
Medical Report (raw text)
        ↓
Text Preprocessing (lowercase, remove punctuation, remove numbers)
        ↓
TF-IDF Vectorization (5000 features, bigrams)
        ↓
Logistic Regression Classifier → Predicted Specialty
        ↓
LLM Prompt (Groq / LLaMA 3.3 70B) → Patient-Friendly Explanation
        ↓
Streamlit Web Interface
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3 |
| ML / NLP | scikit-learn, TF-IDF, Logistic Regression |
| LLM API | Groq (LLaMA 3.3 70B) |
| Web App | Streamlit |
| Deployment | Streamlit Community Cloud |

---

## Dataset

**Medical Transcriptions Dataset** — Kaggle  
- 4,999 real medical transcription samples across 40 specialties
- Preprocessed to top 15 specialties, undersampled to 90 samples per class
- Final training set: 1,350 balanced samples

---

## Model Performance

| Model | Test Accuracy | F1 Score (weighted) | Train/Test Gap |
|---|---|---|---|
| Logistic Regression | 52.2% | 0.51 | 0.28 |
| Naive Bayes | 50.0% | 0.50 | 0.22 |

**Logistic Regression selected** based on superior F1 score.

> Baseline (random guessing across 15 classes) = 6.7%. Both models perform approximately 7.5× better than chance.

### Per-Class Highlights

| Best Performing | F1 | Worst Performing | F1 |
|---|---|---|---|
| Obstetrics / Gynecology | 0.80 | Surgery | 0.20 |
| ENT - Otolaryngology | 0.74 | Consult - History & Phy. | 0.24 |
| Urology | 0.74 | SOAP / Chart / Progress Notes | 0.26 |

**Key insight:** Document-format categories (Surgery, SOAP Notes, Consultations) scored lowest because they span multiple body systems and share vocabulary with nearly every other specialty. Domain-specific specialties (OB/GYN, ENT, Urology) scored highest due to distinctive medical terminology.

---

## Data Decisions

| Decision | Technique | Reasoning |
|---|---|---|
| Removed 33 rows with missing transcriptions | Null dropping | Cannot impute subjective medical text |
| Kept top 15 specialties only | Class filtering | Tail classes had fewer than 10 samples — insufficient to learn from |
| Capped each class at 90 samples | Random undersampling | Prevented Surgery (22% of data) from dominating training |

---

## Known Limitations

- **Small training set** — 90 samples per class produces low confidence scores (~27%). A production system would require thousands of samples per specialty.
- **TF-IDF loses word order** — "no chest pain" and "chest pain" produce identical features. A transformer model (e.g. BioBERT) would capture this context.
- **No confidence threshold enforcement** — low-confidence predictions are flagged but still shown. A production system would route these to human review.
- **15 specialties only** — the original dataset has 40 specialties; tail classes were excluded due to insufficient data.

---

## Project Structure

```
├── app.py                  # Streamlit web application
├── model.pkl               # Trained Logistic Regression model
├── vectorizer.pkl          # Fitted TF-IDF vectorizer
├── requirements.txt        # Python dependencies
├── explore_data.py         # Phase 1 — dataset exploration
├── preprocess.py           # Phase 2 — text cleaning and balancing
├── features.py             # Phase 3 — TF-IDF feature engineering
├── train_model.py          # Phase 4 + 5 — training and evaluation
├── save_model.py           # Phase 6 — model persistence
└── generate_summary.py     # Phase 7 + 8 — LLM integration and pipeline
```

---

## Running Locally

```bash
# Clone the repository
git clone https://github.com/your-username/ai-medical-report-analyzer.git
cd ai-medical-report-analyzer

# Install dependencies
pip install -r requirements.txt

# Set your Groq API key
set GROQ_API_KEY=your_key_here        # Windows
export GROQ_API_KEY=your_key_here     # Mac/Linux

# Run the app
streamlit run app.py
```

---

## What I Learned

This was my first end-to-end ML project, built with genuine understanding of every layer:

- Exploratory data analysis and identifying class imbalance
- NLP preprocessing pipelines and why each step matters
- TF-IDF vectorization and sparse matrix representation
- Supervised classification with scikit-learn
- Model evaluation beyond accuracy — precision, recall, F1, confusion matrices
- Prompt engineering for domain-specific LLM outputs
- Secure API key handling via environment variables
- Streamlit application development and cloud deployment

---

## Disclaimer

This application is for educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider.