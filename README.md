# AI Medical Report Analyzer

An end-to-end Natural Language Processing (NLP) and Machine Learning project that classifies medical reports into medical specialties and generates patient-friendly explanations using Large Language Models (LLMs).

## Project Overview

Healthcare reports often contain complex medical terminology that can be difficult for patients to understand. This project aims to:

* Classify medical reports into their corresponding medical specialties using Machine Learning.
* Generate patient-friendly explanations of medical reports using an LLM API.
* Provide results through an interactive web application.

## Dataset

This project uses the **Medical Transcriptions Dataset (MTSamples)** from Kaggle:

Dataset: https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions

The dataset contains thousands of real-world medical transcription samples across multiple specialties such as:

* Cardiology
* Neurology
* Orthopedics
* Gastroenterology
* Radiology
* Surgery

## Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* TF-IDF Vectorization
* Logistic Regression / Naive Bayes (planned)
* Streamlit (planned)
* LLM API Integration (planned)

## Project Pipeline

### Completed

* Dataset exploration and analysis
* Missing value handling
* Selection of top 15 medical specialties
* Class balancing through undersampling
* Text preprocessing and cleaning
* TF-IDF feature extraction
* Feature inspection and sparsity analysis

### Upcoming

* Train/Test split
* Model training
* Model evaluation
* Hyperparameter tuning
* LLM integration for patient-friendly summaries
* Streamlit web application
* Deployment

## Repository Structure


.
├── explore_data.py
├── preprocess.py
├── features.py
├── README.md
├── class_distribution.png
└── mtsamples.csv
```

## Learning Goals

This project is being developed as a hands-on learning exercise to gain practical experience in:

* Natural Language Processing (NLP)
* Machine Learning workflows
* Feature engineering
* Model evaluation
* AI application development
* Healthcare-focused AI systems

```
```
