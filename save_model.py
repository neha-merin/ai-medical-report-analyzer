import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#rebuild everything for final model
df=pd.read_csv("mtsamples_cleaned.csv")
X=df['clean_transcription']
Y=df['medical_specialty']

vectorizer=TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    ngram_range=(1,2)
)
X_tfidf=vectorizer.fit_transform(X)
model=LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_tfidf,Y)

print("Model trained on full dataset.")
print(f"Training samples used: {X_tfidf.shape[0]}")

#saving model and vectorizer
with open("model.pkl","wb") as f:
    pickle.dump(model,f)

with open("vectorizer.pkl","wb") as f:
    pickle.dump(vectorizer,f)
print("Model saved in model.pkl")
print("Vectorizer saved in vectorizer.pkl")

#verify by loading and predicting
with open("model.pkl","rb") as f:
    loaded_model=pickle.load(f)

with open("vectorizer.pkl","rb") as f:
    loaded_vectorizer = pickle.load(f)

#test on real samples
test_report=[df["clean_transcription"].iloc[42]]
test_features=loaded_vectorizer.transform(test_report)
prediction=loaded_model.predict(test_features)

print(f"\nVerification prediction:")
print(f"Report preveiw:{test_report[0][:100]}")
print(f"Predicted specialty: {prediction[0]}")
print(f"Actual specialty: {df['medical_specialty'].iloc[42]}")
