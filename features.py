import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

#load data set
df=pd.read_csv("mtsamples_cleaned.csv")
print(f"Loaded {len(df)} rows")

#defining x and y 
X=df['clean_transcription']
Y=df['medical_specialty']
print(f"\nX shape: {X.shape}")
print(f"Y shape: {Y.shape}")
print(f"\nSamples X: {X.iloc[0][:100]}")
print(f"\nSamples Y: {Y.iloc[0]}")

#building tf-id features
vectorizer= TfidfVectorizer(
    max_features=5000,#limits 5000 frequent words
    stop_words='english',#removes common stop words
    ngram_range=(1,2) #can help work w both unigram and bigram words
)

#fit and transform
X_tfidf=vectorizer.fit_transform(X)
print(f"\nTF-IDF Matrix shape: {X_tfidf.shape}")
print(f"This means : {X_tfidf.shape[0]} reports x {X_tfidf.shape[1]} features")

#insepcting
feature_names= vectorizer.get_feature_names_out()
print(f"\nSample features the model learned:")
print(feature_names[:20].tolist())
print("...")
print(feature_names[-20:].tolist())

#checks if sparse matrix is seen
total_element= X_tfidf.shape[0]*X_tfidf.shape[1]
non_zero=X_tfidf.nnz
sparsity=(1-non_zero/ total_element)*100
print(f"\nMatrix Sparsity: {sparsity:.1f}% zeros")
print("(This is normal and expected for text data)")