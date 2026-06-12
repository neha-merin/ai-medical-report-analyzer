import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import(
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score
)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


#load cleaned data
df=pd.read_csv("mtsamples_cleaned.csv")
X= df['clean_transcription']
Y=df['medical_specialty']

#tf-idf for consistent preprocessing
vectorizer= TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    ngram_range=(1,2)
)
X_tfidf=vectorizer.fit_transform(X)

#spliting train and test data
X_train, X_test, Y_train, Y_test=train_test_split(
    X_tfidf, Y,
    test_size=0.2,
    random_state=42,
    stratify=Y
)
print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

#apply lr MODEL 1 
lr_model=LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train,Y_train)

lr_predictions= lr_model.predict(X_test)
lr_accuracy =accuracy_score(Y_test,lr_predictions)
print(f"\nLogistic Regression Accuracy:{lr_accuracy:.4f}")

#apply naive bayes MODEL 2
nb_model=MultinomialNB()
nb_model.fit(X_train,Y_train)

nb_predicition=nb_model.predict(X_test)
nb_accuracy=accuracy_score(Y_test,nb_predicition)
print(f"Naive Bayes Accuracy: {nb_accuracy:.4f}")

#compare train vs test accuracy
lr_train_accuracy = accuracy_score(Y_train,lr_model.predict(X_train))
nb_train_accuracy = accuracy_score(Y_train,nb_model.predict(X_train))

print(f"\n{'Model':<25} {'Train Acc':>10}{'Test Acc':>10}{'Gap':>10}")
print("-"*55)
print(f"{'Logisitic Regression':<25}{lr_train_accuracy:>10.4f} {lr_accuracy:>10.4f} {lr_train_accuracy - lr_accuracy:>10.4f}")
print(f"{'Naive Bayes':<25}{nb_train_accuracy:>10.4f} {nb_accuracy:>10.4f} {nb_train_accuracy - nb_accuracy:>10.4f}")

#classification report
print("="*60)
print("LOGISTIC REGRESSION-FULL EVALUATION")
print("="*60)
print(classification_report(Y_test,lr_predictions))

print("="*60)
print("NAIVE BAYES - FULL EVALUATION")
print("="*60)
print(classification_report(Y_test,nb_predicition))

#now we find the weakest, classs by sorting the f1 score from worst to bes
from sklearn.metrics import f1_score
classes=lr_model.classes_
lr_f1_per_class=f1_score(Y_test,lr_predictions, average=None,labels=classes)
print("\nLogistic Regression - F1 per specialty(worse to best): ")
f1_df=pd.DataFrame({
    'Speciality':classes,
    'F1 Score': lr_f1_per_class
}).sort_values('F1 Score')
print(f1_df.to_string(index=False))

#construct confusion matrix
fig,axes=plt.subplots(1,2,figsize=(20,8))

for ax,model,preds,title in zip(
axes,
    [lr_model, nb_model],
    [lr_predictions, nb_predicition],
    ['Logistic Regression', 'Naive Bayes']
):
    cm = confusion_matrix(Y_test, preds, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(ax=ax, xticks_rotation=90, colorbar=False)
    ax.set_title(title, fontsize=13)

plt.suptitle("Confusion Matrices — Medical Specialty Classification", fontsize=15)
plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150, bbox_inches='tight')
plt.show()
print("\nConfusion matrix saved as confusion_matrices.png")

# ── 4. FINAL MODEL COMPARISON ─────────────────────────────────────────
from sklearn.metrics import f1_score as f1

print("\n" + "=" * 55)
print(f"{'Metric':<30} {'Log. Reg':>10} {'Naive Bayes':>12}")
print("-" * 55)
print(f"{'Train Accuracy':<30} {lr_train_accuracy:>10.4f} {nb_train_accuracy:>12.4f}")
print(f"{'Test Accuracy':<30} {lr_accuracy:>10.4f} {nb_accuracy:>12.4f}")
print(f"{'F1 Score (weighted avg)':<30} {f1(Y_test, lr_predictions, average='weighted'):>10.4f} {f1(Y_test, nb_predicition, average='weighted'):>12.4f}")
print(f"{'Train/Test Gap':<30} {lr_train_accuracy-lr_accuracy:>10.4f} {nb_train_accuracy-nb_accuracy:>12.4f}")
print("=" * 55)