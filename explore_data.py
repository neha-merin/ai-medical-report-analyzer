import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#loading the dataset
df=pd.read_csv("mtsamples.csv")

#basic shape
print("SHAPE: ",df.shape)

#understanding field
print("\nColumns: ",df.columns.tolist())

#print first few rows
print("\nFirst few rows: ")
print(df.head(3))

#understand the datatypes
print("\nData Types: ")
print(df.dtypes)

#missing value is checked
print("\nMissing Values: ")
print(df.isnull().sum())

#class distribution
print("\nClass Distribution: ")
print(df['medical_specialty'].value_counts().head(15))

#visualize class distribution
plt.figure(figsize=(12,6))
df['medical_specialty'].value_counts().head(20).plot(kind='bar')
plt.title("Top 20 Medical Specialities by report count")
plt.xlabel("Speciality")
plt.ylabel("Number of Reports")
plt.xticks(rotation=45,ha='right')
plt.tight_layout()
plt.savefig("class_distribution.png")
plt.show()
print("\nCharts saved as class_distribution.png")

#read sample reports
print("\n--- SAMPLE REPORT 1 ---")
print("Specialty:", df['medical_specialty'].iloc[0])
print("Transcription preview:", str(df['transcription'].iloc[0])[:500])

print("\n--- SAMPLE REPORT 2 ---")
print("Specialty:", df['medical_specialty'].iloc[10])
print("Transcription preview:", str(df['transcription'].iloc[10])[:500])