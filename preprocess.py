import pandas as pd
import re 

df=pd.read_csv("mtsamples.csv")

#remove nulls in transcription[33]
df=df.dropna(subset=['transcription'])
print(f"Rows after dropping missing transcriptions: {len(df)}")

#top 15 specialities so surgery isnt only studied
top_15=df['medical_specialty'].value_counts().index[:15] #by freq sorted
df=df[df['medical_specialty'].isin(top_15)]
print(f"Rows after keeping top 15 specialities: {len(df)}")
print(f"Specialties kept: {df['medical_specialty'].nunique()}")

#undersample to keep 90 per class
df_balanced = (
    df.groupby('medical_specialty', group_keys=False)[df.columns]
    .apply(lambda x: x.sample(min(len(x), 90), random_state=42))
)
print(df_balanced.columns)
print(df_balanced.head())
print(f"\nRows after undersampling: {len(df_balanced)}")
print("\nClass distribution after balancing:")
print(df_balanced['medical_specialty'].value_counts())

#text cleaning function
def clean_text(text):
    text=text.lower()
    text=re.sub(r'\d+','',text) #remove numbers
    text=re.sub(r'[^\w\s]',' ',text)#removes punctuation
    text=re.sub(r'\s+',' ',text).strip()#removes multiple spaces
    return text

#apply the fn
df_balanced=df_balanced.copy()
df_balanced['clean_transcription']= df_balanced['transcription'].apply(clean_text)

#verify cleaning
print("\n---------BEFORE CLEANING----------")
print(df_balanced['transcription'].iloc[0][:300])

print("\n---------AFTER CLEANING----------")
print(df_balanced['clean_transcription'].iloc[0][:300])

#save new cleaned dataset
df_balanced.to_csv("mtsamples_cleaned.csv",index=False)
print("\n Cleaned dataset saved as mtsamples_cleaned.csv")
