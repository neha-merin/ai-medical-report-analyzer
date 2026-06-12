import re

text = "PREOPERATIVE DIAGNOSES:,1.  Lumbar osteomyelitis."

print("original:  ", text)

text = text.lower()
print("lowercase: ", text)

text = re.sub(r'\d+', '', text)
print("no numbers:", text)

text = re.sub(r'[^\w\s]', '', text)
print("no punct:  ", text)

text = re.sub(r'\s+', ' ', text).strip()
print("collapsed: ", text)