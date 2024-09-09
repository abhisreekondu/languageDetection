# model.py
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

# Assuming df is your dataset with 'Text' and 'Language'
# Load or define df here
df = pd.read_csv('Language Detection.csv')  # Replace with actual data file

x = df['Text']
y = df['Language']

# Label Encoding
le = LabelEncoder()
y = le.fit_transform(y)

# Text Preprocessing
df_list = []
for text in x:
    text = re.sub(r'[!@#$(),"%^*?:~`0-9]', ' ', text)
    text = re.sub(r'\[|\]', ' ', text)
    text = text.lower()
    df_list.append(text)

# Vectorization
cv = CountVectorizer()
x = cv.fit_transform(df_list).toarray()

# Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=41)

# Model Training
ada = LogisticRegression()
y_pred = ada.fit(x_train, y_train).predict(x_test)

# Model Evaluation
ac = f1_score(y_test, y_pred, average='macro')
print(f'Accuracy={ac:.2f}')

# Prediction Function
def predict_language(text):
    text = re.sub(r'[!@#$(),"%^*?:~`0-9]', ' ', text)
    text = re.sub(r'\[|\]', ' ', text)
    text = text.lower()
    x = cv.transform([text]).toarray()
    lang = ada.predict(x)
    lang = le.inverse_transform(lang)
    return lang[0]
