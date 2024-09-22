# model.py
import re
import numpy as np
import pandas as pd
import nltk
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

# NLTK Downloads (only need to run once)
nltk.download('punkt')
nltk.download('punkt_tab')
# nltk.download('stopwords')
# nltk.download('wordnet')

# NLTK imports
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Initialize NLTK tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Load dataset
df = pd.read_csv('Language Detection.csv')  # Replace with actual data file

x = df['Text']
y = df['Language']

# Label Encoding
le = LabelEncoder()
y = le.fit_transform(y)

# NLTK-based Text Preprocessing
def preprocess_text(text):
    # Remove special characters and digits
    text = re.sub(r'[!@#$(),"%^*?:~`0-9]', ' ', text)
    text = re.sub(r'\[|\]', ' ', text)
    text = text.lower()
    
    # Tokenization using NLTK
    tokens = word_tokenize(text)
    
    # Remove stopwords and apply lemmatization
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    # Join tokens back into a single string
    return ' '.join(tokens)

# Apply preprocessing to the dataset
df_list = [preprocess_text(text) for text in x]

# Vectorization using CountVectorizer
cv = CountVectorizer()
x = cv.fit_transform(df_list).toarray()

# Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=41)

# Model Training using Logistic Regression
ada = LogisticRegression(max_iter=1000)  # max_iter increased to ensure convergence
y_pred = ada.fit(x_train, y_train).predict(x_test)

# Model Evaluation
ac = f1_score(y_test, y_pred, average='macro')
print(f'F1 Score={ac:.2f}')

# Prediction Function with Preprocessing
def predict_language(text):
    text = preprocess_text(text)  # Preprocess the input text using NLTK
    x = cv.transform([text]).toarray()
    lang = ada.predict(x)
    lang = le.inverse_transform(lang)
    return lang[0]

# # Example usage:
# new_text = "Bonjour tout le monde"
# detected_language = predict_language(new_text)
# print(f"Detected Language: {detected_language}")
