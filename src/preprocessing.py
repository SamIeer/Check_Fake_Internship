# src/preprocessing.py
import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# ----------------------------
# 🔹 Text Cleaning Function
# ----------------------------
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    """Clean text: remove URLs, punctuation, stopwords, and lemmatize words."""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)     # remove URLs
    text = re.sub(r'[^a-z\s]', '', text)           # remove non-alphabetic chars
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

# ----------------------------
# 🔹 Preprocessing Function
# ----------------------------
def preprocess_data(df):
    """
    Handles missing values, combines text columns,
    cleans text, creates engineered features, vectorizes with TF-IDF,
    and splits the data for training.
    Returns: X_train, X_test, y_train, y_test
    """

    # Step 1: Keep only relevant columns
    text_cols = ['title', 'company_profile', 'description', 'requirements', 'benefits']
    meta_cols = ['telecommuting', 'has_company_logo', 'has_questions', 'fraudulent']
    df = df[text_cols + meta_cols].copy()

    # Step 2: Fill missing text columns
    for col in text_cols:
        df[col] = df[col].fillna('')

    # Step 3: Combine text columns
    df['text'] = df['title'] + ' ' + df['company_profile'] + ' ' + df['description'] + ' ' + df['benefits']
    df['text'] = df['text'].str.strip()

    # Step 4: Clean text
    df['clean_text'] = df['text'].apply(clean_text)

    # Step 5: Feature Engineering
    df['num_words'] = df['clean_text'].apply(lambda x: len(x.split()))
    df['num_unique_words'] = df['clean_text'].apply(lambda x: len(set(x.split())))
    df['num_chars'] = df['clean_text'].apply(lambda x: len(x))
    df['avg_word_len'] = df['num_chars'] / (df['num_words'] + 1)
    df['num_exclamations'] = df['clean_text'].apply(lambda x: x.count('!'))
    df['num_question_marks'] = df['clean_text'].apply(lambda x: x.count('?'))
    df['num_uppercase'] = df['clean_text'].apply(lambda x: sum(1 for w in x.split() if w.isupper()))

    # Step 6: TF-IDF Vectorization
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')
    X_tfidf = tfidf.fit_transform(df['clean_text'])

    # Step 7: Combine text + numeric features
    X_meta = df[['num_words', 'num_unique_words', 'avg_word_len',
                 'num_exclamations', 'num_question_marks', 'num_uppercase']].fillna(0)
    X_final = hstack((X_tfidf, X_meta))

    y = df['fraudulent']

    # Step 8: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y, test_size=0.2, random_state=42, stratify=y
    )

    print("✅ Preprocessing completed successfully!")
    print("Final Feature Shape:", X_final.shape)

    return X_train, X_test, y_train, y_test, tfidf
