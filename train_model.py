import pandas as pd
import string
import pickle
import nltk

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Download stopwords
nltk.download('stopwords')

# =========================
# LOAD DATASET
# =========================

df = pd.read_csv('spam.csv')

# Keep required columns
df = df[['label', 'text']]

# Rename columns
df.columns = ['label', 'message']

# =========================
# CLEAN LABELS
# =========================

df['label'] = df['label'].astype(str)

df['label'] = df['label'].str.lower()

df['label'] = df['label'].str.strip()

# Convert labels
df['label'] = df['label'].replace({
    'ham': 0,
    'spam': 1
})

# Remove invalid rows
df = df[df['label'].isin([0, 1])]

# =========================
# TEXT PREPROCESSING
# =========================

stop_words = set(stopwords.words('english'))

def clean_text(text):

    # Convert to lowercase
    text = str(text).lower()

    # Remove punctuation
    text = ''.join(
        char for char in text
        if char not in string.punctuation
    )

    # Split words
    words = text.split()

    # Remove stopwords
    words = [
        word for word in words
        if word not in stop_words
    ]

    return ' '.join(words)

# Apply preprocessing
df['message'] = df['message'].apply(clean_text)

# =========================
# FEATURES AND LABELS
# =========================

X = df['message']

y = df['label']

# =========================
# TF-IDF VECTORIZATION
# =========================

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(X)

# =========================
# MODEL
# =========================

model = LogisticRegression(
    max_iter=1000
)

# Train model using full dataset
model.fit(X, y)

# =========================
# SAVE MODEL
# =========================

pickle.dump(
    model,
    open('model.pkl', 'wb')
)

pickle.dump(
    vectorizer,
    open('vectorizer.pkl', 'wb')
)

print("Model trained and saved successfully")