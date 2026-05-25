# ==============================
# STEP 1 - IMPORT LIBRARIES
# ==============================

import pandas as pd
import numpy as np
import re
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from gensim.models import Word2Vec

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# ==============================
# STEP 2 - DOWNLOAD NLTK DATA
# ==============================

nltk.download('punkt')
nltk.download('stopwords')


# ==============================
# STEP 3 - LOAD EXCEL FILE
# ==============================

file_path = "large_raw_sentiment_analysis_dataset.xlsx"

df = pd.read_excel(file_path)

print(df.head())


# ==============================
# STEP 4 - SELECT COLUMNS
# ==============================

text_col = "Review_Text"
label_col = "Sentiment"


# ==============================
# STEP 5 - REMOVE NULL VALUES
# ==============================

df = df.dropna(subset=[text_col, label_col])

# remove empty rows

df = df[df[text_col].astype(str).str.strip() != ""]

print("Remaining Rows :", len(df))


# ==============================
# STEP 6 - TEXT CLEANING
# ==============================

stop_words = set(stopwords.words('english'))

# Keep important negation words
stop_words.remove('not')
stop_words.remove('no')


def clean_text(text):

    # Handle null values
    # if pd.isna(text):
    #     return[]
    # convert to lowercase
    text = text.lower()

    # remove special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # tokenize
    #words = word_tokenize(text)
    words = text.split()

    # remove stopwords
    words = [word for word in words if word not in stop_words]

    return words


# ==============================
# STEP 7 - APPLY CLEANING
# ==============================

df['Cleaned_Text'] = df[text_col].apply(clean_text)

print(df[['Review_Text', 'Cleaned_Text']].head())


# ==============================
# STEP 8 - LABEL ENCODING
# ==============================

encoder = LabelEncoder()

df['Label'] = encoder.fit_transform(df[label_col])

print(df[['Sentiment', 'Label']].head())


# ==============================
# STEP 9 - TRAIN WORD2VEC
# ==============================

word2vec_model = Word2Vec(
    sentences=df['Cleaned_Text'],
    vector_size=100,
    window=5,
    min_count=1,
    workers=4
)

print("Word2Vec Training Completed")


# ==============================
# STEP 10 - CONVERT SENTENCE
# TO VECTOR
# ==============================

def sentence_vector(words, model):

    vectors = []

    for word in words:

        if word in model.wv:
            vectors.append(model.wv[word])

    if len(vectors) == 0:
        return np.zeros(model.vector_size)

    return np.mean(vectors, axis=0)


# ==============================
# STEP 11 - CREATE FEATURES
# ==============================

X = np.array([
    sentence_vector(words, word2vec_model)
    for words in df['Cleaned_Text']
])

y = df['Label']

print("Feature Shape :", X.shape)


# ==============================
# STEP 12 - TRAIN TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Train Size :", len(X_train))
print("Test Size :", len(X_test))


# ==============================
# STEP 13 - MODEL TRAINING
# ==============================

model = model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced'
)

model.fit(X_train, y_train)

print("Model Training Completed")

# ==============================
# STEP 14 - PREDICTION
# ==============================

y_pred = model.predict(X_test)

print("Predictions Completed")

# Print first 10 predictions
for actual, predicted in zip(y_test[:10], y_pred[:10]):
    print(f"Actual: {actual}  --->  Predicted: {predicted}")

    # ==========================================
    # STEP 14 - ACCURACY CHECK
    # ==========================================

print("\n===== MODEL ACCURACY =====")

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

print("Accuracy Percentage:", round(accuracy * 100, 2), "%")

# ==========================================
# STEP 15 - CLASSIFICATION REPORT
# ==========================================


print("\n===== CLASSIFICATION REPORT =====")

print(classification_report(y_test, y_pred))

# ==========================================
# STEP 16 - TEST YOUR OWN INPUT
# ==========================================

print("\n===== CUSTOM PREDICTION =====")

custom_text = "service was acceptable"

# Clean custom text
cleaned_custom_text = clean_text(custom_text)

# Convert sentence into vector
custom_vector = sentence_vector(
    cleaned_custom_text,
    word2vec_model
)

# Reshape because model expects 2D input
custom_vector = custom_vector.reshape(1, -1)

# Predict label
custom_prediction = model.predict(custom_vector)

# Convert numeric label back to text
predicted_label = encoder.inverse_transform(custom_prediction)

print("Input Text :", custom_text)

print("Cleaned Text :", cleaned_custom_text)

print("Predicted Sentiment :", predicted_label[0])