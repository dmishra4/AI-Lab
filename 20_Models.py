# AI/ML/NLP Candidate Test – Complete Solution

## Part 1 — Python Notebook Code (Jupyter Notebook Ready)


# ==============================================
# AI / ML / NLP Expense Classification Project
# ==============================================

# =====================================================
# 1. Install Required Libraries (Run First if Needed)
# =====================================================

# !pip install pandas numpy matplotlib seaborn scikit-learn openpyxl nltk wordcloud

# =====================================================
# 2. Import Libraries
# =====================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import re

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# =====================================================
# 3. Load Dataset
# =====================================================

# Update file path if needed
file_path = 'data.xlsx'

# Read Excel file
# If sheet name is unknown, first check available sheets
xls = pd.ExcelFile(file_path)
print("Available Sheets:", xls.sheet_names)

# Load first sheet
# Change sheet name if required

df = pd.read_excel(file_path)

# Display first rows
print(df.head())

# =====================================================
# 4. Basic Dataset Information
# =====================================================

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# =====================================================
# 5. Data Cleaning
# =====================================================

# Remove duplicate rows
initial_rows = len(df)
df = df.drop_duplicates()
final_rows = len(df)

print(f"Removed {initial_rows - final_rows} duplicate rows")

# =====================================================
# IMPORTANT:
# Replace these column names based on actual Excel data
# =====================================================

# Example assumptions:
# remarks column -> 'Remarks'
# target label column -> 'Expense Type'

# Check actual column names from df.columns output

remarks_col = 'Remarks'
target_col = 'Expense Type'

# Remove rows where target or remarks are missing

df = df.dropna(subset=[remarks_col, target_col])

# Convert to string

df[remarks_col] = df[remarks_col].astype(str)
df[target_col] = df[target_col].astype(str)

# =====================================================
# 6. Exploratory Data Analysis (EDA)
# =====================================================

# -----------------------------
# Target Class Distribution
# -----------------------------

plt.figure(figsize=(8, 5))
sns.countplot(x=target_col, data=df)
plt.title('Expense Type Distribution')
plt.xticks(rotation=45)
plt.show()

# -----------------------------
# Remarks Length Analysis
# -----------------------------

df['remark_length'] = df[remarks_col].apply(len)

plt.figure(figsize=(10, 5))
sns.histplot(df['remark_length'], bins=30)
plt.title('Distribution of Remark Length')
plt.xlabel('Text Length')
plt.show()

# -----------------------------
# Average Remark Length by Category
# -----------------------------

avg_length = df.groupby(target_col)['remark_length'].mean()
print(avg_length)

# =====================================================
# 7. Text Preprocessing
# =====================================================

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    # Lowercase
    text = text.lower()

    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenization
    words = text.split()

    # Remove stopwords and lemmatize
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

    return ' '.join(words)


# Apply preprocessing

df['cleaned_text'] = df[remarks_col].apply(preprocess_text)

print(df[['cleaned_text']].head())

# =====================================================
# 8. Train Test Split
# =====================================================

X = df['cleaned_text']
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training Size:", len(X_train))
print("Testing Size:", len(X_test))

# =====================================================
# 9. APPROACH 1
# TF-IDF + Logistic Regression
# =====================================================

model_lr = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Train model
model_lr.fit(X_train, y_train)

# Prediction
pred_lr = model_lr.predict(X_test)

# =====================================================
# 10. Evaluate Logistic Regression Model
# =====================================================

print("\n===== Logistic Regression Results =====")

accuracy_lr = accuracy_score(y_test, pred_lr)
f1_lr = f1_score(y_test, pred_lr, average='weighted')

print("Accuracy:", accuracy_lr)
print("F1 Score:", f1_lr)

print("\nClassification Report:")
print(classification_report(y_test, pred_lr))

# Confusion Matrix

cm = confusion_matrix(y_test, pred_lr)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Logistic Regression')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# =====================================================
# 11. APPROACH 2
# TF-IDF + Naive Bayes
# =====================================================

model_nb = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('classifier', MultinomialNB())
])

# Train
model_nb.fit(X_train, y_train)

# Predict
pred_nb = model_nb.predict(X_test)

# =====================================================
# 12. Evaluate Naive Bayes Model
# =====================================================

print("\n===== Naive Bayes Results =====")

accuracy_nb = accuracy_score(y_test, pred_nb)
f1_nb = f1_score(y_test, pred_nb, average='weighted')

print("Accuracy:", accuracy_nb)
print("F1 Score:", f1_nb)

print("\nClassification Report:")
print(classification_report(y_test, pred_nb))

# =====================================================
# 13. Model Comparison
# =====================================================

comparison = pd.DataFrame({
    'Model': ['Logistic Regression', 'Naive Bayes'],
    'Accuracy': [accuracy_lr, accuracy_nb],
    'F1 Score': [f1_lr, f1_nb]
})

print(comparison)

# =====================================================
# 14. Final Model Selection
# =====================================================

if f1_lr >= f1_nb:
    final_model = model_lr
    print("Final Selected Model: Logistic Regression")
else:
    final_model = model_nb
    print("Final Selected Model: Naive Bayes")

# =====================================================
# 15. Test Custom Predictions
# =====================================================

sample_texts = [
    'Purchased server equipment for IT team',
    'Consulting charges for audit services',
    'Construction raw material supplied'
]

sample_cleaned = [preprocess_text(text) for text in sample_texts]

predictions = final_model.predict(sample_cleaned)

for text, pred in zip(sample_texts, predictions):
    print(f"Text: {text}")
    print(f"Predicted Category: {pred}")
    print("-" * 50)

# =====================================================
# 16. Save Final Model (Optional)
# =====================================================

import pickle

with open('expense_classifier_model.pkl', 'wb') as f:
    pickle.dump(final_model, f)

print("Model saved successfully")

# =====================================================
# 17. Future Improvements
# =====================================================

print("""
Possible Future Enhancements:
1. Use advanced transformer models like BERT
2. Hyperparameter tuning
3. Use cross-validation
4. Handle class imbalance
5. Deploy using Flask/FastAPI
6. Add real-time prediction APIs
""")


