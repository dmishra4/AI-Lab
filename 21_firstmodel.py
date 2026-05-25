import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import joblib
from flask import Flask, request, jsonify

# ==========================================
# STEP 1 - LOAD EXCEL
# ==========================================

df = pd.read_excel("data_with_fixed_categories.xlsx")

print("Original Shape:", df.shape)

# ==========================================
# STEP 2 - CLEAN DATA
# ==========================================

# Remove null values
df = df.dropna(subset=["Remarks", "Category"])

# Remove duplicate rows
df = df.drop_duplicates()

# Remove extra spaces
df["Remarks"] = df["Remarks"].str.strip()

# Convert to lowercase
df["Remarks"] = df["Remarks"].str.lower()

print("Cleaned Shape:", df.shape)

# ==========================================
# STEP 3 - INPUT AND OUTPUT
# ==========================================

# X = Input text
X = df["Remarks"]

# y = Correct answers / labels
y = df["Category"]

# ==========================================
# STEP 4 - TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================
# STEP 5 - PRINT DETAILS
# ==========================================

print("\nTraining Data Size:", len(X_train))
print("Testing Data Size:", len(X_test))

print("\nSample Training Data:")
print(X_train.head())

print("\nSample Training Labels:")
print(y_train.head())

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# ==========================================
# STEP 1 - LOAD EXCEL
# ==========================================

df = pd.read_excel("data_with_fixed_categories.xlsx")

print("Original Shape:", df.shape)

# ==========================================
# STEP 2 - CLEAN DATA
# ==========================================

# Remove null values
df = df.dropna(subset=["Remarks", "Category"])

# Remove duplicate rows
df = df.drop_duplicates()

# Remove extra spaces
df["Remarks"] = df["Remarks"].str.strip()

# Convert to lowercase
df["Remarks"] = df["Remarks"].str.lower()

print("Cleaned Shape:", df.shape)

# ==========================================
# STEP 3 - INPUT AND OUTPUT
# ==========================================

# X = Input text
X = df["Remarks"]

# y = Correct answers / labels
y = df["Category"]

# ==========================================
# STEP 4 - TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================
# STEP 5 - PRINT DETAILS
# ==========================================

print("\nTraining Data Size:", len(X_train))
print("Testing Data Size:", len(X_test))

print("\nSample Training Data:")
print(X_train.head())

print("\nSample Training Labels:")
print(y_train.head())

# ==========================================
# STEP 6 - TF-IDF VECTORIZATION
# ==========================================

vectorizer = TfidfVectorizer()

# Learn vocabulary + create vectors
X_train_vectors = vectorizer.fit_transform(X_train)

# Convert test data into vectors
X_test_vectors = vectorizer.transform(X_test)

# ==========================================
# STEP 7 - PRINT VECTOR DETAILS
# ==========================================

print("\n===== VECTOR INFORMATION =====")

print("\nTraining Vector Shape:")
print(X_train_vectors.shape)

print("\nTesting Vector Shape:")
print(X_test_vectors.shape)

# ==========================================
# STEP 8 - PRINT VOCABULARY / INDEXING
# ==========================================

print("\n===== VOCABULARY =====")

vocab = vectorizer.vocabulary_

print("Total Vocabulary Words:", len(vectorizer.vocabulary_))

# Print first 20 words with indexes
for word, index in list(vocab.items())[:20]:
    print(f"{word} --> {index}")

# ==========================================
# STEP 9 - PRINT SAMPLE VECTOR
# ==========================================

print("\n===== SAMPLE VECTOR =====")

sample_vector = X_train_vectors[0]

print(sample_vector)

# ==========================================
# STEP 10 - CONVERT VECTOR TO ARRAY
# ==========================================

print("\n===== VECTOR ARRAY =====")

vector_array = sample_vector.toarray()

print(vector_array)

# ==========================================
# STEP 11 - FEATURE NAMES
# ==========================================

print("\n===== FEATURE NAMES =====")

feature_names = vectorizer.get_feature_names_out()

print(feature_names[:20])
print("All Vocabulary Counts   :: ",len(feature_names))
print("All Vocabulary Words   :: ", feature_names)

# ==========================================
# STEP 12 - MODEL TRAINING
# ==========================================

print("\n===== MODEL TRAINING =====")

# Create model object
model = MultinomialNB()

# Train model using training vectors + labels
model.fit(X_train_vectors, y_train)

print("Model Training Completed Successfully")

# ==========================================
# STEP 13 - PREDICTION
# ==========================================

print("\n===== MODEL PREDICTION =====")

# Predict categories for test data
predictions = model.predict(X_test_vectors)

print("Predictions Completed")

# Print first 10 predictions
for actual, predicted in zip(y_test[:10], predictions[:10]):
    print(f"Actual: {actual}  --->  Predicted: {predicted}")

    # ==========================================
    # STEP 14 - ACCURACY CHECK
    # ==========================================

    print("\n===== MODEL ACCURACY =====")

    accuracy = accuracy_score(y_test, predictions)

    print("Accuracy:", accuracy)

    print("Accuracy Percentage:", round(accuracy * 100, 2), "%")

    # ==========================================
    # STEP 15 - CLASSIFICATION REPORT
    # ==========================================


    print("\n===== CLASSIFICATION REPORT =====")

    print(classification_report(y_test, predictions))

    # ==========================================
    # STEP 16 - TEST YOUR OWN INPUT
    # ==========================================

    print("\n===== CUSTOM PREDICTION =====")

    custom_text = [
        "supply and installation of ups battery"
    ]

    # Convert custom text into vector
    custom_vector = vectorizer.transform(custom_text)

    # Predict category
    custom_prediction = model.predict(custom_vector)

    print("Input Text:", custom_text[0])

    print("Predicted Category:", custom_prediction[0])

    # ==========================================
    # STEP 17 - PREDICTION PROBABILITY
    # ==========================================

    print("\n===== PREDICTION CONFIDENCE =====")

    probabilities = model.predict_proba(custom_vector)

    classes = model.classes_

    for category, probability in zip(classes, probabilities[0]):
        print(f"{category} --> {round(probability * 100, 2)}%")

        # ==========================================
        # STEP 18 - SAVE MODEL
        # ==========================================

        print("\n===== SAVING MODEL =====")

        # Save trained model
        joblib.dump(model, "category_prediction_model.pkl")

        # Save TF-IDF vectorizer
        joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

        print("Model Saved Successfully")
        print("Vectorizer Saved Successfully")

        # ==========================================
        # LOAD SAVED MODEL AND VECTORIZER
        # ==========================================

        model = joblib.load("category_prediction_model.pkl")   # Load trained ML Model

        vectorizer = joblib.load("tfidf_vectorizer.pkl")  #  Load Trained TF-IDF Vectorizer

        # ==========================================
        # CREATE FLASK APP
        # ==========================================

        app = Flask(__name__)   #  Create Flask Application Object


        # ==========================================
        # HOME API
        # ==========================================

        @app.route("/")
        def home():
            return "AI Category Prediction API is Running"


        # ==========================================
        # PREDICTION API
        # ==========================================

        @app.route("/predict", methods=["POST"])
        def predict():

            # Get JSON data from request
            data = request.get_json()

            # Extract text
            input_text = data["text"]

            # Convert text into vector
            input_vector = vectorizer.transform([input_text])

            # Predict category
            prediction = model.predict(input_vector)[0]

            # Prediction probabilities
            probabilities = model.predict_proba(input_vector)[0]

            # Category names
            classes = model.classes_

            # Create confidence dictionary
            confidence_scores = {}

            for category, probability in zip(classes, probabilities):
                confidence_scores[category] = round(float(probability) * 100, 2)

            # Final JSON response
            result = {
                "input_text": input_text,
                "predicted_category": prediction,
                "confidence_scores": confidence_scores
            }

            return jsonify(result)


        # ==========================================
        # RUN APPLICATION
        # ==========================================

        if __name__ == "__main__":
            app.run(debug=True)