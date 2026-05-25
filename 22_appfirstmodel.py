from flask import Flask, request, render_template_string
import joblib

# ==========================================
# LOAD MODEL AND VECTORIZER
# ==========================================

model = joblib.load("category_prediction_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# ==========================================
# CREATE FLASK APP
# ==========================================

app = Flask(__name__)

# ==========================================
# MODERN HTML UI
# ==========================================

HTML_PAGE = """

<!DOCTYPE html>
<html>

<head>

    <title>AI Category Prediction</title>

    <style>

        body {
            margin: 0;
            padding: 0;
            font-family: Arial, sans-serif;
            background: linear-gradient(to right, #141e30, #243b55);
            color: white;
        }

        .container {
            width: 60%;
            margin: 50px auto;
            background: white;
            color: black;
            padding: 40px;
            border-radius: 15px;
            box-shadow: 0px 0px 20px rgba(0,0,0,0.3);
        }

        h1 {
            text-align: center;
            color: #243b55;
        }

        textarea {
            width: 100%;
            padding: 15px;
            font-size: 16px;
            border-radius: 10px;
            border: 1px solid #ccc;
            resize: none;
        }

        button {
            background: #243b55;
            color: white;
            border: none;
            padding: 15px 25px;
            font-size: 16px;
            border-radius: 10px;
            cursor: pointer;
            margin-top: 15px;
        }

        button:hover {
            background: #141e30;
        }

        .result-box {
            margin-top: 30px;
            padding: 20px;
            border-radius: 10px;
            background: #f4f4f4;
        }

        .prediction {
            font-size: 24px;
            font-weight: bold;
            color: green;
        }

        ul {
            list-style-type: none;
            padding: 0;
        }

        li {
            background: #e8e8e8;
            margin: 8px 0;
            padding: 10px;
            border-radius: 8px;
        }

        .footer {
            text-align: center;
            margin-top: 20px;
            color: gray;
        }

    </style>

</head>

<body>

    <div class="container">

        <h1>AI Category Prediction System</h1>

        <form method="POST">

            <label><b>Enter Remark:</b></label><br><br>

            <textarea
                name="remark"
                rows="5"
                placeholder="Enter your text here..."
            ></textarea>

            <br>

            <button type="submit">
                Predict Category
            </button>

        </form>

        {% if prediction %}

        <div class="result-box">

            <h2>Prediction Result</h2>

            <p><b>Input Text:</b></p>

            <p>{{ input_text }}</p>

            <p class="prediction">
                Predicted Category : {{ prediction }}
            </p>

            <h3>Confidence Scores</h3>

            <ul>
                {% for category, score in confidence.items() %}
                    <li>
                        <b>{{ category }}</b> : {{ score }}%
                    </li>
                {% endfor %}
            </ul>

        </div>

        {% endif %}

        <div class="footer">
            Powered By NLP + Machine Learning
        </div>

    </div>

</body>

</html>

"""

# ==========================================
# HOME PAGE
# ==========================================

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    confidence_scores = {}
    input_text = ""

    if request.method == "POST":

        input_text = request.form["remark"]

        # Convert text to vector
        input_vector = vectorizer.transform([input_text])

        # Predict category
        prediction = model.predict(input_vector)[0]

        # Prediction probabilities
        probabilities = model.predict_proba(input_vector)[0]

        classes = model.classes_

        # Confidence score dictionary
        for category, probability in zip(classes, probabilities):
            confidence_scores[category] = round(float(probability) * 100, 2)

    return render_template_string(
        HTML_PAGE,
        prediction=prediction,
        confidence=confidence_scores,
        input_text=input_text
    )

# ==========================================
# RUN APPLICATION
# ==========================================

if __name__ == "__main__":
    app.run(debug=True)