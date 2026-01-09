from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

model = joblib.load("student_model.pkl")

@app.route("/", methods=["GET"])
def home():
    return {"message": "Student Exam Prediction API running"}

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data:
        return jsonify({"error": "No input data provided"}), 400

    # Convert JSON to DataFrame
    df = pd.DataFrame([data])

    # Predict
    prediction = model.predict(df)[0]

    return jsonify({
        "predicted_exam_score": round(float(prediction), 2)
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
