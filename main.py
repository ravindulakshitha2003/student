from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("student_exam_model.joblib")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    df = pd.DataFrame([data])

    prediction = model.predict(df)[0]

    return jsonify({
        "predicted_exam_score": round(float(prediction), 2)
    })

if __name__ == "__main__":
    app.run(port=5000)
