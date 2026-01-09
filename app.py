from fastapi import FastAPI
import pickle
import pandas as pd

app = FastAPI()

model = pickle.load(open("student_model.pkl", "rb"))

@app.get("/")
def home():
    return {"message": "Student Exam Score API running"}

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return {"predicted_exam_score": float(prediction[0])}
