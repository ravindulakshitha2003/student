# app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np

# -------------------------------
# Step 1: Load your trained model
# -------------------------------
model = joblib.load("student_model.pkl")  # Make sure model.pkl is in the same folder

# -------------------------------
# Step 2: Create FastAPI app
# -------------------------------
app = FastAPI()

# -------------------------------
# Step 3: Enable CORS for all origins
# -------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Step 4: Define the expected input
# -------------------------------
class StudentData(BaseModel):
    age: float
    gender: str
    course: str
    study_hours: float
    class_attendance: float
    internet_access: str
    sleep_hours: float
    sleep_quality: str
    study_method: str
    facility_rating: str
    exam_difficulty: str

# -------------------------------
# Step 5: Define the prediction endpoint
# -------------------------------
@app.post("/predict")
def predict(data: StudentData):
    # Convert input to list (you may need to encode categorical variables the same way as during training)
    input_data = [
        data.age,
        1 if data.gender.lower() == "male" else 0,
        1 if data.course.lower() == "diploma" else 0,  # simple encoding example
        data.study_hours,
        data.class_attendance,
        1 if data.internet_access.lower() == "yes" else 0,
        data.sleep_hours,
        1 if data.sleep_quality.lower() == "good" else 0,
        1 if data.study_method.lower() == "coaching" else 0,
        1 if data.facility_rating.lower() == "high" else 0,
        1 if data.exam_difficulty.lower() == "hard" else 0
    ]

    # Convert to numpy array and reshape
    input_array = np.array(input_data).reshape(1, -1)

    # Make prediction
    prediction = model.predict(input_array)

    return {"predicted_value": float(prediction[0])}

# -------------------------------
# Step 6: Root endpoint (optional)
# -------------------------------
@app.get("/")
def read_root():
    return {"message": "Student Exam Prediction API is running"}
