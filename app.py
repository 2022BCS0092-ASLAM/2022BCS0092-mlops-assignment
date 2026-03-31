from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd

app = FastAPI()

# Load model
model = joblib.load("models/model.pkl")

# Fixed feature order (VERY IMPORTANT)
FEATURE_ORDER = [
    "fixed_acidity", "volatile_acidity", "citric_acid",
    "residual_sugar", "chlorides", "free_sulfur_dioxide",
    "total_sulfur_dioxide", "density", "pH",
    "sulphates", "alcohol"
]

# Health endpoint (MANDATORY)
@app.get("/")
def health():
    return {
        "name": "Mohammed Aslam",
        "roll_no": "2022BCS0092"
    }

# Prediction endpoint
@app.post("/predict")
def predict(data: dict):
    try:
        # Validate input keys
        for feature in FEATURE_ORDER:
            if feature not in data:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing feature: {feature}"
                )

        # Ensure correct order
        values = [data[feature] for feature in FEATURE_ORDER]

        # Convert to DataFrame (fixes sklearn warning)
        input_df = pd.DataFrame([values], columns=FEATURE_ORDER)

        # Prediction
        prediction = model.predict(input_df)[0]

        return {
            "prediction": int(prediction),
            "name": "Mohammed Aslam",
            "roll_no": "2022BCS0092"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))