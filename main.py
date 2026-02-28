import joblib
import numpy as np

# Load trained model
model = joblib.load("student_model.pkl")

print("✅ Model loaded successfully")

# Example student data:
# studytime, failures, absences, G1, G2
sample = np.array([[2, 0, 4, 12, 13]])

# Predict final grade
prediction = model.predict(sample)

print("Predicted Final Grade (G3):", prediction[0])