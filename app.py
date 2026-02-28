from flask import Flask, request, jsonify
import joblib
import numpy as np

# Create Flask app
app = Flask(__name__)

print("Loading trained model...")
model = joblib.load("student_model.pkl")
print("Model loaded!")

# Home route
@app.route("/")
def home():
    return "Student Performance Prediction API is running!"

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Expected input:
    # studytime, failures, absences, G1, G2
    features = np.array([[
        data["studytime"],
        data["failures"],
        data["absences"],
        data["G1"],
        data["G2"]
    ]])

    prediction = model.predict(features)

    return jsonify({
        "Predicted Final Grade (G3)": float(prediction[0])
    })

# Run server
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)