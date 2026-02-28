
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

print("✅ Training started...")

# Load dataset (make sure file is in same folder)
data = pd.read_csv("student-mat.csv", sep=';')

print("Dataset loaded successfully")
print(data.head())

# Select important columns
data = data[["studytime", "failures", "absences", "G1", "G2", "G3"]]

# Features and target
X = data.drop("G3", axis=1)
y = data["G3"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training model...")

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluation
print("MAE:", mean_absolute_error(y_test, predictions))
print("R2 Score:", r2_score(y_test, predictions))

# Save model
joblib.dump(model, "student_model.pkl")

print("✅ Model saved as student_model.pkl")
print("🎉 Training completed!")