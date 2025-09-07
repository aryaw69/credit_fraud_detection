from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("fraud.pkl")
scaler = joblib.load("scaler.pkl")

app = Flask(__name__)

# Home route (for testing)
@app.route("/")
def home():
    return "✅ Fraud Detection API is running!"

# Predict route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data
        data = request.get_json()

        # Extract features
        # Expecting data like: {"V1": ..., "V2": ..., ..., "V28": ..., "Amount": ...}
        features = []
        for i in range(1, 29):   # V1 to V28
            features.append(float(data.get(f"V{i}", 0)))
        
        # Scale amount
        amount = scaler.transform(np.array([[data["Amount"]]]))[0][0]
        features.append(amount)

        # Convert to numpy array and reshape
        features = np.array(features).reshape(1, -1)

        # Predict
        prediction = model.predict(features)[0]
        result = "Fraudulent" if prediction == 1 else "Legit"

        return jsonify({"prediction": int(prediction), "result": result})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
