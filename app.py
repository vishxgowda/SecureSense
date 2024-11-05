from flask import Flask, request, jsonify
import joblib
import numpy as np
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load the trained model
model_path = '/Users/vishwanathgowda/Documents/SecureSense/your_model_file.pkl'
model = joblib.load(model_path)
logging.info("Model loaded successfully from %s", model_path)

@app.route('/')
def home():
    return "Welcome to the SecureSense Prediction API! Use the /predict endpoint to make predictions."

@app.route('/predict', methods=['POST'])
def predict():
    logging.info("Received prediction request.")
    
    data = request.get_json()

    # Validate input
    if 'features' not in data or len(data['features']) != 78:
        return jsonify({"error": "Input must contain 78 features."}), 400

    # Convert features to numpy array and reshape
    features = np.array(data['features']).reshape(1, -1)

    # Make prediction
    prediction = model.predict(features)
    probabilities = model.predict_proba(features).flatten()

    return jsonify({
        "prediction": int(prediction[0]),
        "probabilities": probabilities.tolist()  # Convert to list for JSON serialization
    })

@app.route('/metrics', methods=['GET'])
def metrics():
    # Example metrics, replace with actual calculated metrics from your training phase
    metrics = {
        "accuracy": 0.95,  # Example value
        "precision": 0.93,
        "recall": 0.90
    }
    return jsonify(metrics)

if __name__ == '__main__':
    app.run(debug=True)
