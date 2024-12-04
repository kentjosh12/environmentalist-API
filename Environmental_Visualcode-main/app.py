from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS

# Initialize the Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Load the pre-trained model in app context
with app.app_context():
    model = joblib.load('pollution_prediction_model.pkl')

# Helper function to make predictions
def make_prediction(data):
    try:
        # Extract features from the request
        pm25 = float(data.get('PM2.5'))  # Convert PM2.5 to float
        co2 = float(data.get('CO2_Level'))  # Convert CO2_Level to float
        noise = float(data.get('Noise_Level'))  # Convert Noise_Level to float
        water_ph = float(data.get('Water_pH'))  # Convert Water_pH to float

        # Prepare the feature vector for prediction
        features = np.array([[pm25, co2, noise, water_ph]])

        # Make a prediction
        prediction = model.predict(features)

        # Convert prediction back to 'Safe'/'Unsafe' label
        prediction_label = 'Safe' if prediction == 0 else 'Unsafe'

        # Return the result
        result = {'prediction': prediction_label}
        return result, 200

    except KeyError as ke:
        return {'error': f'Missing key: {str(ke)}'}, 400
    except ValueError as ve:
        return {'error': f'Invalid value: {str(ve)}'}, 400
    except Exception as e:
        return {'error': str(e)}, 500

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    print("Received data:", data)  # Log received data for debugging
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    result, status_code = make_prediction(data)
    return jsonify(result), status_code

# Define the root route
@app.route('/')
def home():
    return "Welcome to the Pollution Prediction API. Use the /predict endpoint to get predictions."

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
