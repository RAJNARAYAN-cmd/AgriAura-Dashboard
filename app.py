import numpy as np
import torch
import torch.nn as nn
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import json


app = Flask(__name__)
CORS(app)

class CropPredictionModel(nn.Module):
    def __init__(self, input_size):
        super(CropPredictionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, 32)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)
        out = self.sigmoid(out)
        return out

try:
    with open('model.pkl', 'rb') as model_file:
    
        input_size = 7 
        model = CropPredictionModel(input_size)
        model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
        model.eval()

    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
        
    print("Model and scaler loaded successfully.")

except FileNotFoundError as e:
    print(f"Error loading model or scaler: {e}")

    model = None
    scaler = None
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    model = None
    scaler = None
    

@app.route('/')
def home():
    """Renders the main dashboard page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Receives sensor data and returns a crop health prediction."""
    if not model or not scaler:
        return jsonify({'error': 'Model or scaler not loaded. Check server logs.'}), 500

    try:
        data = request.json
        print(f"Received data for prediction: {data}")
        
        features = [
            data['temperature'],
            data['humidity'],
            data['ph'],
            data['water_availability'],
            data['soil_moisture'],
            data['light_intensity'],
            data['NPK_N'],
        ]
        
        
        features_np = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features_np)
        
        
        features_tensor = torch.tensor(features_scaled, dtype=torch.float32)
        
        
        with torch.no_grad():
            prediction_proba = model(features_tensor).item()
        
        
        prediction = 1 if prediction_proba > 0.5 else 0
        
        print(f"Prediction Probability: {prediction_proba}, Prediction: {prediction}")
        
        return jsonify({
            'prediction': prediction,
            'probability': prediction_proba
        })

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'An error occurred during prediction.'}), 500

