import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import json
import threading
import paho.mqtt.client as mqtt
import sqlite3
from datetime import datetime

# --- Configuration ---
MODEL_PATH = 'carbon_emission_model.pth'
SCALER_PATH = 'scaler.pkl'
DB_PATH = 'agriaura.db'
INPUT_DIM = 7

# --- MQTT Configuration ---
MQTT_BROKER = 'broker.hivemq.com'
MQTT_PORT = 1883
MQTT_TOPIC = 'agriaura/sensor_data'

app = Flask(__name__)
CORS(app) 

# --- Database Initialization ---
def init_db():
    """Initializes the SQLite database and all required tables."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cursor = conn.cursor()
    
    # Sensor Data Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sensor_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL, temperature REAL, humidity REAL, soilMoisture REAL,
            nitrogen REAL, phosphorus REAL, potassium REAL, lightIntensity REAL, co2 REAL
        )
    ''')

    # Plugins Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS plugins (
            id TEXT PRIMARY KEY, name TEXT, description TEXT, icon TEXT,
            is_installed INTEGER, is_enabled INTEGER
        )
    ''')

    # Plugin Settings Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS plugin_settings (
            key TEXT PRIMARY KEY, value INTEGER
        )
    ''')

    # --- Seed Initial Data if tables are empty ---
    cursor.execute("SELECT COUNT(*) FROM plugins")
    if cursor.fetchone()[0] == 0:
        initial_plugins = [
            ('weather-integration', 'Advanced Weather Integration', 'Real-time weather forecasting.', '☁️', 1, 1),
            ('smart-irrigation', 'Smart Irrigation Control', 'AI-powered irrigation scheduling.', '💧', 1, 0),
            ('pest-detection', 'AI Pest Detection', 'Computer vision pest identification.', '🐞', 0, 0),
            ('market-analytics', 'Market Price Analytics', 'Real-time commodity pricing.', '📈', 0, 0)
        ]
        cursor.executemany("INSERT INTO plugins VALUES (?, ?, ?, ?, ?, ?)", initial_plugins)

    cursor.execute("SELECT COUNT(*) FROM plugin_settings")
    if cursor.fetchone()[0] == 0:
        initial_settings = [('auto_update', 1), ('beta_access', 0), ('telemetry', 1)]
        cursor.executemany("INSERT INTO plugin_settings VALUES (?, ?)", initial_settings)

    conn.commit()
    conn.close()
    print(f"Database initialized successfully at {DB_PATH}")


# --- PyTorch Model & Dataset Classes (No changes) ---
class FarmSensorDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        x, y = self.data[idx], self.targets[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).unsqueeze(0)

class CO2EmissionPredictor(nn.Module):
    def __init__(self, input_dim):
        super(CO2EmissionPredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.network(x)

# --- Data Generation & Model Training (No changes) ---
def generate_synthetic_data(num_samples=2000):
    np.random.seed(42)
    temperature = np.random.uniform(20, 35, num_samples)
    humidity = np.random.uniform(40, 90, num_samples)
    soilMoisture = np.random.uniform(200, 800, num_samples)
    nitrogen = np.random.uniform(50, 150, num_samples)
    phosphorus = np.random.uniform(20, 80, num_samples)
    potassium = np.random.uniform(50, 200, num_samples)
    lightIntensity = np.random.uniform(1000, 10000, num_samples)
    co2 = (100 + 0.5 * temperature + 0.3 * humidity - 0.1 * nitrogen + 0.05 * soilMoisture + 
           0.02 * potassium - 0.01 * phosphorus + 0.001 * lightIntensity + np.random.normal(0, 10, num_samples))
    features = np.vstack([temperature, humidity, soilMoisture, nitrogen, phosphorus, potassium, lightIntensity]).T
    return features, co2

def build_and_train_model():
    print("Generating synthetic data for training...")
    features, co2_labels = generate_synthetic_data()
    X_train, _, y_train, _ = train_test_split(features, co2_labels, test_size=0.2, random_state=42)
    
    scaler = StandardScaler().fit(X_train)
    with open(SCALER_PATH, 'wb') as f: pickle.dump(scaler, f)
        
    X_train_scaled = scaler.transform(X_train)
    train_dataset = FarmSensorDataset(X_train_scaled, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    model = CO2EmissionPredictor(input_dim=INPUT_DIM)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Training the PyTorch model...")
    epochs = 50
    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
    print("Model training finished.")
    torch.save(model.state_dict(), MODEL_PATH)
    return model, scaler

# --- Load Model and Scaler (No changes) ---
model, scaler = (None, None)
if not os.path.exists(MODEL_PATH):
    model, scaler = build_and_train_model()
else:
    print("Loading existing model and scaler...")
    model = CO2EmissionPredictor(input_dim=INPUT_DIM)
    model.load_state_dict(torch.load(MODEL_PATH))
    with open(SCALER_PATH, 'rb') as f: scaler = pickle.load(f)
if model: model.eval()

# --- MQTT Client Logic (No changes) ---
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT Broker!")
        client.subscribe(MQTT_TOPIC)
    else:
        print(f"Failed to connect to MQTT, return code {rc}\n")

def on_message(client, userdata, msg):
    print(f"Received message from topic `{msg.topic}`")
    try:
        payload = json.loads(msg.payload.decode())
        if 'timestamp' not in payload: payload['timestamp'] = datetime.now().isoformat()
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO sensor_data (timestamp, temperature, humidity, soilMoisture, nitrogen, phosphorus, potassium, lightIntensity, co2)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            payload.get('timestamp'), payload.get('temperature'), payload.get('humidity'),
            payload.get('soilMoisture'), payload.get('nitrogen'), payload.get('phosphorus'),
            payload.get('potassium'), payload.get('lightIntensity'), payload.get('co2')
        ))
        conn.commit()
        conn.close()
        print("MQTT Data Logged to SQLite:", payload)
        input_data = np.array([[
            payload['temperature'], payload['humidity'], payload['soilMoisture'],
            payload['nitrogen'], payload['phosphorus'], payload['potassium'], payload['lightIntensity']
        ]])
        scaled_input = scaler.transform(input_data)
        input_tensor = torch.tensor(scaled_input, dtype=torch.float32)
        with torch.no_grad():
            predicted_co2 = model(input_tensor).item()
        print(f"MQTT-triggered Prediction: CO2 = {predicted_co2:.2f} ppm")
    except Exception as e:
        print(f"Error processing MQTT message: {e}")

def start_mqtt_client():
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
    client.on_connect = on_connect
    client.on_message = on_message
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_start()
    except Exception as e:
        print(f"Could not connect to MQTT broker: {e}")

# --- Flask API Endpoints ---
@app.route('/predict', methods=['POST'])
def predict():
    if not model or not scaler: return jsonify({'error': 'Model or scaler is not loaded.'}), 500
    try:
        data = request.json
        input_data = np.array([[
            data['temperature'], data['humidity'], data['soilMoisture'],
            data['nitrogen'], data['phosphorus'], data['potassium'], data['lightIntensity']
        ]])
        scaled_input = scaler.transform(input_data)
        input_tensor = torch.tensor(scaled_input, dtype=torch.float32)
        with torch.no_grad():
            predicted_co2 = model(input_tensor).item()
        is_sustainable = bool(predicted_co2 < 400.0)
        return jsonify({'predictedCo2': float(predicted_co2), 'isSustainable': is_sustainable})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/record', methods=['POST'])
def record_data():
    try:
        data = request.json
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute('INSERT INTO sensor_data (timestamp, temperature, humidity, lightIntensity, co2) VALUES (?, ?, ?, ?, ?)',
            (data.get('timestamp'), data.get('temperature'), data.get('humidity'), data.get('sunlight'), data.get('co2')))
        conn.commit()
        conn.close()
        return jsonify({'message': 'Data recorded successfully!'})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/get_all_data', methods=['GET'])
def get_all_data():
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM sensor_data ORDER BY timestamp DESC")
        rows = cursor.fetchall()
        conn.close()
        data = [dict(row) for row in rows]
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/reward', methods=['POST'])
def issue_reward():
    # Mock endpoint
    try:
        data = request.json
        user_id, co2_level = data.get('user_id'), data.get('co2_level')
        if not user_id or co2_level is None: return jsonify({'error': 'user_id and co2_level are required.'}), 400
        if float(co2_level) < 400:
            return jsonify({'status': 'success', 'transactionId': f'txn_{np.random.randint(10000, 99999)}', 'message': f'Credits issued to {user_id}.'})
        else:
            return jsonify({'status': 'failed', 'message': f'CO2 level {co2_level} is too high.'})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# --- NEW PLUGIN API ENDPOINTS ---
@app.route('/plugins', methods=['GET'])
def get_plugins():
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM plugins")
        rows = cursor.fetchall()
        conn.close()
        plugins = [dict(row) for row in rows]
        return jsonify(plugins)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/plugins/install/<plugin_id>', methods=['POST'])
def install_plugin(plugin_id):
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        cursor = conn.cursor()
        # When installing, set is_installed and is_enabled to true (1)
        cursor.execute("UPDATE plugins SET is_installed = 1, is_enabled = 1 WHERE id = ?", (plugin_id,))
        conn.commit()
        conn.close()
        return jsonify({'message': f'Plugin {plugin_id} installed successfully.'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/plugins/toggle/<plugin_id>', methods=['POST'])
def toggle_plugin(plugin_id):
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        cursor = conn.cursor()
        # Flip the is_enabled status
        cursor.execute("UPDATE plugins SET is_enabled = 1 - is_enabled WHERE id = ?", (plugin_id,))
        conn.commit()
        conn.close()
        return jsonify({'message': f'Plugin {plugin_id} toggled successfully.'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/plugins/settings', methods=['GET', 'POST'])
def plugin_settings():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    if request.method == 'GET':
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM plugin_settings")
            rows = cursor.fetchall()
            settings = {row['key']: bool(row['value']) for row in rows}
            conn.close()
            return jsonify(settings)
        except Exception as e:
            conn.close()
            return jsonify({'error': str(e)}), 500
    
    if request.method == 'POST':
        try:
            settings_data = request.json
            cursor = conn.cursor()
            for key, value in settings_data.items():
                cursor.execute("UPDATE plugin_settings SET value = ? WHERE key = ?", (int(value), key))
            conn.commit()
            conn.close()
            return jsonify({'message': 'Settings updated successfully.'})
        except Exception as e:
            conn.close()
            return jsonify({'error': str(e)}), 500

# --- Main Execution Block ---
if __name__ == '__main__':
    init_db()
    print("Starting MQTT client...")
    start_mqtt_client()
    print("Starting Flask server at http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
