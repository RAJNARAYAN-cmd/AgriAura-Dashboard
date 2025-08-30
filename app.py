import numpy as np
import torch
import torch.nn as nn
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import sqlite3
from datetime import datetime
import os

# --- Configuration ---
MODEL_PATH = 'carbon_emission_model.pth'
SCALER_PATH = 'scaler.pkl'
DB_PATH = 'agriaura.db'
INPUT_DIM = 7

# Initialize Flask app
app = Flask(__name__, template_folder='templates', static_folder='static')
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

# --- PyTorch Model Definition ---
class CO2EmissionPredictor(nn.Module):
    def __init__(self, input_dim):
        super(CO2EmissionPredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.network(x)

# --- Load Model and Scaler ---
model, scaler = (None, None)
try:
    print("Loading existing model and scaler...")
    model = CO2EmissionPredictor(input_dim=INPUT_DIM)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    with open(SCALER_PATH, 'rb') as f: scaler = pickle.load(f)
    model.eval()
    print("Model and scaler loaded successfully.")
except Exception as e:
    print(f"Could not load model/scaler. Ensure '{MODEL_PATH}' and '{SCALER_PATH}' exist. Error: {e}")

# --- Flask API Endpoints ---

@app.route('/')
def home():
    """Renders the main dashboard page."""
    return render_template('Index.html')

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

# --- This is the route that was missing from the running code ---
@app.route('/api/data/latest', methods=['GET'])
def get_latest_data():
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        latest_reading = conn.execute("SELECT * FROM sensor_data ORDER BY timestamp DESC LIMIT 1").fetchone()
        conn.close()
        return jsonify(dict(latest_reading) if latest_reading else None)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/data', methods=['GET'])
def get_all_data():
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM sensor_data ORDER BY timestamp DESC").fetchall()
        conn.close()
        return jsonify([dict(row) for row in rows])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- Plugin API Endpoints ---
@app.route('/api/plugins', methods=['GET'])
def get_plugins():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    plugins = [dict(row) for row in conn.execute("SELECT * FROM plugins").fetchall()]
    conn.close()
    return jsonify(plugins)

@app.route('/api/plugins/toggle/<plugin_id>', methods=['POST'])
def toggle_plugin(plugin_id):
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("UPDATE plugins SET is_enabled = 1 - is_enabled WHERE id = ?", (plugin_id,))
    conn.commit()
    conn.close()
    return jsonify({'message': 'Plugin toggled.'})
    
@app.route('/api/plugins/install/<plugin_id>', methods=['POST'])
def install_plugin(plugin_id):
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("UPDATE plugins SET is_installed = 1, is_enabled = 1 WHERE id = ?", (plugin_id,))
    conn.commit()
    conn.close()
    return jsonify({'message': 'Plugin installed.'})

@app.route('/api/plugins/settings', methods=['GET', 'POST'])
def plugin_settings():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    if request.method == 'GET':
        conn.row_factory = sqlite3.Row
        settings = {row['key']: bool(row['value']) for row in conn.execute("SELECT * FROM plugin_settings").fetchall()}
        conn.close()
        return jsonify(settings)
    
    if request.method == 'POST':
        for key, value in request.json.items():
            conn.execute("UPDATE plugin_settings SET value = ? WHERE key = ?", (int(value), key))
        conn.commit()
        conn.close()
        return jsonify({'message': 'Settings updated.'})

# --- Main Execution Block ---
if __name__ == '__main__':
    init_db()
    print("Starting Flask server at http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)

