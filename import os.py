

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
    """Initializes the SQLite database and creates the sensor_data table if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sensor_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            temperature REAL,
            humidity REAL,
            soilMoisture REAL,
            nitrogen REAL,
            phosphorus REAL,
            potassium REAL,
            lightIntensity REAL,
            co2 REAL
        )
    ''')
    conn.commit()
    conn.close()
    print(f"Database initialized at {DB_PATH}")

# --- PyTorch Model & Dataset Classes ---
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

# --- Data Generation & Model Training ---
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
    print("Generating synthetic data...")
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
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
    print("Training finished.")
    torch.save(model.state_dict(), MODEL_PATH)
    return model, scaler

# --- Load Model and Scaler ---
model, scaler = (None, None)
if not os.path.exists(MODEL_PATH):
    model, scaler = build_and_train_model()
else:
    print("Loading existing model and scaler...")
    model = CO2EmissionPredictor(input_dim=INPUT_DIM)
    model.load_state_dict(torch.load(MODEL_PATH))
    with open(SCALER_PATH, 'rb') as f: scaler = pickle.load(f)
if model: model.eval()

# --- MQTT Client Logic ---
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT Broker!")
        client.subscribe(MQTT_TOPIC)
    else:
        print(f"Failed to connect, return code {rc}\n")

def on_message(client, userdata, msg):
    print(f"Received message from topic `{msg.topic}`")
    try:
        payload = json.loads(msg.payload.decode())
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
            payload['nitrogen'], payload['phosphorus'], payload['potassium'],
            payload['lightIntensity']
        ]])
        scaled_input = scaler.transform(input_data)
        input_tensor = torch.tensor(scaled_input, dtype=torch.float32)
        with torch.no_grad():
            predicted_co2 = model(input_tensor).item()
        print(f"MQTT-triggered Prediction: CO2 = {predicted_co2:.2f} ppm")
    except Exception as e:
        print(f"Error processing MQTT message: {e}")

def start_mqtt_client():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_start()

# --- Flask API Endpoints ---
@app.route('/predict', methods=['POST'])
def predict():
    if not model or not scaler: return jsonify({'error': 'Model/scaler not loaded.'}), 500
    try:
        data = request.json
        input_data = np.array([[
            data['temperature'], data['humidity'], data['soilMoisture'],
            data['nitrogen'], data['phosphorus'], data['potassium'],
            data['lightIntensity']
        ]])
        scaled_input = scaler.transform(input_data)
        input_tensor = torch.tensor(scaled_input, dtype=torch.float32)
        with torch.no_grad():
            predicted_co2 = model(input_tensor).item()
        is_sustainable = bool(predicted_co2 < 150.0)
        return jsonify({'predictedCo2': float(predicted_co2), 'isSustainable': is_sustainable})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/record', methods=['POST'])
def record_data():
    try:
        data = request.json
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO sensor_data (timestamp, temperature, humidity, lightIntensity, co2)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            data.get('timestamp'), data.get('temperature'), data.get('humidity'),
            data.get('lightIntensity', data.get('sunlight')), data.get('co2')
        ))
        conn.commit()
        conn.close()
        return jsonify({'message': 'Data recorded successfully to SQLite!'})
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

if __name__ == '__main__':
    init_db()
    print("Starting MQTT client...")
    start_mqtt_client()
    print("Starting Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)

# --- HTML Separator ---
# The content below should be saved as a separate file, e.g., 'index.html'

'''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AgriAura Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns/dist/chartjs-adapter-date-fns.bundle.min.js"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Inter', sans-serif; background-color: #F0F2F5; }
        .main-header { background: #FFFFFF; border-bottom: 1px solid #E5E7EB; }
        .card { background-color: #FFFFFF; border-radius: 12px; padding: 24px; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.05), 0 2px 4px -2px rgb(0 0 0 / 0.05); }
        .stat-card { border: 1px solid #E5E7EB; }
        .chart-container { position: relative; width: 100%; height: 350px; }
        .loader { border: 4px solid #f3f4f6; border-top: 4px solid #2563eb; border-radius: 50%; width: 32px; height: 32px; animation: spin 1s linear infinite; margin: 1rem auto; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .tabs-list button.active { background-color: #ffffff; color: #1e3a8a; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
        .tabs-content { display: none; }
        .tabs-content.active { display: block; }
        .switch { position: relative; display: inline-block; width: 34px; height: 20px; }
        .switch input { opacity: 0; width: 0; height: 0; }
        .slider { position: absolute; cursor: pointer; top: 0; left: 0; right: 0; bottom: 0; background-color: #ccc; transition: .4s; border-radius: 20px; }
        .slider:before { position: absolute; content: ""; height: 16px; width: 16px; left: 2px; bottom: 2px; background-color: white; transition: .4s; border-radius: 50%; }
        input:checked + .slider { background-color: #2563eb; }
        input:checked + .slider:before { transform: translateX(14px); }
    </style>
</head>
<body class="text-gray-800">

    <header class="main-header sticky top-0 z-30">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div class="flex justify-between items-center h-16">
                <div class="flex items-center space-x-2">
                    <svg class="h-8 w-8 text-blue-600" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M3.75 13.5l10.5-11.25L12 10.5h8.25L9.75 21.75 12 13.5H3.75z" /></svg>
                    <h1 class="text-2xl font-bold text-gray-900">AgriAura</h1>
                </div>
                <div class="text-sm text-gray-500">Nagpur, Maharashtra, India</div>
            </div>
        </div>
    </header>

    <main class="max-w-7xl mx-auto p-4 sm:p-6 lg:p-8 space-y-8">
        <section id="monitoring-section" class="card">
            <h2 class="text-xl font-bold text-gray-900 mb-1">Real-time Monitoring</h2>
            <p class="text-gray-500 mb-6">Live sensor data from your fields.</p>
            <div id="stats-loader" class="text-center py-8"><div class="loader"></div><p class="text-gray-500 mt-2">Loading live data...</p></div>
            <div id="stats-grid" class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6 hidden"></div>
            <div id="chart-container" class="chart-container hidden"><canvas id="sensor-chart"></canvas></div>
        </section>

        <div class="grid grid-cols-1 lg:grid-cols-3 gap-8">
            <div class="lg:col-span-2 card">
                <h2 class="text-xl font-bold text-gray-900 mb-1">CO₂ Emission Predictor</h2>
                <p class="text-gray-500 mb-6">Use the AI model to predict CO₂ emissions based on new sensor data.</p>
                <form id="predict-form" class="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <input type="number" step="0.1" id="pred_temperature" class="p-2 border rounded-md" placeholder="Temperature (°C)" required>
                    <input type="number" step="0.1" id="pred_humidity" class="p-2 border rounded-md" placeholder="Humidity (%)" required>
                    <input type="number" step="0.1" id="pred_soilMoisture" class="p-2 border rounded-md" placeholder="Soil Moisture" required>
                    <input type="number" step="0.1" id="pred_nitrogen" class="p-2 border rounded-md" placeholder="Nitrogen (ppm)" required>
                    <input type="number" step="0.1" id="pred_phosphorus" class="p-2 border rounded-md" placeholder="Phosphorus (ppm)" required>
                    <input type="number" step="0.1" id="pred_potassium" class="p-2 border rounded-md" placeholder="Potassium (ppm)" required>
                    <input type="number" step="0.1" id="pred_lightIntensity" class="p-2 border rounded-md" placeholder="Light (lux)" required>
                    <button type="submit" class="w-full bg-blue-600 text-white font-semibold p-2 rounded-md hover:bg-blue-700 transition md:col-span-2">Predict Emissions</button>
                </form>
                <div id="predict-result-container" class="mt-4 hidden">
                    <h3 class="font-semibold text-gray-800">Prediction Result:</h3>
                    <pre id="predict-result" class="mt-2 p-3 bg-gray-50 rounded-lg text-sm"></pre>
                </div>
            </div>
            <div class="card">
                <h2 class="text-xl font-bold text-gray-900 mb-1">Record Data</h2>
                <p class="text-gray-500 mb-6">Manually log new sensor readings.</p>
                <form id="record-form" class="space-y-4">
                    <input type="number" step="0.1" id="temperature" class="w-full p-2 border rounded-md" placeholder="Temperature (°C)" required>
                    <input type="number" step="0.1" id="humidity" class="w-full p-2 border rounded-md" placeholder="Humidity (%)" required>
                    <input type="number" step="0.1" id="sunlight" class="w-full p-2 border rounded-md" placeholder="Sunlight (lux)" required>
                    <input type="number" step="0.1" id="co2" class="w-full p-2 border rounded-md" placeholder="CO₂ (ppm)" required>
                    <button type="submit" class="w-full bg-gray-800 text-white font-semibold p-2 rounded-md hover:bg-gray-900 transition">Save Record</button>
                </form>
                <div id="record-message" class="mt-3"></div>
            </div>
        </div>
        
        <section id="plugin-section" class="card">
            <div class="flex items-center justify-between mb-4">
                <div>
                    <h2 class="text-xl font-bold text-gray-900 mb-1">Plugin Management</h2>
                    <p class="text-gray-500">Extend your AgriAura dashboard with powerful plugins.</p>
                </div>
                <div id="installed-count-badge" class="text-sm font-medium bg-blue-100 text-blue-800 px-3 py-1 rounded-full"></div>
            </div>
            <div class="tabs">
                <div class="tabs-list bg-gray-100 p-1 rounded-lg grid grid-cols-3 gap-1 mb-6">
                    <button data-tab="installed" class="tab-btn px-4 py-2 text-sm font-semibold text-gray-600 rounded-md transition active">Installed</button>
                    <button data-tab="marketplace" class="tab-btn px-4 py-2 text-sm font-semibold text-gray-600 rounded-md transition">Marketplace</button>
                    <button data-tab="settings" class="tab-btn px-4 py-2 text-sm font-semibold text-gray-600 rounded-md transition">Settings</button>
                </div>
                <div id="installed-tab" class="tabs-content active space-y-4"></div>
                <div id="marketplace-tab" class="tabs-content"><div id="marketplace-grid" class="grid grid-cols-1 md:grid-cols-2 gap-4"></div></div>
                <div id="settings-tab" class="tabs-content space-y-6">
                    <div class="bg-gray-50 p-6 rounded-lg border">
                        <h3 class="font-semibold text-gray-800 mb-4">Plugin System Settings</h3>
                        <div class="space-y-4">
                            <div class="flex items-center justify-between">
                                <div><p class="text-gray-700">Auto-update plugins</p><p class="text-gray-500 text-sm">Automatically install plugin updates</p></div>
                                <label class="switch"><input type="checkbox" checked><span class="slider"></span></label>
                            </div>
                            <div class="flex items-center justify-between">
                                <div><p class="text-gray-700">Beta plugin access</p><p class="text-gray-500 text-sm">Access to experimental features</p></div>
                                <label class="switch"><input type="checkbox"><span class="slider"></span></label>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </section>

        <section id="log-section" class="card">
            <h2 class="text-xl font-bold text-gray-900 mb-1">Data Log</h2>
            <p class="text-gray-500 mb-6">A complete history of all recorded sensor data.</p>
            <div id="table-loader" class="text-center py-8"><div class="loader"></div><p class="text-gray-500 mt-2">Loading historical data...</p></div>
            <div class="overflow-x-auto hidden" id="table-container">
                <table class="min-w-full divide-y divide-gray-200">
                    <thead class="bg-gray-50">
                        <tr>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Timestamp</th>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Temp (°C)</th>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Humidity (%)</th>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Sunlight (lux)</th>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">CO₂ (ppm)</th>
                        </tr>
                    </thead>
                    <tbody id="sensor-data-table" class="bg-white divide-y divide-gray-200"></tbody>
                </table>
            </div>
        </section>
    </main>

    <script>
        // All JavaScript logic from the previous version is included here.
        // It is omitted for brevity in this display but is present in the actual artifact.
        document.addEventListener('DOMContentLoaded', () => {
            const API_BASE_URL = 'http://127.0.0.1:5000';
            let sensorChart = null;

            const statsLoader = document.getElementById('stats-loader');
            const statsGrid = document.getElementById('stats-grid');
            const chartContainer = document.getElementById('chart-container');
            const tableLoader = document.getElementById('table-loader');
            const tableContainer = document.getElementById('table-container');
            const sensorDataTable = document.getElementById('sensor-data-table');
            const predictForm = document.getElementById('predict-form');
            const predictResultContainer = document.getElementById('predict-result-container');
            const predictResultEl = document.getElementById('predict-result');
            const recordForm = document.getElementById('record-form');
            const recordMessageEl = document.getElementById('record-message');

            const showMessage = (element, message, isError = false) => {
                element.textContent = message;
                element.className = `p-2 rounded-md text-sm ${isError ? 'bg-red-100 text-red-700' : 'bg-green-100 text-green-700'}`;
                setTimeout(() => { element.textContent = ''; element.className = ''; }, 5000);
            };

            const renderStats = (latestData) => {
                if (!latestData) {
                    statsGrid.innerHTML = '<p class="text-gray-500 col-span-4">No data available to display stats.</p>';
                    return;
                }
                const stats = [
                    { label: 'Latest CO₂', value: `${latestData.co2} ppm`, icon: '<svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6 text-blue-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" /></svg>' },
                    { label: 'Temperature', value: `${latestData.temperature}°C`, icon: '<svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6 text-red-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M17.657 18.657A8 8 0 016.343 7.343S7 9 9 10c0-2 .5-5 2.986-7.014A8.003 8.003 0 0112 2a8.003 8.003 0 015.014 1.986C19.5 6 20 9 20 11c2 1 2.343 2.343 2.343 2.343a8 8 0 01-11.314 5.314z" /></svg>' },
                    { label: 'Humidity', value: `${latestData.humidity}%`, icon: '<svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6 text-green-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M9 19l3 3m0 0l3-3m-3 3V10" /></svg>' },
                    { label: 'Sunlight', value: `${latestData.sunlight || latestData.lightIntensity} lux`, icon: '<svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6 text-yellow-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" /></svg>' }
                ];
                statsGrid.innerHTML = stats.map(stat => `<div class="stat-card p-4 rounded-lg flex items-center space-x-4"><div class="p-3 bg-gray-100 rounded-full">${stat.icon}</div><div><p class="text-sm text-gray-500">${stat.label}</p><p class="text-xl font-bold text-gray-900">${stat.value}</p></div></div>`).join('');
            };

            const renderChart = (data) => {
                const ctx = document.getElementById('sensor-chart').getContext('2d');
                const labels = data.map(d => new Date(d.timestamp)).reverse();
                if (sensorChart) { sensorChart.destroy(); }
                sensorChart = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: labels,
                        datasets: [
                            { label: 'CO₂ (ppm)', data: data.map(d => d.co2).reverse(), borderColor: '#3b82f6', backgroundColor: '#bfdbfe', yAxisID: 'y', tension: 0.3, pointRadius: 2 },
                            { label: 'Temperature (°C)', data: data.map(d => d.temperature).reverse(), borderColor: '#ef4444', backgroundColor: '#fecaca', yAxisID: 'y1', tension: 0.3, pointRadius: 2 },
                        ]
                    },
                    options: {
                        responsive: true, maintainAspectRatio: false, interaction: { mode: 'index', intersect: false },
                        scales: {
                            x: { type: 'time', time: { unit: 'hour' }, grid: { display: false } },
                            y: { type: 'linear', position: 'left', title: { display: true, text: 'CO₂ (ppm)' } },
                            y1: { type: 'linear', position: 'right', title: { display: true, text: 'Temp (°C)' }, grid: { drawOnChartArea: false } },
                        }
                    }
                });
            };

            const renderTable = (data) => {
                sensorDataTable.innerHTML = '';
                if (data.length === 0) {
                    sensorDataTable.innerHTML = `<tr><td colspan="5" class="text-center py-4 text-gray-500">No data recorded yet.</td></tr>`;
                    return;
                }
                data.slice(0, 10).forEach(row => {
                    const tr = document.createElement('tr');
                    tr.innerHTML = `<td class="px-6 py-4 whitespace-nowrap text-sm text-gray-600">${new Date(row.timestamp).toLocaleString()}</td><td class="px-6 py-4 whitespace-nowrap text-sm text-gray-600">${row.temperature}</td><td class="px-6 py-4 whitespace-nowrap text-sm text-gray-600">${row.humidity}</td><td class="px-6 py-4 whitespace-nowrap text-sm text-gray-600">${row.sunlight || row.lightIntensity}</td><td class="px-6 py-4 whitespace-nowrap text-sm text-gray-600 font-semibold">${row.co2}</td>`;
                    sensorDataTable.appendChild(tr);
                });
            };

            const fetchAllSensorData = async () => {
                try {
                    const response = await fetch(`${API_BASE_URL}/get_all_data`);
                    if (!response.ok) throw new Error('Failed to fetch sensor data. Is the backend running?');
                    const data = await response.json();
                    statsLoader.classList.add('hidden');
                    statsGrid.classList.remove('hidden');
                    chartContainer.classList.remove('hidden');
                    tableLoader.classList.add('hidden');
                    tableContainer.classList.remove('hidden');
                    renderStats(data[0]);
                    renderChart(data);
                    renderTable(data);
                } catch (error) {
                    console.error('Error fetching sensor data:', error);
                    statsLoader.innerHTML = `<p class="text-red-500 p-4 bg-red-100 rounded-md">${error.message}</p>`;
                    tableLoader.innerHTML = `<p class="text-red-500 p-4 bg-red-100 rounded-md">${error.message}</p>`;
                }
            };
            
            predictForm.addEventListener('submit', async (e) => {
                e.preventDefault();
                const data = {
                    temperature: parseFloat(document.getElementById('pred_temperature').value),
                    humidity: parseFloat(document.getElementById('pred_humidity').value),
                    soilMoisture: parseFloat(document.getElementById('pred_soilMoisture').value),
                    nitrogen: parseFloat(document.getElementById('pred_nitrogen').value),
                    phosphorus: parseFloat(document.getElementById('pred_phosphorus').value),
                    potassium: parseFloat(document.getElementById('pred_potassium').value),
                    lightIntensity: parseFloat(document.getElementById('pred_lightIntensity').value)
                };
                if (Object.values(data).some(v => isNaN(v))) {
                    predictResultEl.textContent = 'Error: Please fill all fields with valid numbers.';
                    predictResultContainer.classList.remove('hidden');
                    return;
                }
                try {
                    predictResultEl.textContent = 'Getting prediction...';
                    predictResultContainer.classList.remove('hidden');
                    const response = await fetch(`${API_BASE_URL}/predict`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(data)
                    });
                    const result = await response.json();
                    predictResultEl.textContent = JSON.stringify(result, null, 2);
                } catch (error) {
                    predictResultEl.textContent = `Error: Could not connect to the backend. ${error.message}`;
                }
            });

            recordForm.addEventListener('submit', async (e) => {
                e.preventDefault();
                const data = {
                    timestamp: new Date().toISOString(),
                    temperature: parseFloat(document.getElementById('temperature').value),
                    humidity: parseFloat(document.getElementById('humidity').value),
                    sunlight: parseFloat(document.getElementById('sunlight').value),
                    co2: parseFloat(document.getElementById('co2').value)
                };
                if (Object.values(data).some(v => isNaN(v) && typeof v !== 'string')) {
                    showMessage(recordMessageEl, 'Please fill all fields with valid numbers.', true);
                    return;
                }
                try {
                    const response = await fetch(`${API_BASE_URL}/record`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(data)
                    });
                    const result = await response.json();
                    if (response.ok) {
                        showMessage(recordMessageEl, result.message, false);
                        recordForm.reset();
                        fetchAllSensorData();
                    } else {
                        showMessage(recordMessageEl, `Error: ${result.error}`, true);
                    }
                } catch (error) {
                    showMessage(recordMessageEl, `Error: Could not connect to the backend. ${error.message}`, true);
                }
            });

            const pluginIcons = {
                Cloud: '<svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M3 15a4 4 0 004 4h9a5 5 0 10-.1-9.999 5.002 5.002 0 10-9.78 2.096A4.001 4.001 0 003 15z" /></svg>',
                Droplets: '<svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M16 17L16 17C14.3431 17 13 15.6569 13 14C13 12.3431 14.3431 11 16 11C17.6569 11 19 12.3431 19 14C19 15.6569 17.6569 17 16 17ZM16 17L16 19M8 13L8 13C6.34315 13 5 11.6569 5 10C5 8.34315 6.34315 7 8 7C9.65685 7 11 8.34315 11 10C11 11.6569 9.65685 13 8 13ZM8 13L8 15M12 7L12 5" /></svg>',
                Bug: '<svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M16 8v8m-3-5v5m-3-2v2m-2 4h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" /></svg>',
                TrendingUp: '<svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" /></svg>',
                Camera: '<svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" /><path stroke-linecap="round" stroke-linejoin="round" d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" /></svg>',
                Brain: '<svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M9 8h6m-5 4h4m5 6a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>',
            };
            let plugins = [
                { id: 'weather-integration', name: 'Advanced Weather Integration', description: 'Real-time weather forecasting with 7-day predictions.', icon: pluginIcons.Cloud, category: 'Environmental', isInstalled: true, isEnabled: true, rating: 4.8, downloads: '12.5K', version: '2.1.0', status: 'stable' },
                { id: 'smart-irrigation', name: 'Smart Irrigation Control', description: 'AI-powered irrigation scheduling based on soil moisture.', icon: pluginIcons.Droplets, category: 'Automation', isInstalled: true, isEnabled: false, rating: 4.9, downloads: '8.3K', version: '1.5.2', status: 'stable' },
                { id: 'pest-detection', name: 'AI Pest Detection', description: 'Computer vision-based pest and disease identification.', icon: pluginIcons.Bug, category: 'Monitoring', isInstalled: false, isEnabled: false, rating: 4.6, downloads: '6.7K', version: '3.0.1', status: 'beta' },
                { id: 'market-analytics', name: 'Market Price Analytics', description: 'Real-time commodity pricing and market trend analysis.', icon: pluginIcons.TrendingUp, category: 'Analytics', isInstalled: false, isEnabled: false, rating: 4.4, downloads: '4.2K', version: '1.8.0', status: 'stable' },
                { id: 'drone-monitoring', name: 'Drone Field Monitoring', description: 'Integrate drone data for aerial crop monitoring.', icon: pluginIcons.Camera, category: 'Monitoring', isInstalled: false, isEnabled: false, rating: 4.7, downloads: '3.1K', version: '2.3.0', status: 'stable' },
                { id: 'predictive-ai', name: 'Predictive AI Assistant', description: 'Machine learning models for yield prediction.', icon: pluginIcons.Brain, category: 'AI', isInstalled: false, isEnabled: false, rating: 4.3, downloads: '2.8K', version: '1.2.0', status: 'experimental' }
            ];
            const getStatusBadge = (status) => {
                switch (status) {
                    case 'stable': return `<span class="text-xs font-medium bg-green-100 text-green-800 px-2 py-1 rounded-full">Stable</span>`;
                    case 'beta': return `<span class="text-xs font-medium bg-yellow-100 text-yellow-800 px-2 py-1 rounded-full">Beta</span>`;
                    case 'experimental': return `<span class="text-xs font-medium bg-purple-100 text-purple-800 px-2 py-1 rounded-full">Experimental</span>`;
                    default: return '';
                }
            };
            const renderPlugins = () => {
                const installedContainer = document.getElementById('installed-tab');
                const marketplaceContainer = document.getElementById('marketplace-grid');
                const installedCountBadge = document.getElementById('installed-count-badge');
                const installedPlugins = plugins.filter(p => p.isInstalled);
                const availableForInstall = plugins.filter(p => !p.isInstalled);
                installedCountBadge.textContent = `${installedPlugins.length} Installed`;
                if (installedPlugins.length === 0) {
                    installedContainer.innerHTML = `<div class="text-center py-12"><p class="text-gray-500">No plugins installed yet.</p></div>`;
                } else {
                    installedContainer.innerHTML = installedPlugins.map(plugin => `<div class="p-4 border rounded-lg flex items-center justify-between"><div class="flex items-center gap-4"><div class="p-3 bg-gray-100 rounded-lg text-blue-600">${plugin.icon}</div><div><h3 class="font-semibold text-gray-800">${plugin.name}</h3><p class="text-sm text-gray-500">${plugin.description}</p></div></div><div class="flex items-center gap-4"><span class="text-sm text-gray-600">${plugin.isEnabled ? 'Enabled' : 'Disabled'}</span><label class="switch"><input type="checkbox" ${plugin.isEnabled ? 'checked' : ''} data-plugin-id="${plugin.id}" class="plugin-toggle"><span class="slider"></span></label></div></div>`).join('');
                }
                marketplaceContainer.innerHTML = availableForInstall.map(plugin => `<div class="p-4 border rounded-lg flex flex-col justify-between"><div><div class="flex items-start gap-4 mb-3"><div class="p-3 bg-gray-100 rounded-lg text-blue-600">${plugin.icon}</div><div><h3 class="font-semibold text-gray-800">${plugin.name}</h3><p class="text-sm text-gray-500 mb-2">${plugin.description}</p><div class="flex items-center gap-2">${getStatusBadge(plugin.status)}<span class="text-xs text-gray-500">v${plugin.version}</span></div></div></div><div class="flex items-center gap-4 text-sm text-gray-500 mb-4"><span>⭐ ${plugin.rating}</span><span>📦 ${plugin.downloads}</span><span>${plugin.category}</span></div></div><button data-plugin-id="${plugin.id}" class="install-btn w-full bg-blue-600 text-white font-semibold p-2 rounded-md hover:bg-blue-700 transition">Install</button></div>`).join('');
                addPluginEventListeners();
            };
            const addPluginEventListeners = () => {
                document.querySelectorAll('.plugin-toggle').forEach(toggle => {
                    toggle.addEventListener('change', (e) => {
                        const pluginId = e.target.dataset.pluginId;
                        plugins = plugins.map(p => p.id === pluginId ? { ...p, isEnabled: !p.isEnabled } : p);
                        renderPlugins();
                    });
                });
                document.querySelectorAll('.install-btn').forEach(button => {
                    button.addEventListener('click', (e) => {
                        const pluginId = e.target.dataset.pluginId;
                        plugins = plugins.map(p => p.id === pluginId ? { ...p, isInstalled: true, isEnabled: true } : p);
                        renderPlugins();
                    });
                });
            };
            const tabs = document.querySelectorAll('.tab-btn');
            const tabContents = document.querySelectorAll('.tabs-content');
            tabs.forEach(tab => {
                tab.addEventListener('click', () => {
                    tabs.forEach(t => t.classList.remove('active'));
                    tab.classList.add('active');
                    const target = tab.getAttribute('data-tab');
                    tabContents.forEach(content => {
                        content.classList.remove('active');
                        if (content.id === `${target}-tab`) {
                            content.classList.add('active');
                        }
                    });
                });
            });
            fetchAllSensorData();
            renderPlugins();
        });
    </script>
</body>
</html>
'''
