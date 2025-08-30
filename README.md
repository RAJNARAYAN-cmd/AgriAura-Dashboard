# 🌱 AgriAura – Smart Carbon Emission Monitoring & Prediction Platform  

AgriAura is an AI-powered platform designed to help farmers and researchers **monitor, predict, and reduce carbon emissions in agriculture**.  
It integrates **IoT sensors, AI/ML (PyTorch), MQTT, SQLite, and a modern web dashboard** to provide actionable insights on sustainability.  

---

## 🚀 Features  

### 🔧 Backend (Flask + PyTorch + MQTT + SQLite)  
- **AI Model** – Trains a PyTorch neural network to predict CO₂ emissions from farm sensor data.  
- **Synthetic Data Generator** – Generates realistic environmental data for model training.  
- **REST API** – Flask endpoints to:  
  - `/predict` → Predict CO₂ levels.  
  - `/record` → Save new sensor readings.  
  - `/get_all_data` → Retrieve logged sensor data.  
  - `/reward` → Issue mock carbon credit rewards.  
  - `/plugins/*` → Manage plugin system (install, toggle, settings).  
- **MQTT Integration** – Listens to real-time sensor data from `agriaura/sensor_data`.  
- **SQLite Database** – Stores sensor logs, plugin info, and settings.  

### 🎨 Frontend (HTML + Tailwind + Chart.js)  
- **Real-Time Monitoring** – Live updates of temperature, humidity, soil moisture, sunlight, and CO₂.  
- **AI Predictor** – Input new sensor readings and get CO₂ emission predictions instantly.  
- **Data Logger** – Manually record field data into the system.  
- **Charts & Analytics** – Visualize trends with interactive Chart.js graphs.  
- **Plugin System** – Extend functionality with weather, irrigation, pest detection, and market analytics plugins.  
- **AI Reports (Demo)** – Simulated AI-powered sustainability analysis & recommendations.  


## 📂 Project Structure  

```

├── app.py # Main backend (Flask, PyTorch, MQTT, SQLite)
├── import os.py # Lightweight backend version
├── Index.html # Frontend dashboard
├── carbon_emission_model.pth # (Generated) trained model file
├── scaler.pkl # (Generated) feature scaler
├── agriaura.db # (Generated) SQLite database 


```

## 🛠️ Installation & Setup  

### 1️⃣ Clone Repository  
```bash
git clone https://github.com/your-username/agriaura.git
cd agriaura

2️⃣ Install Dependencies

pip install torch numpy scikit-learn flask flask-cors paho-mqtt

3️⃣ Run Backend

python app.py

4️⃣ Run Frontend

Open Index.html in a browser (or use VS Code Live Server extension).

🔌 API Endpoints

| Endpoint                | Method   | Description                             |
| ----------------------- | -------- | --------------------------------------- |
| `/predict`              | POST     | Predict CO₂ emissions from sensor input |
| `/record`               | POST     | Store new sensor readings               |
| `/get_all_data`         | GET      | Retrieve all recorded sensor data       |
| `/reward`               | POST     | Issue mock carbon credit rewards        |
| `/plugins`              | GET      | List available plugins                  |
| `/plugins/install/<id>` | POST     | Install & enable plugin                 |
| `/plugins/toggle/<id>`  | POST     | Enable/disable plugin                   |
| `/plugins/settings`     | GET/POST | Manage plugin system settings           |


🧩 Plugins Available

☁️ Weather Integration – Real-time forecasts

💧 Smart Irrigation – AI-powered irrigation control

🐞 Pest Detection – Computer vision pest identification

📈 Market Analytics – Commodity pricing insights