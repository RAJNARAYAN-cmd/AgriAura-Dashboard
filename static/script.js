document.addEventListener('DOMContentLoaded', () => {
    const API_BASE_URL = ''; // Relative path works for local and Vercel
    let sensorChart = null; // Variable to hold the chart instance

    // --- Main Dashboard Initialization ---
    const initializeDashboard = async () => {
        updateTime();
        setInterval(updateTime, 1000 * 30);

        await loadLatestStats();
        await loadHistoricalData();

        await loadPlugins();
        setupEventListeners();
    };

    // --- Data Fetching Functions ---
    const fetchData = async (endpoint) => {
        try {
            const response = await fetch(`${API_BASE_URL}${endpoint}`);
            if (!response.ok) throw new Error(`Network response was not ok for ${endpoint}`);
            return await response.json();
        } catch (error) {
            console.error(`Failed to fetch from ${endpoint}:`, error);
            throw error;
        }
    };
    
    const postData = async (endpoint, body) => {
        try {
            const response = await fetch(`${API_BASE_URL}${endpoint}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body)
            });
            if (!response.ok) throw new Error(`Network POST error for ${endpoint}`);
            return await response.json();
        } catch (error) {
            console.error(`Failed to POST to ${endpoint}:`, error);
            throw error;
        }
    };

    // --- UI Update Functions ---
    const updateTime = () => document.getElementById('current-time').textContent = new Date().toLocaleString();

    const loadLatestStats = async () => {
        const loader = document.getElementById('stats-loader');
        const grid = document.getElementById('stats-grid');
        try {
            const latest = await fetchData('/api/data/latest');
            loader.classList.add('hidden');
            grid.classList.remove('hidden');

            if (latest) {
                grid.innerHTML = `
                    <div class="p-4 rounded-lg border"><p class="text-sm text-gray-500">Temperature</p><p class="text-xl font-bold">${latest.temperature ?? 'N/A'}°C</p></div>
                    <div class="p-4 rounded-lg border"><p class="text-sm text-gray-500">Humidity</p><p class="text-xl font-bold">${latest.humidity ?? 'N/A'}%</p></div>
                    <div class="p-4 rounded-lg border"><p class="text-sm text-gray-500">Light</p><p class="text-xl font-bold">${latest.lightIntensity ?? 'N/A'} lux</p></div>
                    <div class="p-4 rounded-lg border"><p class="text-sm text-gray-500">CO₂ Level</p><p class="text-xl font-bold">${latest.co2 ?? 'N/A'} ppm</p></div>
                `;
            } else {
                grid.innerHTML = '<p class="text-gray-500 col-span-4 text-center py-4">No real-time data available.</p>';
            }
        } catch (error) {
            loader.classList.add('hidden');
            grid.classList.remove('hidden');
            grid.innerHTML = '<p class="text-red-500 col-span-4 text-center py-4">Error: Failed to load real-time data.</p>';
        }
    };
    
    const renderChart = (data) => {
        const chartContainer = document.getElementById('chart-container');
        if (!chartContainer) return;
        chartContainer.classList.remove('hidden');

        const ctx = document.getElementById('sensor-chart').getContext('2d');
        const labels = data.map(d => new Date(d.timestamp)).reverse();
        
        if (sensorChart) sensorChart.destroy();

        sensorChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'CO₂ (ppm)', data: data.map(d => d.co2).reverse(),
                        borderColor: '#10b981', backgroundColor: 'rgba(16, 185, 129, 0.1)',
                        yAxisID: 'y', tension: 0.3, pointRadius: 2, fill: true
                    },
                    {
                        label: 'Temperature (°C)', data: data.map(d => d.temperature).reverse(),
                        borderColor: '#ef4444', backgroundColor: 'rgba(239, 68, 68, 0.1)',
                        yAxisID: 'y1', tension: 0.3, pointRadius: 2, fill: true
                    }
                ]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                interaction: { mode: 'index', intersect: false },
                scales: {
                    x: { type: 'time', time: { unit: 'day' }, grid: { display: false } },
                    y: { type: 'linear', position: 'left', title: { display: true, text: 'CO₂ (ppm)' } },
                    y1: { type: 'linear', position: 'right', title: { display: true, text: 'Temp (°C)' }, grid: { drawOnChartArea: false } },
                }
            }
        });
    };

    const loadHistoricalData = async () => {
        const loader = document.getElementById('table-loader');
        const container = document.getElementById('table-container');
        const tableBody = document.getElementById('sensor-data-table');
        try {
            const data = await fetchData('/api/data');
            loader.classList.add('hidden');
            container.classList.remove('hidden');

            if (data && data.length > 0) {
                tableBody.innerHTML = data.slice(0, 10).map(row => `
                    <tr class="text-sm">
                        <td class="px-6 py-4">${new Date(row.timestamp).toLocaleString()}</td>
                        <td class="px-6 py-4">${row.temperature ?? 'N/A'}</td>
                        <td class="px-6 py-4">${row.humidity ?? 'N/A'}</td>
                        <td class="px-6 py-4">${row.lightIntensity ?? 'N/A'}</td>
                        <td class="px-6 py-4 font-semibold">${row.co2 ?? 'N/A'}</td>
                    </tr>
                `).join('');
                renderChart(data);
            } else {
                tableBody.innerHTML = '<tr><td colspan="5" class="text-center py-4 text-gray-500">No historical data found.</td></tr>';
            }
        } catch (error) {
            loader.classList.add('hidden');
            container.classList.remove('hidden');
            tableBody.innerHTML = '<tr><td colspan="5" class="text-center py-4 text-red-500">Error: Failed to load history.</td></tr>';
        }
    };

    const loadPlugins = async () => {
        const plugins = await fetchData('/api/plugins');
        if (!plugins) return;
        
        const installed = plugins.filter(p => p.is_installed);
        const marketplace = plugins.filter(p => !p.is_installed);

        document.getElementById('installed-tab').innerHTML = installed.length > 0 ? installed.map(p => `
            <div class="p-4 border rounded-lg flex items-center justify-between">
                <div><h3 class="font-semibold">${p.icon} ${p.name}</h3><p class="text-sm text-gray-500">${p.description}</p></div>
                <label class="switch"><input type="checkbox" class="plugin-toggle" data-id="${p.id}" ${p.is_enabled ? 'checked' : ''}><span class="slider"></span></label>
            </div>`).join('') : '<p class="text-gray-500 text-center py-4">No plugins installed.</p>';
        
        document.getElementById('marketplace-tab').innerHTML = marketplace.map(p => `
            <div class="p-4 border rounded-lg">
                <h3 class="font-semibold">${p.icon} ${p.name}</h3><p class="text-sm text-gray-500 mb-4">${p.description}</p>
                <button class="w-full bg-emerald-600 text-white font-semibold p-2 rounded-md hover:bg-emerald-700 transition install-btn" data-id="${p.id}">Install</button>
            </div>`).join('');

        const settings = await fetchData('/api/plugins/settings');
        if(settings) {
            document.getElementById('settings-tab').innerHTML = Object.entries(settings).map(([key, value]) => `
                <div class="p-4 border rounded-lg flex items-center justify-between">
                    <p class="font-semibold capitalize">${key.replace('_', ' ')}</p>
                    <label class="switch"><input type="checkbox" class="setting-toggle" data-key="${key}" ${value ? 'checked' : ''}><span class="slider"></span></label>
                </div>`).join('');
        }
    };

    // --- Event Listeners Setup ---
    const setupEventListeners = () => {
        document.getElementById('predict-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(e.target);
            const data = Object.fromEntries(formData.entries());
            for (const key in data) { data[key] = parseFloat(data[key]); }
            
            const resultEl = document.getElementById('predict-result');
            const resultContainer = document.getElementById('predict-result-container');
            resultContainer.classList.remove('hidden');
            resultEl.textContent = 'Predicting...';

            try {
                const result = await postData('/predict', data);
                if (!result) throw new Error("Empty response");
                const sustainableText = result.isSustainable ? '✅ Sustainable' : '❌ Unsustainable';
                resultEl.textContent = `Predicted CO₂: ${result.predictedCo2.toFixed(2)} ppm\nStatus: ${sustainableText}`;
            } catch (error) {
                resultEl.textContent = 'Prediction failed.';
            }
        });
        
        // --- FIX: Added event listener for the new record form ---
        document.getElementById('record-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(e.target);
            const data = Object.fromEntries(formData.entries());
             for (const key in data) { data[key] = parseFloat(data[key]); }

            const messageEl = document.getElementById('record-message');
            messageEl.textContent = 'Saving...';
            messageEl.className = 'mt-3 text-sm text-gray-500';

            try {
                // The backend API for recording data is at '/api/record'
                await postData('/api/record', data);
                messageEl.textContent = 'Record saved successfully!';
                messageEl.className = 'mt-3 text-sm text-green-600';
                e.target.reset(); // Clear the form
                
                // Refresh dashboard data to show the new entry
                await loadLatestStats();
                await loadHistoricalData();
            } catch (error) {
                messageEl.textContent = 'Failed to save record.';
                messageEl.className = 'mt-3 text-sm text-red-600';
            }
        });

        document.querySelector('.tabs').addEventListener('click', async (e) => {
            if (e.target.matches('.plugin-toggle')) {
                await postData(`/api/plugins/toggle/${e.target.dataset.id}`, {});
            }
            if (e.target.matches('.install-btn')) {
                e.target.textContent = 'Installed!';
                e.target.disabled = true;
                await postData(`/api/plugins/install/${e.target.dataset.id}`, {});
                await loadPlugins();
            }
        });

        document.getElementById('settings-tab').addEventListener('change', async (e) => {
            if (e.target.matches('.setting-toggle')) {
                const key = e.target.dataset.key;
                const value = e.target.checked;
                await postData('/api/plugins/settings', { [key]: value });
            }
        });

        document.querySelectorAll('.tab-btn').forEach(tab => {
            tab.addEventListener('click', () => {
                document.querySelectorAll('.tab-btn').forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                const target = tab.dataset.tab;
                document.querySelectorAll('.tabs-content').forEach(c => c.classList.toggle('active', c.id === `${target}-tab`));
            });
        });
    };

    initializeDashboard();
});

