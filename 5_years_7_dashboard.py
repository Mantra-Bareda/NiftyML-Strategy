from flask import Flask, render_template_string, jsonify, request
import json
import os
import threading
import subprocess
from datetime import datetime, timedelta

app = Flask(__name__)

TASK_STATUS = {
    "strategy": "Idle",
    "global": "Idle",
    "data": "Idle"
}

PREDICT_STATUS = {
    "status": "Idle",
    "target": ""
}

def run_notebook_pipeline(notebook_name, task_key):
    global TASK_STATUS
    TASK_STATUS[task_key] = "Running... ⏳"
    try:
        subprocess.run(["jupyter", "nbconvert", "--execute", "--inplace", notebook_name], check=True)
        TASK_STATUS[task_key] = "Completed ✅"
    except Exception as e:
        TASK_STATUS[task_key] = f"Error ❌"

def run_data_job():
    global TASK_STATUS
    TASK_STATUS["data"] = "Running... ⏳ (Downloading & Processing)"
    try:
        subprocess.run(["python3", "5_years_6.0_data_update_job.py"], check=True)
        TASK_STATUS["data"] = "Completed ✅"
    except Exception as e:
        TASK_STATUS["data"] = f"Error ❌"

def run_live_prediction(symbol):
    global PREDICT_STATUS
    PREDICT_STATUS["status"] = "Running... ⏳ (Downloading live data & generating features)"
    PREDICT_STATUS["target"] = symbol
    try:
        subprocess.run(["python3", "live_predictor_job.py", symbol], check=True)
        PREDICT_STATUS["status"] = "Completed ✅"
    except Exception as e:
        PREDICT_STATUS["status"] = f"Error ❌"


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Factory Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }</style>
</head>
<body class="bg-gray-50 min-h-screen">

    <nav class="bg-white border-b border-gray-200 shadow-sm">
        <div class="max-w-7xl mx-auto px-8">
            <div class="flex justify-between h-16">
                <div class="flex space-x-8">
                    <div class="flex-shrink-0 flex items-center">
                        <span class="text-xl font-black text-indigo-600">📈 AI Factory</span>
                    </div>
                    <a href="/" class="inline-flex items-center px-1 pt-1 border-b-2 {% if page == 'dashboard' %}border-indigo-500 text-gray-900{% else %}border-transparent text-gray-500 hover:text-gray-700{% endif %} text-sm font-medium transition">📊 Dashboard</a>
                    <a href="/data" class="inline-flex items-center px-1 pt-1 border-b-2 {% if page == 'data' %}border-indigo-500 text-gray-900{% else %}border-transparent text-gray-500 hover:text-gray-700{% endif %} text-sm font-medium transition">💾 Data Manager</a>
                    <a href="/update" class="inline-flex items-center px-1 pt-1 border-b-2 {% if page == 'update' %}border-indigo-500 text-gray-900{% else %}border-transparent text-gray-500 hover:text-gray-700{% endif %} text-sm font-medium transition">⚙️ Update Models</a>
                    <a href="/predict" class="inline-flex items-center px-1 pt-1 border-b-2 {% if page == 'predict' %}border-indigo-500 text-gray-900{% else %}border-transparent text-gray-500 hover:text-gray-700{% endif %} text-sm font-medium transition">🔮 Live Predictions</a>
                </div>
            </div>
        </div>
    </nav>

    <div class="max-w-7xl mx-auto p-8">

        {% if page == 'dashboard' %}
        <!-- ==================== DASHBOARD PAGE ==================== -->
        <div class="flex justify-between items-end mb-8 border-b border-gray-200 pb-4">
            <div>
                <h1 class="text-3xl font-extrabold text-gray-900">AI Model Performance</h1>
            </div>
            <button onclick="location.reload()" class="bg-white hover:bg-gray-50 text-sm font-semibold py-2 px-4 border border-gray-300 rounded shadow-sm">🔄 Refresh Data</button>
        </div>
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {% for model_name, data in models.items() %}
            <div class="bg-white rounded-xl shadow-sm border border-gray-200 p-6 flex flex-col hover:shadow-md transition">
                <div class="flex justify-between items-start mb-4">
                    <h2 class="text-lg font-bold text-gray-800">{{ model_name.replace('_', ' ') }}</h2>
                </div>
                <p class="text-xs text-gray-400 mb-5 font-mono">🗓️ Trained: {{ data.date_trained }}</p>
                <div class="mb-5 flex-grow">
                    <div class="grid grid-cols-2 gap-3">
                        {% for metric_name, metric_value in data.metrics.items() %}
                        <div class="bg-gray-50 rounded-lg p-3 border border-gray-100">
                            <p class="text-[10px] text-gray-500 font-bold uppercase tracking-wider mb-1">{{ metric_name.replace('_', ' ') }}</p>
                            <p class="text-lg font-bold text-indigo-600">{{ metric_value }}</p>
                        </div>
                        {% endfor %}
                    </div>
                </div>
            </div>
            {% endfor %}
        </div>
        


        </div>
        
        <!-- Individual Models Table -->
        <div class="mt-12 mb-8 border-b border-gray-200 pb-4">
            <h1 class="text-2xl font-extrabold text-gray-900">Individual Stock Models (400 Models)</h1>
            <p class="text-gray-500 mt-2 text-sm">Performance of the 200 custom Strategy and 200 custom Global AI models.</p>
        </div>
        
        <div class="bg-white rounded-xl shadow-sm border border-gray-200 p-6 mb-8">
            <div class="mb-4">
                <input type="text" id="model-search" onkeyup="filterModels()" placeholder="Search Symbol (e.g. RELIANCE)..." class="w-full border-gray-300 rounded-md shadow-sm border p-3 focus:ring-indigo-500 text-lg">
            </div>
            <div class="overflow-y-auto" style="max-height: 500px;">
                <table class="min-w-full divide-y divide-gray-200" id="models-table">
                    <thead class="bg-gray-50 sticky top-0">
                        <tr>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Symbol</th>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Strategy Accuracy</th>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Strategy MAE</th>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Global Gain MAE</th>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Global Drop MAE</th>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Global Dir Acc</th>
                        </tr>
                    </thead>
                    <tbody class="bg-white divide-y divide-gray-200">
                        {% for sym in ind_strat.keys() | list + ind_glob.keys() | list | unique %}
                        <tr class="hover:bg-gray-50 model-row">
                            <td class="px-6 py-4 whitespace-nowrap font-bold text-indigo-600 symbol-cell">{{ sym }}</td>
                            <td class="px-6 py-4 whitespace-nowrap">{{ ind_strat.get(sym, {}).get('Accuracy', 'N/A') }}</td>
                            <td class="px-6 py-4 whitespace-nowrap">{{ ind_strat.get(sym, {}).get('MAE', 'N/A') }}</td>
                            <td class="px-6 py-4 whitespace-nowrap">{{ ind_glob.get(sym, {}).get('MAE_Gain', 'N/A') }}</td>
                            <td class="px-6 py-4 whitespace-nowrap">{{ ind_glob.get(sym, {}).get('MAE_Drop', 'N/A') }}</td>
                            <td class="px-6 py-4 whitespace-nowrap">{{ ind_glob.get(sym, {}).get('Directional_Accuracy', 'N/A') }}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
        </div>
        
        <script>
        function filterModels() {
            var input, filter, table, tr, td, i, txtValue;
            input = document.getElementById("model-search");
            filter = input.value.toUpperCase();
            table = document.getElementById("models-table");
            tr = table.getElementsByClassName("model-row");
            for (i = 0; i < tr.length; i++) {
                td = tr[i].getElementsByClassName("symbol-cell")[0];
                if (td) {
                    txtValue = td.textContent || td.innerText;
                    if (txtValue.toUpperCase().indexOf(filter) > -1) {
                        tr[i].style.display = "";
                    } else {
                        tr[i].style.display = "none";
                    }
                }       
            }
        }
        </script>

        {% elif page == 'data' %}

        <!-- ==================== DATA MANAGER PAGE ==================== -->
        <div class="mb-8 border-b border-gray-200 pb-4">
            <h1 class="text-3xl font-extrabold text-gray-900">Data Manager</h1>
            <p class="text-gray-500 mt-2 text-sm">Manage the historical dataset that your models use for training.</p>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-8">
            <!-- Data Status Card -->
            <div class="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
                <h2 class="text-xl font-bold text-gray-900 mb-4">Dataset Status</h2>
                <div class="space-y-4">
                    <div class="bg-indigo-50 border border-indigo-100 rounded p-4 flex justify-between items-center">
                        <span class="text-sm font-bold text-indigo-900 uppercase tracking-wider">Last Download Date</span>
                        <span class="text-lg font-black text-indigo-600">{{ last_updated_date }}</span>
                    </div>
                    <div class="flex justify-between items-center border-b border-gray-100 py-2">
                        <span class="text-gray-600 text-sm">Final Dataset Size</span>
                        <span class="font-bold text-gray-800">{{ final_csv_size }}</span>
                    </div>
                    <div class="flex justify-between items-center py-2">
                        <span class="text-gray-600 text-sm">Raw CSVs Tracked</span>
                        <span class="font-bold text-gray-800">{{ total_stocks }} Stocks</span>
                    </div>
                </div>
            </div>

            <!-- Manual Data Download Card -->
            <div class="bg-white rounded-xl shadow-sm border border-gray-200 p-6 flex flex-col">
                <h2 class="text-xl font-bold text-gray-900 mb-2">Update Dataset Manually</h2>
                <p class="text-sm text-gray-500 mb-6">Downloads missing days up to yesterday, generates all calculated columns, and cleans the final dataset. <b>This does NOT retrain the AI models.</b></p>
                
                <div class="bg-gray-50 rounded p-4 mb-6">
                    <span class="text-xs font-bold text-gray-500 uppercase tracking-wider block mb-1">Current Job Status</span>
                    <span id="status-data" class="text-sm font-medium text-gray-900">{{ status.data }}</span>
                </div>
                
                <div class="mt-auto">
                    <button id="btn-data" onclick="startDataUpdate()" class="w-full bg-emerald-600 hover:bg-emerald-700 text-white font-bold py-3 px-4 rounded transition disabled:opacity-50">
                        📥 Download & Process Missing Data
                    </button>
                </div>
            </div>
        </div>
        <script>
            function startDataUpdate() {
                document.getElementById('btn-data').disabled = true;
                fetch('/api/start_data', { method: 'POST' }).then(r => pollDataStatus());
            }
            function pollDataStatus() {
                fetch('/api/status').then(r => r.json()).then(s => {
                    document.getElementById('status-data').innerText = s.data;
                    if(s.data.includes('Running')) {
                        document.getElementById('btn-data').disabled = true;
                        setTimeout(pollDataStatus, 2000);
                    } else if(s.data.includes('Completed')) {
                        location.reload();
                    }
                });
            }
            fetch('/api/status').then(r => r.json()).then(s => { if(s.data.includes('Running')) pollDataStatus(); });
        </script>


        {% elif page == 'update' %}
        <!-- ==================== UPDATE MODELS PAGE ==================== -->
        <div class="mb-8 border-b border-gray-200 pb-4">
            <h1 class="text-3xl font-extrabold text-gray-900">Update Models</h1>
            <p class="text-gray-500 mt-2 text-sm">Retrain your AI models automatically. If the dataset is not up to date, it will automatically download missing data first.</p>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div class="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
                <h2 class="text-xl font-bold text-gray-900 mb-4">Strategy Models</h2>
                <div class="bg-gray-50 rounded p-4 mb-6">
                    <span id="status-strategy" class="text-sm font-medium text-gray-900">{{ status.strategy }}</span>
                </div>
                <button id="btn-strategy" onclick="startUpdate('strategy')" class="w-full bg-indigo-600 text-white font-bold py-3 px-4 rounded">Run Strategy Update</button>
            </div>
            <div class="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
                <h2 class="text-xl font-bold text-gray-900 mb-4">Global Models</h2>
                <div class="bg-gray-50 rounded p-4 mb-6">
                    <span id="status-global" class="text-sm font-medium text-gray-900">{{ status.global }}</span>
                </div>
                <button id="btn-global" onclick="startUpdate('global')" class="w-full bg-blue-600 text-white font-bold py-3 px-4 rounded">Run Global Update</button>
            </div>
        </div>
        <script>
            function startUpdate(type) {
                document.getElementById('btn-' + type).disabled = true;
                fetch('/api/start_update/' + type, { method: 'POST' }).then(r => pollStatus(type));
            }
            function pollStatus(type) {
                fetch('/api/status').then(r => r.json()).then(s => {
                    document.getElementById('status-strategy').innerText = s.strategy;
                    document.getElementById('status-global').innerText = s.global;
                    if(s[type].includes('Running')) {
                        setTimeout(() => pollStatus(type), 2000);
                    } else if(s[type].includes('Completed')) {
                        document.getElementById('btn-' + type).disabled = false;
                    }
                });
            }
            fetch('/api/status').then(r => r.json()).then(s => { 
                if(s.strategy.includes('Running')) pollStatus('strategy');
                if(s.global.includes('Running')) pollStatus('global');
            });
        </script>


        {% elif page == 'predict' %}
        <!-- ==================== PREDICTIONS PAGE ==================== -->
        <div class="flex justify-between items-end mb-8 border-b border-gray-200 pb-4">
            <div>
                <h1 class="text-3xl font-extrabold text-gray-900">Live Predictions</h1>
                <p class="text-gray-500 mt-2 text-sm">Predict today's movement using live market data (Valid for 24h).</p>
            </div>
            <button id="btn-predict-all" onclick="startPrediction('ALL')" class="bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-2 px-6 rounded shadow transition">🚀 Predict All (200 Stocks)</button>
        </div>

        <div class="bg-white rounded-xl shadow-sm border border-gray-200 p-6 mb-8 flex items-center space-x-4">
            <div class="flex-grow">
                <label class="block text-sm font-medium text-gray-700 mb-1">Search & Predict Single Stock</label>
                <input type="text" id="symbol-search" placeholder="e.g. RELIANCE.NS, TCS.NS" class="w-full border-gray-300 rounded-md shadow-sm border p-2 focus:ring-indigo-500">
            </div>
            <div class="pt-6">
                <button id="btn-predict-single" onclick="startPrediction(document.getElementById('symbol-search').value)" class="bg-gray-800 hover:bg-gray-900 text-white font-bold py-2 px-6 rounded shadow transition">Predict Stock</button>
            </div>
        </div>

        <div id="loading-banner" class="hidden bg-blue-50 border border-blue-200 rounded-lg p-4 mb-8 flex items-center space-x-4 animate-pulse">
            <div class="text-blue-500 text-2xl">⏳</div>
            <div>
                <h3 id="predict-status-text" class="text-blue-800 font-bold">Predicting...</h3>
                <p class="text-blue-600 text-sm">Downloading live market data and running AI models...</p>
            </div>
        </div>

        {% if predictions %}
        <div class="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
            <div class="bg-gray-50 px-6 py-3 border-b border-gray-200 flex justify-between items-center">
                <h3 class="font-bold text-gray-700">Latest Predictions</h3>
                <span class="text-xs text-gray-500 font-mono">Calculated at: {{ timestamp }}</span>
            </div>
            <table class="min-w-full divide-y divide-gray-200">
                <thead class="bg-gray-50">
                    <tr>
                        <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Symbol</th>
                        <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Strategy Target</th>
                        <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Predicted High (Gain)</th>
                        <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Predicted Low (Drop)</th>
                        <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Direction</th>
                    </tr>
                </thead>
                <tbody class="bg-white divide-y divide-gray-200">
                    {% for sym, pred in predictions.items() %}
                    <tr class="hover:bg-gray-50">
                        <td class="px-6 py-4 whitespace-nowrap font-bold text-indigo-600">{{ sym }}</td>
                        <td class="px-6 py-4 whitespace-nowrap">
                            {% if pred.strategy_triggered %}
                                {% if pred.predicted_r >= 1.0 %}
                                    <span class="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-green-100 text-green-800">Buy (+{{ pred.predicted_r }} R)</span>
                                {% else %}
                                    <span class="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-red-100 text-red-800">Skip (+{{ pred.predicted_r }} R)</span>
                                {% endif %}
                            {% else %}
                                <span class="text-gray-400 text-sm">No Signal Today</span>
                            {% endif %}
                        </td>
                        <td class="px-6 py-4 whitespace-nowrap font-medium text-green-600">+{{ pred.predicted_gain_pct }}%</td>
                        <td class="px-6 py-4 whitespace-nowrap font-medium text-red-600">{{ pred.predicted_drop_pct }}%</td>
                        <td class="px-6 py-4 whitespace-nowrap">
                            {% if pred.predicted_gain_pct > (pred.predicted_drop_pct * -1) %}
                                <span class="text-green-500 font-bold">↑ UP</span>
                            {% else %}
                                <span class="text-red-500 font-bold">↓ DOWN</span>
                            {% endif %}
                        </td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        {% else %}
        <div class="text-center py-12 bg-white rounded-xl border border-gray-200">
            <p class="text-gray-500 mb-2">No active predictions found for today.</p>
        </div>
        {% endif %}

        <script>
            function startPrediction(symbol) {
                if(!symbol) return;
                fetch('/api/start_predict/' + symbol, { method: 'POST' }).then(r => pollPredictStatus());
            }
            function pollPredictStatus() {
                fetch('/api/predict_status').then(r => r.json()).then(s => {
                    if (s.status.includes('Running')) {
                        document.getElementById('loading-banner').classList.remove('hidden');
                        document.getElementById('predict-status-text').innerText = s.status + " for " + s.target;
                        setTimeout(pollPredictStatus, 2000);
                    } else if (s.status.includes('Completed')) {
                        location.reload();
                    }
                });
            }
            fetch('/api/predict_status').then(r => r.json()).then(s => { if(s.status.includes('Running')) pollPredictStatus(); });
        </script>
        {% endif %}

    </div>
</body>
</html>
"""


@app.route('/')
def dashboard():
    log_file = 'models/model_performance_log.json'
    log_data = {}
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            try: log_data = json.load(f)
            except: pass
            
    ind_strat_file = 'models/individual_strategy_log.json'
    ind_strat_data = {}
    if os.path.exists(ind_strat_file):
        with open(ind_strat_file, 'r') as f:
            try: ind_strat_data = json.load(f)
            except: pass
            
    ind_glob_file = 'models/individual_global_log.json'
    ind_glob_data = {}
    if os.path.exists(ind_glob_file):
        with open(ind_glob_file, 'r') as f:
            try: ind_glob_data = json.load(f)
            except: pass
            
    return render_template_string(HTML_TEMPLATE, page='dashboard', models=log_data, ind_strat=ind_strat_data, ind_glob=ind_glob_data)


@app.route('/data')
def data_page():
    meta_file = 'dataset_metadata.json'
    last_updated = "Unknown"
    total_stocks = 0
    if os.path.exists(meta_file):
        with open(meta_file, 'r') as f:
            try: 
                meta_data = json.load(f)
                last_updated = meta_data.get('last_updated', meta_data.get('global_last_date', 'Unknown'))
                total_stocks = len(meta_data.get('symbols', {}))
            except: pass
            
    final_csv_size = "Unknown"
    if os.path.exists('5_year_final_data.csv'):
        size_bytes = os.path.getsize('5_year_final_data.csv')
        final_csv_size = f"{size_bytes / (1024*1024):.1f} MB"

    return render_template_string(HTML_TEMPLATE, page='data', status=TASK_STATUS, 
                                  last_updated_date=last_updated, total_stocks=total_stocks, 
                                  final_csv_size=final_csv_size)

@app.route('/update')
def update_page():
    return render_template_string(HTML_TEMPLATE, page='update', status=TASK_STATUS)

@app.route('/predict')
def predict_page():
    pred_file = 'live_predictions.json'
    predictions = {}
    timestamp = ""
    if os.path.exists(pred_file):
        with open(pred_file, 'r') as f:
            try:
                data = json.load(f)
                ts = datetime.fromisoformat(data['timestamp'])
                if datetime.now() - ts < timedelta(hours=24):
                    predictions = data['predictions']
                    timestamp = ts.strftime("%Y-%m-%d %H:%M")
                else:
                    os.remove(pred_file)
            except: pass

    return render_template_string(HTML_TEMPLATE, page='predict', predictions=predictions, timestamp=timestamp)

@app.route('/api/status')
def api_status(): return jsonify(TASK_STATUS)

@app.route('/api/predict_status')
def api_predict_status(): return jsonify(PREDICT_STATUS)

@app.route('/api/start_data', methods=['POST'])
def api_start_data():
    global TASK_STATUS
    if TASK_STATUS['data'].startswith('Running'): return jsonify({"status": "already running"})
    threading.Thread(target=run_data_job, daemon=True).start()
    return jsonify({"status": "started"})

@app.route('/api/start_update/<task_key>', methods=['POST'])
def api_start_update(task_key):
    global TASK_STATUS
    if TASK_STATUS[task_key].startswith('Running'): return jsonify({"status": "already running"})
    nb_name = '5_years_6_strategy_update.ipynb' if task_key == 'strategy' else '5_years_6.2_global_update.ipynb'
    threading.Thread(target=run_notebook_pipeline, args=(nb_name, task_key), daemon=True).start()
    return jsonify({"status": "started"})

@app.route('/api/start_predict/<symbol>', methods=['POST'])
def api_start_predict(symbol):
    global PREDICT_STATUS
    if PREDICT_STATUS['status'].startswith('Running'): return jsonify({"status": "already running"})
    threading.Thread(target=run_live_prediction, args=(symbol,), daemon=True).start()
    return jsonify({"status": "started"})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True, use_reloader=False)
