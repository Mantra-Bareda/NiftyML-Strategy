import sys
import json
import os
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Setup paths
OUT_FILE = 'live_predictions.json'

def get_code_from_nb(nb_path):
    with open(nb_path, 'r') as f:
        nb = json.load(f)
    code = ""
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            code += "".join(cell['source']) + "\n\n"
    return code

print("Extracting feature engineering logic...")
code2 = get_code_from_nb('5_years_2_feature_generation.ipynb')
# We just need the calculate_features function
import re
match = re.search(r'def calculate_features\(.*?\n    return df', code2, re.DOTALL)
if match:
    calc_func_code = match.group(0)
    exec("import pandas_ta as ta\n" + calc_func_code, globals())
else:
    print("Could not find calculate_features in nb2!")
    sys.exit(1)

code3 = get_code_from_nb('5_years_3_data_cleaning.ipynb')
match3 = re.search(r'column_mapping = \{.*?\}', code3, re.DOTALL)
if match3:
    exec(match3.group(0), globals())
else:
    print("Could not find column_mapping in nb3!")
    sys.exit(1)

symbol = sys.argv[1] if len(sys.argv) > 1 else 'ALL'

if symbol == 'ALL':
    # read from nifty200 list or 5_year_final_data
    df_sym = pd.read_csv('5_year_final_data.csv', usecols=['symbol'])
    symbols = df_sym['symbol'].dropna().unique().tolist()
else:
    symbols = [symbol]

print(f"Starting live prediction for {len(symbols)} symbols...")
predictions = {}

try:
    strat_model = joblib.load('models/5_year_xgboost_model.pkl')
    strat_scaler = joblib.load('models/5_year_scaler.pkl')
    strat_imputer = joblib.load('models/5_year_imputer.pkl')
    with open('final_features.txt', 'r') as f: strat_feats = [line.strip() for line in f.readlines()]
except:
    print("Models not found! Run Update Models first.")
    sys.exit(1)

try:
    gain_model = joblib.load('models/5_year_global_gain_model.pkl')
    drop_model = joblib.load('models/5_year_global_drop_model.pkl')
    glob_scaler = joblib.load('models/5_year_global_scaler.pkl')
    glob_imputer = joblib.load('models/5_year_global_imputer.pkl')
    with open('global_features.txt', 'r') as f: glob_feats = [line.strip() for line in f.readlines()]
    has_global = True
except:
    has_global = False

for sym in symbols:
    try:
        # Download 200d live data
        df = yf.download(sym, period="200d", progress=False)
        if len(df) < 100: continue
        
        # Flatten multiindex if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df['Stock Symbol'] = sym
        df['Sector'] = 'Unknown'
        df = df.reset_index()
        
        # Calculate features
        df_feat = calculate_features(df, None)
        
        # Map columns
        df_feat.rename(columns=column_mapping, inplace=True)
        # some remaining lowercasing
        for c in df_feat.columns:
            if c not in column_mapping.values() and c != 'date':
                df_feat.rename(columns={c: c.lower()}, inplace=True)
                
        if 'stock symbol' in df_feat.columns: df_feat.rename(columns={'stock symbol': 'symbol'}, inplace=True)
        
        # Get the VERY LAST ROW (today's live data)
        last_row = df_feat.iloc[[-1]].copy()
        
        # Extract features and predict
        # Strategy
        X_strat = last_row[strat_feats]
        X_strat_s = strat_scaler.transform(strat_imputer.transform(X_strat))
        pred_r = float(strat_model.predict(X_strat_s)[0])
        
        # Check if strategy is actually triggered today!
        is_triggered = bool(last_row['base_signal'].iloc[0]) if 'base_signal' in last_row.columns else False
        
        # Global
        if has_global:
            X_glob = last_row[glob_feats]
            X_glob_s = glob_scaler.transform(glob_imputer.transform(X_glob))
            pred_gain = float(gain_model.predict(X_glob_s)[0])
            pred_drop = float(drop_model.predict(X_glob_s)[0])
        else:
            pred_gain, pred_drop = 0.0, 0.0
            
        # Save to predictions
        predictions[sym] = {
            "date": str(last_row['date'].iloc[0]),
            "close": float(last_row['close'].iloc[0]),
            "strategy_triggered": is_triggered,
            "predicted_r": round(pred_r, 2),
            "predicted_gain_pct": round(pred_gain * 100, 2),
            "predicted_drop_pct": round(pred_drop * 100, 2),
            "advice": "Hold/Buy" if pred_r >= 1.0 and is_triggered else "Skip"
        }
    except Exception as e:
        print(f"Failed {sym}: {e}")

# Save
out_data = {
    "timestamp": datetime.now().isoformat(),
    "predictions": predictions
}
with open(OUT_FILE, 'w') as f:
    json.dump(out_data, f, indent=2)

print("Live predictions generated successfully!")
