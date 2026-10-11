import sys
import json
import os
import warnings
warnings.filterwarnings('ignore')
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

OUT_FILE = 'live_predictions.json'

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def calc_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(period).mean()

def calculate_stock_features(df, nifty_df=None):
    df = df.copy()
    
    # 1. Moving Averages
    for ma in [15, 30, 50, 100, 150, 200]:
        df[f'MA{ma}'] = df['Close'].rolling(window=ma).mean()
        df[f'MA{ma}_slope'] = df[f'MA{ma}'].diff(1) / (df[f'MA{ma}'].shift(1) + 1e-9) * 100
        df[f'Close_vs_MA{ma}'] = df['Close'] / (df[f'MA{ma}'] + 1e-9)
        
    df['MA15_vs_MA30'] = df['MA15'] / (df['MA30'] + 1e-9)
    df['MA30_vs_MA50'] = df['MA30'] / (df['MA50'] + 1e-9)
    df['MA50_vs_MA100'] = df['MA50'] / (df['MA100'] + 1e-9)
    df['MA50_vs_MA150'] = df['MA50'] / (df['MA150'] + 1e-9)
    df['MA100_vs_MA200'] = df['MA100'] / (df['MA200'] + 1e-9)
    
    # 2. Bollinger Bands (20, 2)
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
    df['BB_Width_pct'] = df['BB_Width'] / (df['BB_Middle'] + 1e-9)
    df['BB_pctB'] = (df['Close'] - df['BB_Lower']) / (df['BB_Width'] + 1e-9)
    df['Close_dist_BB_Upper'] = df['Close'] - df['BB_Upper']
    df['Close_dist_BB_Middle'] = df['Close'] - df['BB_Middle']
    df['Close_dist_BB_Lower'] = df['Close'] - df['BB_Lower']
    df['BB_Width_change'] = df['BB_Width'].diff(1)
    
    # 3. RSI
    for r in [7, 14, 21]:
        df[f'RSI_{r}'] = calc_rsi(df['Close'], period=r)
    df['RSI_slope'] = df['RSI_14'].diff(1)
    df['RSI_change_3D'] = df['RSI_14'].diff(3)
    df['RSI_change_5D'] = df['RSI_14'].diff(5)
    
    # 4. Returns
    for d in [1, 3, 5, 7, 10, 20, 50]:
        df[f'Ret_{d}D'] = df['Close'].pct_change(d) * 100
        
    # 5. Price Structure
    for w in [5, 20]:
        df[f'{w}D_High'] = df['High'].rolling(w).max()
        df[f'{w}D_Low'] = df['Low'].rolling(w).min()
        df[f'Dist_{w}D_High'] = (df['Close'] - df[f'{w}D_High']) / (df[f'{w}D_High'] + 1e-9) * 100
        df[f'Dist_{w}D_Low'] = (df['Close'] - df[f'{w}D_Low']) / (df[f'{w}D_Low'] + 1e-9) * 100
        
    df['52W_High'] = df['High'].rolling(250).max()
    df['52W_Low'] = df['Low'].rolling(250).min()
    df['Dist_52W_High'] = (df['Close'] - df['52W_High']) / (df['52W_High'] + 1e-9) * 100
    df['Dist_52W_Low'] = (df['Close'] - df['52W_Low']) / (df['52W_Low'] + 1e-9) * 100
    
    # 6. Candlestick Features
    df['Body'] = np.abs(df['Close'] - df['Open'])
    df['Range'] = df['High'] - df['Low']
    df['Upper_Wick'] = df['High'] - np.maximum(df['Close'], df['Open'])
    df['Lower_Wick'] = np.minimum(df['Close'], df['Open']) - df['Low']
    df['Body_to_Range'] = df['Body'] / (df['Range'] + 1e-9)
    df['Upper_Wick_to_Range'] = df['Upper_Wick'] / (df['Range'] + 1e-9)
    df['Lower_Wick_to_Range'] = df['Lower_Wick'] / (df['Range'] + 1e-9)
    df['Close_Pos_in_Range'] = (df['Close'] - df['Low']) / (df['Range'] + 1e-9)
    df['Gap_pct'] = (df['Open'] - df['Close'].shift(1)) / (df['Close'].shift(1) + 1e-9) * 100
    df['Is_Hammer'] = (df['Lower_Wick'] > (2 * df['Body'])) & (df['Upper_Wick'] < (0.2 * df['Body']))
    df['Is_Inside_Candle'] = (df['High'] < df['High'].shift(1)) & (df['Low'] > df['Low'].shift(1))
    df['Is_Outside_Candle'] = (df['High'] > df['High'].shift(1)) & (df['Low'] < df['Low'].shift(1))
    
    # 7. Volume Features
    for v in [5, 20, 50]:
        df[f'Vol_MA{v}'] = df['Volume'].rolling(v).mean()
        df[f'Vol_Ratio_{v}D'] = df['Volume'] / (df[f'Vol_MA{v}'] + 1e-9)
    df['Vol_Change_1D'] = df['Volume'].pct_change(1) * 100
    df['Vol_Change_5D'] = df['Volume'].pct_change(5) * 100
    
    # OBV
    obv_dir = np.sign(df['Close'].diff().fillna(0))
    df['OBV'] = (obv_dir * df['Volume']).cumsum()
    df['OBV_slope'] = df['OBV'].diff(5)
    
    # 8. Volatility
    df['ATR14'] = calc_atr(df, 14)
    df['ATR_pct'] = df['ATR14'] / (df['Close'] + 1e-9) * 100
    for d in [5, 10, 20, 50]:
        df[f'Volat_{d}D'] = df['Ret_1D'].rolling(d).std()
    df['Volat_change'] = df['Volat_10D'].diff()
    
    # 9. Trend Structure
    df['Trend_15_30'] = df['MA15'] > df['MA30']
    df['Trend_30_50'] = df['MA30'] > df['MA50']
    df['Trend_50_100'] = df['MA50'] > df['MA100']
    df['Trend_100_200'] = df['MA100'] > df['MA200']
    df['Trend_15_30_50_Up'] = df['Trend_15_30'] & df['Trend_30_50']
    df['Trend_15_30_50_Dn'] = (df['MA15'] < df['MA30']) & (df['MA30'] < df['MA50'])
    df['MA15_cross_MA50'] = (df['MA15'].shift(1) <= df['MA50'].shift(1)) & (df['MA15'] > df['MA50'])
    df['MA100_above_MA200'] = df['MA100'] > df['MA200']
    
    # 10. Support / Resistance
    df['Support_dist'] = df['Dist_20D_Low']
    df['Resistance_dist'] = df['Dist_20D_High']
    df['Support_touches'] = (df['Low'] <= (df['20D_Low'] * 1.01)).rolling(20).sum()
    df['Resistance_touches'] = (df['High'] >= (df['20D_High'] * 0.99)).rolling(20).sum()
    
    # 11. Market Context (Nifty 50)
    if nifty_df is not None and not nifty_df.empty:
        n_close = nifty_df['Close'].reindex(df.index).ffill()
        for d in [1, 5, 7, 20]:
            df[f'Nifty_{d}D_ret'] = n_close.pct_change(d) * 100
        df['Nifty_MA50'] = n_close.rolling(50).mean()
        df['Nifty_MA200'] = n_close.rolling(200).mean()
        df['Nifty_vs_MA50'] = n_close / (df['Nifty_MA50'] + 1e-9)
        df['Nifty_vs_MA200'] = n_close / (df['Nifty_MA200'] + 1e-9)
        df['Nifty_RSI'] = calc_rsi(n_close)
        df['Nifty_Trend'] = df['Nifty_MA50'] > df['Nifty_MA200']
        for d in [5, 7, 20]:
            df[f'Rel_str_Nifty_{d}D'] = df[f'Ret_{d}D'] - df[f'Nifty_{d}D_ret']
            
    # 12. Base Strategy Signal
    df['MA50_trend_7D_up'] = (df['MA50'] > df['MA50'].shift(7)) & (df['MA50'] > df['MA50'].shift(1))
    df['Dist_Low_BB'] = df['Low'] - df['BB_Lower']
    df['BB_near_Low'] = (df['Dist_Low_BB'] >= -10) & (df['Dist_Low_BB'] <= 20)
    df['Candle_Green'] = df['Close'] > df['Open']
    df['Dist_Low_MA50'] = df['Low'] - df['MA50']
    df['Setup_1'] = (df['Dist_Low_MA50'] >= -10) & (df['Dist_Low_MA50'] <= 30)
    df['Setup_2'] = df['Dist_Low_MA50'] > 30
    df['Setup_3'] = df['Dist_Low_MA50'] < -10
    df['Base_Signal'] = df['MA50_trend_7D_up'] & df['BB_near_Low'] & df['Candle_Green'] & (df['Setup_1'] | df['Setup_2'] | df['Setup_3'])
    
    return df


COLUMN_MAPPING = {
    'Date': 'date', 'Stock Symbol': 'symbol', 'Sector': 'sector',
    'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close',
    'Adj Close': 'adjusted_close', 'Volume': 'volume',
    'MA15': 'ma15', 'MA30': 'ma30', 'MA50': 'ma50', 'MA100': 'ma100', 'MA150': 'ma150', 'MA200': 'ma200',
    'MA15_slope': 'ma15_slope', 'MA30_slope': 'ma30_slope', 'MA50_slope': 'ma50_slope',
    'MA100_slope': 'ma100_slope', 'MA150_slope': 'ma150_slope', 'MA200_slope': 'ma200_slope',
    'Close_vs_MA15': 'close_ma15_ratio', 'Close_vs_MA30': 'close_ma30_ratio', 'Close_vs_MA50': 'close_ma50_ratio',
    'Close_vs_MA100': 'close_ma100_ratio', 'Close_vs_MA150': 'close_ma150_ratio',
    'MA15_vs_MA30': 'ma15_ma30_ratio', 'MA30_vs_MA50': 'ma30_ma50_ratio',
    'MA50_vs_MA100': 'ma50_ma100_ratio', 'MA50_vs_MA150': 'ma50_ma150_ratio', 'MA100_vs_MA200': 'ma100_ma200_ratio',
    'BB_Middle': 'bb_middle', 'BB_Upper': 'bb_upper', 'BB_Lower': 'bb_lower',
    'BB_Width': 'bb_width', 'BB_Width_pct': 'bb_width_pct', 'BB_pctB': 'bb_percent_b',
    'Close_dist_BB_Upper': 'close_bb_upper_dist', 'Close_dist_BB_Middle': 'close_bb_middle_dist', 'Close_dist_BB_Lower': 'close_bb_lower_dist',
    'BB_Width_change': 'bb_width_change',
    'RSI_7': 'rsi7', 'RSI_14': 'rsi14', 'RSI_21': 'rsi21',
    'RSI_slope': 'rsi_slope', 'RSI_change_3D': 'rsi_change_3d', 'RSI_change_5D': 'rsi_change_5d',
    'Ret_1D': 'return_1d', 'Ret_3D': 'return_3d', 'Ret_5D': 'return_5d', 'Ret_7D': 'return_7d',
    'Ret_10D': 'return_10d', 'Ret_20D': 'return_20d', 'Ret_50D': 'return_50d',
    '5D_Low': 'low_5d', 'Dist_5D_High': 'dist_high_5d', 'Dist_5D_Low': 'dist_low_5d',
    'Dist_20D_High': 'dist_high_20d', 'Dist_20D_Low': 'dist_low_20d',
    'Dist_52W_High': 'dist_high_52w', 'Dist_52W_Low': 'dist_low_52w',
    'Body': 'candle_body', 'Range': 'candle_range', 'Upper_Wick': 'upper_wick', 'Lower_Wick': 'lower_wick',
    'Body_to_Range': 'body_range_ratio', 'Upper_Wick_to_Range': 'upper_wick_range_ratio', 'Lower_Wick_to_Range': 'lower_wick_range_ratio',
    'Close_Pos_in_Range': 'close_position', 'Gap_pct': 'gap_pct',
    'Is_Hammer': 'is_hammer', 'Is_Inside_Candle': 'is_inside_candle', 'Is_Outside_Candle': 'is_outside_candle',
    'Vol_MA5': 'volume_avg_5d', 'Vol_MA20': 'volume_avg_20d', 'Vol_MA50': 'volume_avg_50d',
    'Vol_Ratio_20D': 'volume_ratio_20d', 'Vol_Ratio_50D': 'volume_ratio_50d',
    'Vol_Change_1D': 'volume_change_1d', 'Vol_Change_5D': 'volume_change_5d',
    'OBV': 'obv', 'OBV_slope': 'obv_slope', 'ATR_pct': 'atr_pct',
    'Volat_5D': 'volatility_5d', 'Volat_10D': 'volatility_10d', 'Volat_20D': 'volatility_20d', 'Volat_50D': 'volatility_50d',
    'Volat_change': 'volatility_change',
    'MA100_above_MA200': 'ma100_above_ma200', 'MA15_cross_MA50': 'ma_crossover',
    'Support_touches': 'support_touches', 'Resistance_touches': 'resistance_touches',
    'Nifty_1D_ret': 'nifty_return_1d', 'Nifty_5D_ret': 'nifty_return_5d', 'Nifty_7D_ret': 'nifty_return_7d', 'Nifty_20D_ret': 'nifty_return_20d',
    'Nifty_MA50': 'nifty_ma50', 'Nifty_MA200': 'nifty_ma200', 'Nifty_vs_MA50': 'nifty_ma50_distance', 'Nifty_vs_MA200': 'nifty_ma200_distance',
    'Nifty_RSI': 'nifty_rsi', 'Nifty_Trend': 'nifty_trend',
    'Rel_str_Nifty_5D': 'relative_strength_5d', 'Rel_str_Nifty_7D': 'relative_strength_7d', 'Rel_str_Nifty_20D': 'relative_strength_20d',
    'Base_Signal': 'base_signal'
}


def run_predictor():
    raw_symbol = sys.argv[1].strip() if len(sys.argv) > 1 and sys.argv[1].strip() else 'RELIANCE'
    
    # Load Models
    try:
        strat_model = joblib.load('models/5_year_xgboost_model.pkl')
        strat_scaler = joblib.load('models/5_year_scaler.pkl')
        strat_imputer = joblib.load('models/5_year_imputer.pkl')
        with open('final_features.txt', 'r') as f:
            strat_feats = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"Error loading strategy models: {e}")
        sys.exit(1)

    try:
        gain_model = joblib.load('models/5_year_global_gain_model.pkl')
        drop_model = joblib.load('models/5_year_global_drop_model.pkl')
        glob_scaler = joblib.load('models/5_year_global_scaler.pkl')
        glob_imputer = joblib.load('models/5_year_global_imputer.pkl')
        with open('global_features.txt', 'r') as f:
            glob_feats = [line.strip() for line in f if line.strip()]
        has_global = True
    except Exception as e:
        print(f"Warning: Global models not loaded: {e}")
        has_global = False

    # Target symbols
    if raw_symbol.upper() == 'ALL':
        if os.path.exists('dataset_metadata.json'):
            with open('dataset_metadata.json') as f:
                meta = json.load(f)
            # Take top 30 key liquid stocks to keep it fast (<20s)
            all_syms = list(meta.get('symbols', {}).keys())
            symbols = all_syms[:30] if len(all_syms) > 30 else all_syms
        else:
            symbols = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 'SBIN', 'BHARTIARTL', 'TATAMOTORS']
    else:
        clean = raw_symbol.upper().replace('.NS', '').strip()
        symbols = [clean]

    print(f"Fetching market data for {len(symbols)} symbol(s)...")

    # Download Nifty 50 benchmark once
    try:
        nifty = yf.download('^NSEI', period='250d', progress=False)
        if isinstance(nifty.columns, pd.MultiIndex):
            nifty.columns = nifty.columns.get_level_values(0)
    except Exception:
        nifty = None

    # Load existing predictions to merge
    predictions = {}
    if os.path.exists(OUT_FILE):
        try:
            with open(OUT_FILE, 'r') as f:
                existing = json.load(f)
                predictions = existing.get('predictions', {})
        except Exception:
            predictions = {}

    for sym in symbols:
        try:
            yf_ticker = f"{sym}.NS"
            df = yf.download(yf_ticker, period="250d", progress=False)
            if df.empty or len(df) < 50:
                print(f"Warning: Not enough data for {sym}")
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df['Stock Symbol'] = sym
            df['Sector'] = 'General'
            if 'Adj Close' not in df.columns:
                df['Adj Close'] = df['Close']
            df = df.reset_index()

            # Calculate technical features
            df_feat = calculate_stock_features(df, nifty)
            df_feat.rename(columns=COLUMN_MAPPING, inplace=True)

            # Lowercase remaining columns
            for col in df_feat.columns:
                if col not in COLUMN_MAPPING.values() and col != 'Date':
                    df_feat.rename(columns={col: col.lower()}, inplace=True)

            if 'stock symbol' in df_feat.columns:
                df_feat.rename(columns={'stock symbol': 'symbol'}, inplace=True)
            if 'adjusted_close' not in df_feat.columns:
                df_feat['adjusted_close'] = df_feat['close']

            if 'ma_crossover' in df_feat.columns:
                crossover_id = df_feat['ma_crossover'].cumsum()
                df_feat['days_since_ma_crossover'] = df_feat.groupby(crossover_id).cumcount()

            # Ensure all required columns exist
            for col in strat_feats:
                if col not in df_feat.columns:
                    df_feat[col] = np.nan
            if has_global:
                for col in glob_feats:
                    if col not in df_feat.columns:
                        df_feat[col] = np.nan

            df_feat.replace([np.inf, -np.inf], np.nan, inplace=True)

            last_row = df_feat.iloc[[-1]].copy()

            # Strategy prediction
            X_strat = last_row[strat_feats]
            X_strat_s = strat_scaler.transform(strat_imputer.transform(X_strat))
            pred_r = float(strat_model.predict(X_strat_s)[0])

            is_triggered = bool(last_row.get('base_signal', pd.Series([False])).iloc[0])

            # Global prediction
            if has_global:
                X_glob = last_row[glob_feats]
                X_glob_s = glob_scaler.transform(glob_imputer.transform(X_glob))
                pred_gain = float(gain_model.predict(X_glob_s)[0])
                pred_drop = float(drop_model.predict(X_glob_s)[0])
            else:
                pred_gain, pred_drop = 0.0, 0.0

            # Friendly recommendation logic
            close_price = round(float(last_row['close'].iloc[0]), 2)
            pred_gain_pct = round(pred_gain, 2)
            pred_drop_pct = round(pred_drop, 2)

            if is_triggered:
                if pred_r >= 1.0:
                    verdict = "BUY SIGNAL"
                    verdict_class = "buy"
                    explanation = f"Base 50MA setup triggered. High expected reward of {pred_r:.2f}R."
                else:
                    verdict = "SKIP SIGNAL"
                    verdict_class = "skip"
                    explanation = f"Base setup triggered but AI predicts lower payoff ({pred_r:.2f}R). Filtered out."
            else:
                if pred_gain_pct > abs(pred_drop_pct):
                    verdict = "BULLISH SETUP"
                    verdict_class = "bullish"
                    explanation = "Upward momentum expected (+{0}%). Awaiting entry pullback.".format(pred_gain_pct)
                else:
                    verdict = "NEUTRAL / CONSOLIDATION"
                    verdict_class = "neutral"
                    explanation = "Rangebound action expected. No high-edge setup currently."

            date_val = str(last_row['Date'].iloc[0]) if 'Date' in last_row.columns else str(last_row['date'].iloc[0])

            predictions[sym] = {
                "date": date_val[:10],
                "close": close_price,
                "strategy_triggered": is_triggered,
                "predicted_r": round(pred_r, 2),
                "predicted_gain_pct": pred_gain_pct,
                "predicted_drop_pct": pred_drop_pct,
                "verdict": verdict,
                "verdict_class": verdict_class,
                "explanation": explanation
            }
            print(f"[OK] Predicted {sym}: {close_price} INR | {verdict}")

        except Exception as e:
            print(f"Error predicting {sym}: {e}")

    # Save to JSON
    out_data = {
        "timestamp": datetime.now().isoformat(),
        "predictions": predictions
    }
    with open(OUT_FILE, 'w') as f:
        json.dump(out_data, f, indent=2)

    print("Live prediction task completed successfully!")


if __name__ == '__main__':
    run_predictor()
