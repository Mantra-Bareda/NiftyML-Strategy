# NiftyML-Strategy

An experimental machine-learning classification layer built on top of a traditional algorithmic trading strategy, designed to identify and filter potentially weak trade setups.

## Overview

NiftyML-Strategy combines rule-based technical analysis with machine learning.

The underlying strategy looks for bullish setups based on a **50-period Moving Average trend, Bollinger Band pullbacks, and a green candle confirmation**. The project then generates historical trade outcomes and uses those setups to build an ML classification dataset.

The long-term goal is to train a model that can distinguish between setups more likely to reach the target and setups more likely to hit the stop loss.

This project is currently an **experimental research and development pipeline**, not a production trading system or a claim of profitable automated trading.

## Strategy Pipeline

```text
Nifty Market Data
       |
       v
Historical OHLCV Data
       |
       v
Technical Feature Engineering
       |
       v
Base Trading Strategy
       |
       v
Trade Detection
       |
       v
Forward-Walk Simulation
       |
       v
Trade Outcome
       |
       v
ML Dataset
       |
       v
Feature Selection / ML Training
```

## Base Strategy

The current base strategy uses:

* 50MA trend confirmation
* Price proximity to the lower Bollinger Band
* Green candle confirmation
* 1:2 risk-to-reward target
* Stop-loss evaluation

Trade outcomes are simulated chronologically over **Day 1 through Day 7** after a signal.

The simulator also handles the ambiguity that can occur when both the target and stop-loss could be reached within the same trading day, particularly on Day 1.

Pending and timeout trades are removed from the ML dataset.

## Data Pipeline

The project dynamically retrieves historical market information using `yfinance`.

The current dataset covers approximately:

* 200 Nifty constituent stocks
* ~565 working days
* 110,000+ combined market-data rows
* 1,825 valid trade setups after filtering

The collected data includes:

* Open
* High
* Low
* Close
* Volume
* Sector information

## Feature Engineering

The pipeline generates **130+ technical and contextual features**, including:

* Moving Averages
* RSI
* ATR
* OBV
* Bollinger Bands
* Bollinger Band distances
* Price relationships
* Volume information
* Candle structure
* Wick characteristics
* Doji patterns
* Engulfing patterns
* Other contextual market features

## Machine Learning

The ML layer is framed as a binary classification problem.

```text
ml_target = 1  -> Target hit
ml_target = 0  -> Stop loss hit
```

The project prepares data for classification experiments using the Scikit-learn / XGBoost ecosystem.

ML experimentation and model development are currently handled in:

```text
trade_final_analysis.ipynb
```

The project is currently in **Phase 5+**, with feature selection and active model training underway.

No production-level accuracy or profitability claim is made at this stage.

## Data Validation & Leakage Prevention

Data quality is treated as an important part of the pipeline.

`phase1_validate.py` performs checks across the large dataset, including:

* Duplicate detection
* Missing values
* Infinite values
* Negative prices
* Negative volumes
* Invalid calculated values
* Potential division-by-zero issues

The ML preparation pipeline also includes an explicit anti-leakage step through:

```text
prepare_ml_dataset.py
```

This removes future-looking columns before model training to reduce the risk of look-ahead bias.

## Backtesting Approach

The trade simulator follows a strictly chronological forward-walk approach.

```text
Signal Day
    |
    +--> Day 1
    +--> Day 2
    +--> Day 3
    +--> Day 4
    +--> Day 5
    +--> Day 6
    +--> Day 7
```

The purpose is to determine whether the predefined target or stop-loss was reached first without using future information during signal generation.

## Tech Stack

* Python
* Pandas
* NumPy
* yfinance
* Scikit-learn
* XGBoost
* Jupyter Notebook
* CSV-based datasets

## Project Structure

```text
NiftyML-Strategy/
├── data/
├── scripts/
│   ├── phase1_validate.py
│   └── prepare_ml_dataset.py
├── notebooks/
│   └── trade_final_analysis.ipynb
├── README.md
└── requirements.txt
```

Update the structure above to match the actual repository.

## Current Status

**Active Development**

### Completed

* Historical data collection
* Data engineering pipeline
* Dataset validation
* Technical feature generation
* Base strategy implementation
* Forward-walk trade simulation
* ML dataset preparation
* Anti-leakage checks

### In Progress

* Feature selection
* ML model experimentation
* Classification evaluation
* Strategy/ML comparison

## Disclaimer

This project is for educational and research purposes.

It is an experimental machine-learning and algorithmic trading research project and should not be interpreted as financial advice, investment advice, or a guaranteed profitable trading strategy.

## Author

**Mantra Bareda**

B.Tech — Computer Science & Engineering (Artificial Intelligence)
Mandsaur University


