import torch
import os
import json
import numpy as np

from ..training.model import FlexibleMLPModel
from ..data.data import DataExtractor, DataProcessor


def predict(ticker, model_path, return_features=False):
    """
    Predict next period return for ANY ticker using flexible model.
    
    Works for any stock, even those not in the training set!
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'TSLA')
        model_path: Path to trained flexible model
        return_features: If True, also return the feature vector
    
    Returns:
        dict with prediction, date, and metadata
    """
    # Load model config
    log_path = os.path.join(model_path, "log.json")
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Model log not found: {log_path}")
    
    with open(log_path, "r") as fp:
        log = json.load(fp)
    
    data_period = log["data_period"]
    interval = log["interval"]
    feature_configs = log["feature_configs"]
    in_features = log["in_features"]  # n_features_per_asset
    hidden_layers = tuple(log["hyperparams"]["hidden_layers"])
    max_window = max([item["window"] for item in feature_configs])
    
    print(f"Loading model: {model_path}")
    print(f"  Data period: {data_period}, Interval: {interval}")
    print(f"  Features per asset: {in_features}")
    
    # Extract and process data for this ticker
    extractor = DataExtractor([ticker])
    timeseries = extractor.extract_yfinance(period=data_period, interval=interval)
    
    if timeseries is None or timeseries.empty:
        raise ValueError(f"No data extracted for {ticker}")
    
    # Process data
    processor = DataProcessor(timeseries, date_col="Date", null_threshold=0.1)
    processor.clean_data().compute_returns().add_features(feature_configs).finalize(max_window)
    
    # Get latest features
    latest_features, latest_date = processor.get_latest_features()
    
    # For single asset, features are already in correct shape: (1, n_features_per_asset)
    if latest_features.shape[1] != in_features:
        raise ValueError(
            f"Feature mismatch! Model expects {in_features} features, "
            f"got {latest_features.shape[1]}"
        )
    
    # Load model
    model = FlexibleMLPModel(in_features, *hidden_layers)
    weights_path = os.path.join(model_path, "weights.pt")
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found: {weights_path}")
    
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()
    
    # Predict
    X_input = torch.tensor(latest_features, dtype=torch.float32)
    
    with torch.inference_mode():
        prediction = model(X_input)
    
    result = {
        'ticker': ticker,
        'prediction': float(prediction.item()),
        'latest_date': str(latest_date),
        'interval': interval,
        'model_path': model_path,
    }
    
    if return_features:
        result['features'] = latest_features.tolist()
    
    print(f"\n{ticker} prediction:")
    print(f"  Latest data: {latest_date}")
    print(f"  Predicted return: {result['prediction']:.4f} ({result['prediction']*100:.2f}%)")
    
    return result


def predict_multiple(tickers, model_path, verbose=True):
    """
    Predict returns for multiple tickers.
    
    Args:
        tickers: List of ticker symbols
        model_path: Path to trained flexible model
        verbose: Print progress
    
    Returns:
        dict with predictions for all tickers
    """
    results = {}
    successful = 0
    failed = 0
    
    for i, ticker in enumerate(tickers, 1):
        if verbose:
            print(f"\n[{i}/{len(tickers)}] Processing {ticker}...")
        
        try:
            result = predict(ticker, model_path, return_features=False)
            results[ticker] = result
            successful += 1
        except Exception as e:
            if verbose:
                print(f"  ❌ Failed: {e}")
            results[ticker] = {'error': str(e)}
            failed += 1
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Summary: {successful} successful, {failed} failed")
    
    return results

