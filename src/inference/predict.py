import torch
import os
from pathlib import Path

from ..training.model import MLPModel
from ..data.data import DataExtractor, DataProcessor



# def which_models(model_path="models/"):
#     for file in os.listdir(model_path):
        



def predict(model_path: str, ticker: str, tickers_list: list[str] = None, period: str = "5y", interval: str = "1mo"):
    """
    Run inference on a pre-trained model to forecast future returns for a specific ticker.
    
    Args:
        model_path (str): Path to the saved model state dict (.pt file)
        ticker (str): Stock ticker symbol to predict returns for
        tickers_list (list[str], optional): List of all tickers the model was trained on. 
                                           If None, will attempt to load from default CSV.
        period (str): Historical data period for feature calculation (default: "5y")
        interval (str): Data interval (default: "1mo")
    
    Returns:
        dict: Contains predicted return for the ticker and metadata
            {
                'ticker': str,
                'predicted_return': float,
                'feature_date': str (date of latest features used)
            }
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Step 1: Load tickers list (same as training)
    if tickers_list is None:
        # Try to load from default CSV
        csv_path = Path(__file__).parent.parent.parent / "sp500-stocks.csv"
        if csv_path.exists():
            extractor = DataExtractor()
            tickers_list = extractor.extract_csv(str(csv_path))
        else:
            raise ValueError("No tickers_list provided and default CSV not found. "
                           "Please provide tickers_list parameter.")
    
    # Validate that requested ticker is in the training set
    if ticker not in tickers_list:
        raise ValueError(f"Ticker '{ticker}' not found in the model's training set. "
                        f"Available tickers: {tickers_list}")
    
    # Step 2: Extract data for all tickers (model needs features for all assets)
    extractor = DataExtractor(tickers=tickers_list)
    timeseries = extractor.extract_yfinance(period=period, interval=interval)
    
    # Step 3: Process data using DataProcessor
    processor = DataProcessor(timeseries, date_col="Date", null_threshold=0.1)
    
    # Define feature configuration (same as training)
    feature_configs = [
        {'type': 'momentum', 'window': 1},
        {'type': 'momentum', 'window': 12},
        {'type': 'volatility', 'window': 3},
        {'type': 'volatility', 'window': 12},
    ]
    
    # Process data through the pipeline
    processor.clean_data().compute_returns().add_features(feature_configs).finalize(max_window=13)
    
    # Get the most recent observation for prediction
    latest_features, latest_date = processor.get_latest_features()
    asset_cols = processor.get_asset_columns()
    
    # Step 5: Load model architecture and weights
    n_features = latest_features.shape[1]
    n_assets = len(asset_cols)
    
    # Initialize model (assume default architecture - no hidden layers specified)
    model = MLPModel(n_features, n_assets)
    
    # Load trained weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Step 6: Run inference
    X_latest = torch.tensor(latest_features, dtype=torch.float32).to(device)
    
    with torch.inference_mode():
        predictions = model(X_latest)
    
    # Step 7: Extract prediction for the requested ticker
    ticker_idx = asset_cols.index(ticker)
    predicted_return = predictions[0, ticker_idx].item()
    
    result = {
        'ticker': ticker,
        'predicted_return': predicted_return,
        'feature_date': str(latest_date),
        'all_predictions': {asset_cols[i]: predictions[0, i].item() 
                           for i in range(len(asset_cols))}
    }
    
    return result


if __name__ == "__main__":
    # Example usage
    model_path = "2024-12-28-DL-weights.pt"  # Replace with your actual model path
    ticker = "AAPL"
    
    try:
        result = predict(model_path, ticker)
        print(f"\n{'='*50}")
        print(f"Prediction for {result['ticker']}")
        print(f"{'='*50}")
        print(f"Predicted Return: {result['predicted_return']:.4%}")
        print(f"Feature Date: {result['feature_date']}")
        print(f"\nTop 5 Predicted Returns (All Assets):")
        sorted_predictions = sorted(result['all_predictions'].items(), 
                                   key=lambda x: x[1], reverse=True)
        for ticker_name, ret in sorted_predictions[:5]:
            print(f"  {ticker_name}: {ret:.4%}")
    except Exception as e:
        print(f"Error: {e}")
    