import torch.nn as nn

"""
Long term TODO:
- Create model using LSTM
- Create random forest model
- Create model using causal convolutions (need to read more)
- Create model using Attention & Transformers (kelly did a paper on this I think)
"""

class MLPModel(nn.Module):
    """
    Portfolio MLP model that predicts returns for a fixed set of assets.
    
    This model expects features from ALL assets concatenated together,
    and outputs predictions for all assets simultaneously.
    """
    def __init__(self, d_features, n_assets, *h_layers):
        # (i, h_layers[i]) = i'th hidden layer and the number of neurons in that layer 
        assert all(isinstance(x, int) for x in h_layers)
        super().__init__()

        # build list of linear layers based on h_layers  
        layers = []
        input_dim = d_features
        
        for hidden_layer in h_layers:
            layers.append(nn.Linear(in_features=input_dim, out_features=hidden_layer))
            # for now, just ReLU. Consider LSTM later.
            layers.append(nn.ReLU())
            input_dim = hidden_layer

        layers.append(nn.Linear(in_features=input_dim, out_features=n_assets))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class FlexibleMLPModel(nn.Module):
    """
    Flexible MLP that predicts returns for individual assets.
    
    Unlike the portfolio model which expects features from ALL assets,
    this model processes one asset at a time using shared weights.
    This allows it to predict returns for ANY stock, even ones not
    seen during training.
    
    Args:
        n_features_per_asset: Number of features per asset (e.g., 4)
        *h_layers: Hidden layer sizes (e.g., 32, 16)
    
    Example:
        model = FlexibleMLPModel(4, 32, 16)
        # Input: (batch_size, 4) - features for any stocks
        # Output: (batch_size, 1) - predicted return for each
    """
    def __init__(self, n_features_per_asset, *h_layers):
        super().__init__()
        
        if not h_layers:
            raise ValueError("Must specify at least one hidden layer")
        
        assert all(isinstance(x, int) for x in h_layers), "Hidden layers must be integers"
        
        # Build sequential network
        layers = []
        input_dim = n_features_per_asset
        
        for hidden_size in h_layers:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.ReLU())
            input_dim = hidden_size
        
        # Output layer: 1 return prediction
        layers.append(nn.Linear(input_dim, 1))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Tensor of shape (batch_size, n_features_per_asset)
        
        Returns:
            Tensor of shape (batch_size, 1) - predicted returns
        """
        return self.net(x)
