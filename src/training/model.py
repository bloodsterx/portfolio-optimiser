import torch.nn as nn

"""
Long term TODO:
- Create model using LSTM dropout layers
- Create random forest model
- Create model using causal convolutions (need to read more)
- Create model using Attention & Transformers (kelly did a paper on this I think)
"""

class MLPModel(nn.Module):
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
