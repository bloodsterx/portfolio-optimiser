from torch.utils.data import Dataset
class CostDataset(Dataset):
    # for internal usage; DataLoader wraps me, used for readability, clean code and also multithreading is easy

    def __init__(self, X, C):
        self.X = X # shape = (T-1, d_features) for T-1 time period
        self.C = C # shape = (T-1, n_assets) -> actual returns 'costs'

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # retrieve one data sample
        return self.X[idx], self.C[idx]