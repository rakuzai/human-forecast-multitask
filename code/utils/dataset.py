from torch.utils.data import Dataset
import torch

class PoseDataset(Dataset):
    def __init__(self, data_dict):
        self.src = data_dict["src"]
        self.trg_forecast = data_dict["trg_forecast"]
        self.trg_class = data_dict["trg_class"]

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.src[idx], dtype=torch.float32),
            (
                torch.tensor(self.trg_forecast[idx], dtype=torch.float32),
                torch.tensor(self.trg_class[idx], dtype=torch.float32)
            )
        )