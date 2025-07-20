import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import sys
import os
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.mlp import MLP
from models.rnn import RNNModel
from models.lstm import LSTMModel
from sklearn.metrics import confusion_matrix, accuracy_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------
# MPJPE Metric
# -------------------------
def mpjpe(pred, target):
    return torch.mean(torch.norm(pred - target, dim=-1))

model_name = "lstm"  # or "rnn", "mlp" — change this to train different models
batch_size = 32
num_epochs = 500
lr = 1e-3

# -------------------------
# Training Loop
# -------------------------
def train(model, train_loader, num_epochs=10, lr=1e-3, model_name="default_model"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_forecast = nn.MSELoss()
    loss_class = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss, total_loss_f, total_loss_c = 0.0, 0.0, 0.0
        correct, total = 0, 0
        all_mpjpe = []

        for src, (trg_forecast, trg_class), _, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            src = src.to(device)
            trg_forecast = trg_forecast.to(device)
            trg_class = trg_class.to(device)

            src = src.view(src.size(0), src.size(1), -1)
            trg_forecast = trg_forecast.view(trg_forecast.size(0), trg_forecast.size(1), -1)

            optimizer.zero_grad()
            forecast_out, class_out = model(src)

            loss_f = loss_forecast(forecast_out, trg_forecast)
            loss_c = loss_class(class_out, torch.argmax(trg_class, dim=1))
            loss = loss_f + loss_c

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * src.size(0)
            total_loss_f += loss_f.item() * src.size(0)
            total_loss_c += loss_c.item() * src.size(0)

            all_mpjpe.append(mpjpe(forecast_out, trg_forecast).item())
            pred = torch.argmax(class_out, dim=1)
            correct += (pred == torch.argmax(trg_class, dim=1)).sum().item()
            total += trg_class.size(0)

        avg_loss = total_loss / total
        avg_loss_f = total_loss_f / total
        avg_loss_c = total_loss_c / total
        avg_mpjpe = sum(all_mpjpe) / len(all_mpjpe)
        acc = correct / total

        print(f"[Epoch {epoch+1}] Train → Loss: {avg_loss:.4f} | Forecast: {avg_loss_f:.4f} | "
              f"Class: {avg_loss_c:.4f} | MPJPE: {avg_mpjpe:.4f} | Acc: {acc:.4f}")

        # # Optional: visualize prediction after final epoch
        # if epoch == num_epochs - 1:
        #     visualize_forecast(src, forecast_out, trg_forecast, step_name=f"Epoch_{epoch+1}", model_name=model_name)

    # Save the model to ../models/
    model_path = f"../models/{model_name}_{num_epochs}.pt"
    torch.save(model.state_dict(), model_path)

    print(f"[INFO] Model saved to {model_path}")

# -------------------------
# Run Training
# -------------------------
if __name__ == "__main__":
    import re
    import os
    import pickle
    from utils.dataset import PoseDataset
    from torch.utils.data import DataLoader
    from collections import defaultdict, Counter

    # Load preprocessed dataset from utils/dataset.pkl
    with open(os.path.join(os.path.dirname(__file__), "dataset.pkl"), "rb") as f:
        raw_data = pickle.load(f)

    # Assuming you already have raw_data loaded
    subjects = raw_data["subjects"]
    motion_types = raw_data["motion_types"]
    unique_subjects = sorted(set(subjects))
    train_subjects = unique_subjects[0:4]

    # Prepare train_data dict
    train_data = {
        "src": [], "trg_forecast": [], "trg_class": [],
        "subjects": [], "motion_types": []
    }
    
    for i, subj in enumerate(subjects):
        if subj in train_subjects:
            train_data["src"].append(raw_data["src"][i])
            train_data["trg_forecast"].append(raw_data["trg_forecast"][i])
            train_data["trg_class"].append(raw_data["trg_class"][i])
            train_data["subjects"].append(subj)
            train_data["motion_types"].append(raw_data["motion_types"][i])

    # Create dataset and loaders
    train_dataset = PoseDataset(train_data, return_subject=True)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Kumpulan untuk menyimpan hasil
    subject_to_motions = defaultdict(set)

    # Mapping: subject -> motion_type -> jumlah window
    subject_motion_windows = defaultdict(Counter)

    # Iterasi semua batch di train_loader
    for src, (trg_forecast, trg_class), subjects, motions in train_loader:
        for subj, motion in zip(subjects, motions):
            subject_motion_windows[subj][motion] += 1

    # Cetak hasil
    print("\n[INFO] Sliding window count per subject and motion type:")
    for subj in sorted(subject_motion_windows.keys()):
        print(f"{subj}:")
        for motion, count in subject_motion_windows[subj].items():
            print(f"  - {motion}: {count} sliding windows")

    # Extract input dimensions
    input_window = len(train_data["src"][0])  # e.g. 15
    keypoints_dim = len(train_data["src"][0][0]) * len(train_data["src"][0][0][0])  # 17 * 2 = 34

    # Build model
    if model_name == "mlp":
        from models.mlp import MLP
        model = MLP(
            input_size=input_window * keypoints_dim,
            hidden_size=512,
            forecast_window=input_window,
            output_class_size=2
        )

    elif model_name == "rnn":
        from models.rnn import RNNModel
        model = RNNModel(
            input_size=34,
            hidden_size=128,
            forecast_window=input_window,
            output_class_size=2
        )

    elif model_name == "lstm":
        from models.lstm import LSTMModel
        model = LSTMModel(
            input_size=34,
            hidden_size=128,
            forecast_window=input_window,
            output_class_size=2
        )

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    model.to(device)

    print(f"Using model: {model_name.upper()}")

    # Train and save model
    train(model, train_loader, num_epochs=num_epochs, lr=lr, model_name=model_name)