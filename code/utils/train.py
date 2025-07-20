import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.mlp import MLP
from models.rnn import RNNModel
from models.lstm import LSTMModel
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import KFold


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------
# MPJPE Metric
# -------------------------
def mpjpe(pred, target):
    return torch.mean(torch.norm(pred - target, dim=-1))

model_name = 'mlp'

import matplotlib.pyplot as plt

SKELETON_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4), 
    (1, 5), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10),
    (8, 11), (11, 12), (12, 13),
    (0, 14), (0, 15), (14, 16), (15, 17)
]

def draw_pose(ax, pose, color):
    for (i, j) in SKELETON_EDGES:
        if i < pose.shape[0] and j < pose.shape[0]:
            ax.plot([pose[i, 0], pose[j, 0]], [-pose[i, 1], -pose[j, 1]], color=color, linewidth=2)
    ax.scatter(pose[:, 0], -pose[:, 1], c=color, s=10)

def visualize_forecast(src, forecast_out, trg_forecast, step_name="", model_name=model_name):
    os.makedirs("result", exist_ok=True)

    src = src[7].detach().cpu().numpy()               # [input_window, 34]
    forecast = forecast_out[7].detach().cpu().numpy() # [forecast_window, 34]
    target = trg_forecast[7].detach().cpu().numpy()   # [forecast_window, 34]

    forecast_window = min(8, forecast.shape[0])  # Limit to 8 frames

    fig, axs = plt.subplots(1, forecast_window, figsize=(forecast_window * 1.5, 3))
    for i in range(forecast_window):
        ax = axs[i]
        pred_pose = forecast[i].reshape(17, 2)
        true_pose = target[i].reshape(17, 2)

        draw_pose(ax, true_pose, color='blue')
        draw_pose(ax, pred_pose, color='red')

        ax.axis('off')
        ax.set_aspect('equal')

    # Manual legend
    legend_elements = [
        mpatches.Patch(color='blue', label='Ground Truth'),
        mpatches.Patch(color='red', label='Prediction')
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2, fontsize=10, bbox_to_anchor=(0.5, 1.05))

    plt.tight_layout()
    save_path = f"../result/forecast_output_{model_name}_{step_name.replace(' ', '_')}.jpg"
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved forecast visualization to: {save_path}")

# -------------------------
# Training Loop
# -------------------------
def train(model, train_loader, val_loader=None, num_epochs=10, lr=1e-3, model_name="mlp"):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_forecast = nn.MSELoss()
    loss_class = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss, total_loss_f, total_loss_c = 0.0, 0.0, 0.0
        correct, total = 0, 0
        all_mpjpe = []

        for src, (trg_forecast, trg_class) in train_loader:
            src = src.to(device)
            trg_forecast = trg_forecast.to(device)
            trg_class = trg_class.to(device)

            src = src.view(src.size(0), src.size(1), -1)  # [B, T, 34]
            trg_forecast = trg_forecast.view(trg_forecast.size(0), trg_forecast.size(1), -1)  # [B, T, 34]

            optimizer.zero_grad()

            forecast_out, class_out = model(src)

            loss_f = loss_forecast(forecast_out, trg_forecast)
            loss_c = loss_class(class_out, torch.argmax(trg_class, dim=1))
            loss = loss_f + loss_c

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_loss_f += loss_f.item()
            total_loss_c += loss_c.item()

            all_mpjpe.append(mpjpe(forecast_out, trg_forecast).item())

            pred = torch.argmax(class_out, dim=1)
            correct += (pred == torch.argmax(trg_class, dim=1)).sum().item()
            total += trg_class.size(0)

        avg_loss = total_loss / len(train_loader)
        avg_loss_f = total_loss_f / len(train_loader)
        avg_loss_c = total_loss_c / len(train_loader)
        avg_mpjpe = sum(all_mpjpe) / len(train_loader)
        acc = correct / total

        print(f"[Epoch {epoch+1}] Train → Loss: {avg_loss:.4f} | Forecast: {avg_loss_f:.4f} | "
              f"Class: {avg_loss_c:.4f} | MPJPE: {avg_mpjpe:.4f} | Acc: {acc:.4f}")

        # Optional validation
        if val_loader is not None:
            model.eval()
            val_loss_f, val_loss_c, val_mpjpe, val_correct, val_total = 0.0, 0.0, 0.0, 0, 0
            with torch.no_grad():
                for src, (trg_forecast, trg_class) in val_loader:
                    src = src.to(device)
                    trg_forecast = trg_forecast.to(device)
                    trg_class = trg_class.to(device)

                    src = src.view(src.size(0), src.size(1), -1)
                    trg_forecast = trg_forecast.view(trg_forecast.size(0), trg_forecast.size(1), -1)

                    forecast_out, class_out = model(src)

                    val_loss_f += loss_forecast(forecast_out, trg_forecast).item()
                    val_loss_c += loss_class(class_out, torch.argmax(trg_class, dim=1)).item()
                    val_mpjpe += mpjpe(forecast_out, trg_forecast).item()

                    pred = torch.argmax(class_out, dim=1)
                    val_correct += (pred == torch.argmax(trg_class, dim=1)).sum().item()
                    val_total += trg_class.size(0)

            val_loss_f /= len(val_loader)
            val_loss_c /= len(val_loader)
            val_mpjpe /= len(val_loader)
            val_acc = val_correct / val_total

            print(f"[Epoch {epoch+1}] Val → Forecast: {val_loss_f:.4f} | "
                  f"Class: {val_loss_c:.4f} | MPJPE: {val_mpjpe:.4f} | Acc: {val_acc:.4f}")

        # Visualize predictions from current batch
        # if epoch % 1 == 0:
        #     visualize_forecast(src, forecast_out, trg_forecast, step_name=f"Epoch_{epoch+1}", model_name=model_name)
   
# -------------------------
# Run Training
# -------------------------
if __name__ == "__main__":
    import os
    import pickle
    from utils.dataset import PoseDataset

    # Load full dataset
    with open(os.path.join(os.path.dirname(__file__), "dataset.pkl"), "rb") as f:
        raw_data = pickle.load(f)

    full_dataset = PoseDataset(raw_data)

    k = 5  # Number of folds
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(full_dataset)):
        print(f"\n[INFO] Fold {fold+1}/{k}")

        train_subset = torch.utils.data.Subset(full_dataset, train_idx)
        val_subset = torch.utils.data.Subset(full_dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)

        # Build model
        if model_name == "mlp":
            from models.mlp import MLP
            model = MLP(
                input_size=len(raw_data["src"][0]) * 34,
                hidden_size=512,
                forecast_window=len(raw_data["src"][0]),
                output_class_size=2
            )
        elif model_name == "rnn":
            from models.rnn import RNNModel
            model = RNNModel(
                input_size=34,
                hidden_size=128,
                forecast_window=len(raw_data["src"][0]),
                output_class_size=2
            )
        elif model_name == "lstm":
            from models.lstm import LSTMModel
            model = LSTMModel(
                input_size=34,
                hidden_size=128,
                forecast_window=len(raw_data["src"][0]),
                output_class_size=2
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")

        print(f"Training Fold {fold+1}")
        train(model, train_loader, val_loader, num_epochs=100, lr=1e-3, model_name=f"{model_name}_fold{fold+1}")
