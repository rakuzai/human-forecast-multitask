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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------
# MPJPE Metric
# -------------------------
def mpjpe(pred, target):
    return torch.mean(torch.norm(pred - target, dim=-1))

model_name = 'lstm'

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
def train(model, dataloader, num_epochs=10, lr=1e-3, model_name=model_name):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_forecast = nn.MSELoss()
    loss_class = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss, total_loss_f, total_loss_c = 0.0, 0.0, 0.0
        correct, total = 0, 0
        all_mpjpe = []

        # loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

        for src, (trg_forecast, trg_class) in dataloader:
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

            # MPJPE
            all_mpjpe.append(mpjpe(forecast_out, trg_forecast).item())

            # Accuracy
            pred = torch.argmax(class_out, dim=1)
            correct += (pred == torch.argmax(trg_class, dim=1)).sum().item()
            total += trg_class.size(0)

            # loop.set_postfix({
            #     "Loss": f"{loss.item():.4f}",
            #     "Acc": f"{(correct / total):.4f}"
            # })

        avg_loss = total_loss / len(dataloader)
        avg_loss_f = total_loss_f / len(dataloader)
        avg_loss_c = total_loss_c / len(dataloader)
        avg_mpjpe = sum(all_mpjpe) / len(all_mpjpe)
        acc = correct / total

        print(f"[Epoch {epoch+1}] Total Loss: {avg_loss:.4f} | Forecast Loss: {avg_loss_f:.4f} | "
              f"Class Loss: {avg_loss_c:.4f} | MPJPE: {avg_mpjpe:.4f} | Accuracy: {acc:.4f}")
        
    if epoch % 1 == 0:
        visualize_forecast(src, forecast_out, trg_forecast, step_name=f"Epoch_{epoch+1}", model_name=model_name)
        
# -------------------------
# Run Training
# -------------------------
if __name__ == "__main__":
    import os
    import pickle
    import torch
    from utils.dataset import PoseDataset
    from torch.utils.data import DataLoader

    # Load preprocessed dataset from utils/dataset.pkl
    with open(os.path.join(os.path.dirname(__file__), "dataset.pkl"), "rb") as f:
        raw_data = pickle.load(f)
        dataset = PoseDataset(raw_data)

    dataloader = DataLoader(dataset, batch_size=32)

    # Extract input dimensions
    input_window = len(raw_data["src"][0])              # e.g. 15
    keypoints_dim = len(raw_data["src"][0][0]) * len(raw_data["src"][0][0][0])  # 17 * 2 = 34

    if model_name == "mlp":
        from models.mlp import MLP
        model = MLP(
            input_size=input_window * keypoints_dim,  # flattened
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

    print(f"Using model: {model_name.upper()} for multi-task learning.")
    from utils.train import train
    train(model, dataloader, num_epochs=500, lr=1e-3, model_name=model_name)
