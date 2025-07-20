import sys
import os
import torch
import pickle
import numpy as np
from torch.utils.data import DataLoader
from train import mpjpe
from utils.dataset import PoseDataset

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.lstm import LSTMModel  # Adjust if you're using another model

# Your existing skeleton edges
SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),     # Head connections
    (5, 6),                              # Shoulder to shoulder
    (5, 11), (6, 12), (11, 12),         # Torso
    (5, 7), (7, 9),                     # Left arm
    (6, 8), (8, 10),                    # Right arm  
    (11, 13), (13, 15),                 # Left leg
    (12, 14), (14, 16),                 # Right leg
    (0, 5), (0, 6)                      # Head to torso
]

def draw_pose(ax, pose, color, alpha=1.0, linewidth=2):
    """Draw pose skeleton with proper connections"""
    for (i, j) in SKELETON_EDGES:
        if i < pose.shape[0] and j < pose.shape[0]:
            if not (np.allclose(pose[i], 0) or np.allclose(pose[j], 0)):
                ax.plot([pose[i, 0], pose[j, 0]], 
                       [-pose[i, 1], -pose[j, 1]], 
                       color=color, linewidth=linewidth, alpha=alpha)
    
    ax.scatter(pose[:, 0], -pose[:, 1], 
              c=color, s=30, alpha=alpha, 
              edgecolors='white', linewidth=0.5, zorder=10)

def visualize_cross_windows(pred_batch, tgt_batch, model_name, batch_idx=0, 
                           start_sample=0, frame_stride=3, num_frames=8):
    os.makedirs("../result", exist_ok=True)
    
    batch_size, seq_len, pose_dim = pred_batch.shape  # (32, 15, 34)
    
    # Calculate which sliding windows and frames we need
    selected_frames = []
    selected_windows = []
    current_sample = start_sample
    current_frame = 0
    
    frame_count = 0
    while frame_count < num_frames and current_sample < batch_size:
        if current_frame < seq_len:
            selected_windows.append(current_sample)
            selected_frames.append(current_frame)
            current_frame += frame_stride
            frame_count += 1
        else:
            # Move to next sliding window and reset frame counter
            current_sample += 1
            current_frame = current_frame - seq_len  # Continue the stride pattern
            if current_frame < 0:
                current_frame = 0
    
    print(f"[INFO] Using {len(selected_frames)} frames from windows {selected_windows} at frames {selected_frames}")
    
    # Create visualization
    fig, axs = plt.subplots(1, len(selected_frames), 
                           figsize=(len(selected_frames) * 3, 4))
    
    if len(selected_frames) == 1:
        axs = [axs]
    
    for i, (window_idx, frame_idx) in enumerate(zip(selected_windows, selected_frames)):
        ax = axs[i]
        
        # Get poses from specific window and frame
        pred_pose = pred_batch[window_idx, frame_idx].reshape(17, 2)
        true_pose = tgt_batch[window_idx, frame_idx].reshape(17, 2)
        
        # Draw poses
        draw_pose(ax, true_pose, color='blue', alpha=0.7, linewidth=3)
        draw_pose(ax, pred_pose, color='red', alpha=0.9, linewidth=2)
        
        # Set axis properties
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f'W{window_idx}_F{frame_idx}', fontsize=10, pad=10)
        
        # Set reasonable axis limits
        all_points = np.concatenate([pred_pose, true_pose], axis=0)
        valid_points = all_points[~np.all(np.isclose(all_points, 0), axis=1)]
        
        if len(valid_points) > 0:
            margin = 0.1
            x_min, x_max = valid_points[:, 0].min(), valid_points[:, 0].max()
            y_min, y_max = -valid_points[:, 1].max(), -valid_points[:, 1].min()
            
            x_range = x_max - x_min
            y_range = y_max - y_min
            
            ax.set_xlim(x_min - margin * x_range, x_max + margin * x_range)
            ax.set_ylim(y_min - margin * y_range, y_max + margin * y_range)
    
    # Add legend
    legend_elements = [
        mpatches.Patch(color='blue', label='Ground Truth', alpha=0.7),
        mpatches.Patch(color='red', label='Prediction', alpha=0.9)
    ]
    fig.legend(handles=legend_elements, loc='upper center', 
              ncol=2, fontsize=12, bbox_to_anchor=(0.5, 0.95))
    
    plt.suptitle(f'Cross-Window Visualization (Stride: {frame_stride})', fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    # Save figure
    save_path = f"../result/cross_window_viz_{model_name}_batch{batch_idx}_stride{frame_stride}.jpg"
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"[INFO] Saved cross-window visualization to {save_path}")


def evaluate(model_path="../models/lstm_500.pt", batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    dataset_path = os.path.join(os.path.dirname(__file__), "dataset.pkl")
    with open(dataset_path, "rb") as f:
        data = pickle.load(f)

    # Select test subjects (last 2)
    subjects = data["subjects"]
    unique_subjects = sorted(set(subjects))
    test_subjects = unique_subjects[-2:]

    test_data = {"src": [], "trg_forecast": [], "trg_class": [], "subjects": []}
    for i, subj in enumerate(subjects):
        if subj in test_subjects:
            test_data["src"].append(data["src"][i])
            test_data["trg_forecast"].append(data["trg_forecast"][i])
            test_data["trg_class"].append(data["trg_class"][i])
            test_data["subjects"].append(subj)

    test_dataset = PoseDataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Infer input_window from data
    input_window = len(test_data["src"][0])
    input_dim = len(test_data["src"][0][0]) * len(test_data["src"][0][0][0])  # 17 * 2 = 34

    # Load model
    model = LSTMModel(
        input_size=34,
        hidden_size=128,
        forecast_window=input_window,
        output_class_size=2
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Evaluation
    loss_forecast = torch.nn.MSELoss()
    loss_class = torch.nn.CrossEntropyLoss()

    total_loss_f = 0.0
    total_loss_c = 0.0
    correct, total = 0, 0
    all_mpjpe = []

    with torch.no_grad():
        for idx, (src, (trg_forecast, trg_class)) in enumerate(test_loader):
            src = src.to(device)
            trg_forecast = trg_forecast.to(device)
            trg_class = trg_class.to(device)

            src = src.view(src.size(0), src.size(1), -1)
            trg_forecast = trg_forecast.view(trg_forecast.size(0), trg_forecast.size(1), -1)

            forecast_out, class_out = model(src)

            loss_f = loss_forecast(forecast_out, trg_forecast)
            loss_c = loss_class(class_out, torch.argmax(trg_class, dim=1))

            total_loss_f += loss_f.item() * src.size(0)
            total_loss_c += loss_c.item() * src.size(0)
            pred = torch.argmax(class_out, dim=1)
            correct += (pred == torch.argmax(trg_class, dim=1)).sum().item()
            total += trg_class.size(0)

            all_mpjpe.append(mpjpe(forecast_out, trg_forecast).item())

            if idx==1:
                print(f"[INFO] Visualizing batch {idx}")
                visualize_cross_windows(
                    forecast_out.cpu().numpy(),   # shape: [batch_size, forecast_window, 34]
                    trg_forecast.cpu().numpy(),   # same shape as forecast_out
                    model_name="lstm",
                    batch_idx=idx,
                    start_sample=0,
                    frame_stride=5,
                    num_frames=8
                )

    avg_loss_f = total_loss_f / total
    avg_loss_c = total_loss_c / total
    avg_mpjpe = sum(all_mpjpe) / len(all_mpjpe)
    acc = correct / total

    print(f"[Evaluation] Forecast Loss: {avg_loss_f:.4f} | Class Loss: {avg_loss_c:.4f} | "
          f"MPJPE: {avg_mpjpe:.4f} | Accuracy: {acc:.4f}")
    

if __name__ == "__main__":
    evaluate()