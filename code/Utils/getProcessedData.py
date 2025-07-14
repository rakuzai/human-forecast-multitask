import numpy as np
import cv2 as cv
import torch
import os
import matplotlib.pyplot as plt

from ultralytics import YOLO
from torch.utils.data import Dataset

class getData(Dataset):
    def __init__(self, folder="../../TelUP_HumanFallDataset/", input_window=15, output_window=15, step=5):
        self.src, self.trg_forecast, self.trg_class = [], [], []
        to_onehot = np.eye(2)
        missing_frame_dir = "missing_frames"
        os.makedirs(missing_frame_dir, exist_ok=True)

        model = YOLO('../yolov8n-pose.pt')
        for i, class_name in enumerate(os.listdir(folder)):  # loop class fall non fall
            for j, subject_name in enumerate(os.listdir(folder + class_name)):  # loop subject
                for k, video_file in enumerate(os.listdir(folder + class_name + "/" + subject_name)):  # loop video
                    # video to frames
                    frames = self.extract_frames_from_video(folder + class_name + "/" + subject_name + "/" + video_file)

                    # 0 = non fall
                    # 1 = fall
                    # frames to keypoints
                    keypoints_in_video = []
                    
                    for frame in frames:  # loop frames
                        keypoints = []
                        output = model(frame, save=False, verbose=False)
                        if output and output[0].keypoints is not None and len(output[0].keypoints) > 0:
                            # print keypoints index number and x,y coordinates
                            print("berhasil = ", output)
                            for idx, kpt in enumerate(output[0].keypoints[0]):
                                for m in kpt.data[0][:, 0:2].cpu().detach().numpy():
                                    m = [x for x in m]
                                    keypoints.append([m[0] / 500, m[1] / 500])
                        else:
                            print(f'❌ No Keypoints in Frame {idx}')
                            # 🔽 Save missing frame plot 🔽
                            plt.figure(figsize=(4, 4))
                            plt.imshow(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
                            plt.title(f"Missing Keypoints - Frame {idx}")
                            plt.axis("off")
                            save_path = os.path.join(missing_frame_dir, f"missing_frame_{idx:04d}.png")
                            plt.savefig(save_path)
                            plt.close()
                        keypoints_in_video.append(keypoints)

                    # sliding window

                    for slider in range(0, len(keypoints_in_video), step):
                        self.src.append(keypoints_in_video[slider:slider + input_window])
                        self.trg_forecast.append(
                            keypoints_in_video[slider + input_window:slider + input_window + output_window])
                        self.trg_class.append(
                            keypoints_in_video[slider + input_window:slider + input_window + output_window])

    def __len__(self):
        return len(self.src)

    def __getitem__(self, item):
        return (torch.tensor(self.src[item], dtype=torch.float32),
                (torch.tensor(self.trg_forecast[item], dtype=torch.float32),
                 torch.tensor(self.trg_class[item], dtype=torch.float32)))

    def extract_frames_from_video(self, video_path):
        frames = []
        cap = cv.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}")
            pass

        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # frame_filename = os.path.join(output_dir, f"frame_{frame_idx:04d}.png")
            frame = cv.resize(frame, (500, 500))
            frames.append(frame)
            frame_idx += 1

        return frames


if __name__ == "__main__":
    dataset = getData()