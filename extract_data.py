import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pandas as pd
import numpy as np
import os

# Inisialisasi MediaPipe Tasks API yang baru
base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE
)
detector = vision.PoseLandmarker.create_from_options(options)

KEY_JOINTS = [11, 12, 15, 16, 23, 24, 27, 28]
WINDOW_SIZE = 15

def get_normalized_landmarks(landmarks):
    hip_x = (landmarks[23].x + landmarks[24].x) / 2
    hip_y = (landmarks[23].y + landmarks[24].y) / 2
    shoulder_y = (landmarks[11].y + landmarks[12].y) / 2
    torso_size = abs(shoulder_y - hip_y) + 1e-6

    coords = []
    for idx in KEY_JOINTS:
        norm_x = (landmarks[idx].x - hip_x) / torso_size
        norm_y = (landmarks[idx].y - hip_y) / torso_size
        coords.extend([norm_x, norm_y])
    return np.array(coords)

data_rows = []
categories = {"not_skipping": 0, "skipping": 1}

for folder_name, label in categories.items():
    folder_path = os.path.join("videos", folder_name)
    for video_file in os.listdir(folder_path):
        if not video_file.lower().endswith((".mp4", ".avi", ".mov")):
            continue
        
        cap = cv2.VideoCapture(os.path.join(folder_path, video_file))
        frame_buffer = []
        print(f"Memproses: {folder_name}/{video_file}...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = detector.detect(mp_image)

            if res.pose_landmarks and len(res.pose_landmarks) > 0:
                norm_coords = get_normalized_landmarks(res.pose_landmarks[0])
                frame_buffer.append(norm_coords)

                if len(frame_buffer) == WINDOW_SIZE:
                    window_arr = np.array(frame_buffer)
                    std_dev = np.std(window_arr, axis=0)
                    ptp_range = np.ptp(window_arr, axis=0)
                    
                    features = np.concatenate([std_dev, ptp_range, [label]])
                    data_rows.append(features)
                    frame_buffer.pop(0)

        cap.release()

df = pd.DataFrame(data_rows)
df.to_csv("dataset.csv", index=False)
print(f"Selesai! Total {len(df)} baris data tersimpan di dataset.csv")