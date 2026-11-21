import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import joblib
import time
import threading

model = joblib.load("skipping_model.pkl")

base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO
)
detector = vision.PoseLandmarker.create_from_options(options)

KEY_JOINTS = [11, 12, 15, 16, 23, 24, 27, 28]
WINDOW_SIZE = 15
MAX_GRACE_FRAMES = 10

frame_buffer = []
hip_y_buffer = []
hip_x_buffer = []
torso_buffer = []

def play_finish_sound():
    def _beep():
        try:
            import winsound
            winsound.Beep(1000, 250)
            winsound.Beep(1300, 400)
        except Exception:
            pass
    threading.Thread(target=_beep, daemon=True).start()

def check_upright_posture(landmarks):
    for idx in [11, 12, 23, 24]:
        vis = getattr(landmarks[idx], 'visibility', 1.0)
        if vis is not None and vis < 0.45:
            return False, "Bahu/Pinggul Tidak Terlihat"

    shoulder_x = (landmarks[11].x + landmarks[12].x) / 2
    shoulder_y = (landmarks[11].y + landmarks[12].y) / 2
    hip_x = (landmarks[23].x + landmarks[24].x) / 2
    hip_y = (landmarks[23].y + landmarks[24].y) / 2

    dy_torso = hip_y - shoulder_y
    dx_torso = abs(hip_x - shoulder_x)

    if dy_torso <= 0 or dy_torso < dx_torso * 1.4:
        return False, "Posisi Tiduran / Tidak Tegak"

    return True, "Berdiri Tegak"

def get_landmarks_data(landmarks):
    shoulder_x = (landmarks[11].x + landmarks[12].x) / 2
    shoulder_y = (landmarks[11].y + landmarks[12].y) / 2
    hip_x = (landmarks[23].x + landmarks[24].x) / 2
    hip_y = (landmarks[23].y + landmarks[24].y) / 2

    torso_size = np.hypot(shoulder_x - hip_x, shoulder_y - hip_y) + 1e-6

    coords = []
    for idx in KEY_JOINTS:
        coords.extend([
            (landmarks[idx].x - hip_x) / torso_size,
            (landmarks[idx].y - hip_y) / torso_size
        ])

    return np.array(coords), hip_x, hip_y, torso_size

target_seconds = max(1, int(input("Set durasi skipping (detik): ")))
remaining_time = float(target_seconds)

print("Membuka kamera...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Kamera gagal dibuka! Pastikan tidak sedang dipakai aplikasi lain atau restart terminal.")
    detector.close()
    exit()

# Paksa jendela kamera muncul di paling depan
window_name = "AI Smart Skipping Timer"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

prev_time = time.time()
last_timestamp_ms = 0

is_skipping = False
finished_played = False
grace_counter = 0
lost_frames = 0

prob_skipping = 0.0
hip_bounce = 0.0
reversals = 0
posture_info = "Menunggu..."

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame dari kamera.")
        break

    h_orig, w_orig, _ = frame.shape
    if w_orig > 800:
        scale = 800 / w_orig
        frame = cv2.resize(frame, (800, int(h_orig * scale)))

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    timestamp_ms = int(time.monotonic() * 1000)
    if timestamp_ms <= last_timestamp_ms:
        timestamp_ms = last_timestamp_ms + 1
    last_timestamp_ms = timestamp_ms

    res = detector.detect_for_video(mp_image, timestamp_ms)

    now = time.time()
    dt = now - prev_time
    prev_time = now

    if res.pose_landmarks and len(res.pose_landmarks) > 0:
        landmarks = res.pose_landmarks[0]

        for idx in KEY_JOINTS:
            vis = getattr(landmarks[idx], 'visibility', 1.0)
            if vis is None or vis >= 0.45:
                cx, cy = int(landmarks[idx].x * w), int(landmarks[idx].y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)

        upright, posture_info = check_upright_posture(landmarks)

        if upright:
            lost_frames = 0
            norm_coords, raw_hip_x, raw_hip_y, torso_size = get_landmarks_data(landmarks)
            frame_buffer.append(norm_coords)
            hip_y_buffer.append(raw_hip_y)
            hip_x_buffer.append(raw_hip_x)
            torso_buffer.append(torso_size)

            if len(frame_buffer) == WINDOW_SIZE:
                window_arr = np.array(frame_buffer)
                features = np.concatenate([
                    np.std(window_arr, axis=0),
                    np.ptp(window_arr, axis=0)
                ]).reshape(1, -1)

                prob_skipping = model.predict_proba(features)[0][1]

                hip_y_arr = np.array(hip_y_buffer)
                hip_x_arr = np.array(hip_x_buffer)
                torso_arr = np.array(torso_buffer)
                mean_torso = np.mean(torso_arr) + 1e-6

                hip_bounce = np.ptp(hip_y_arr) / mean_torso

                zoom_ratio = np.ptp(torso_arr) / mean_torso
                x_drift = np.ptp(hip_x_arr) / mean_torso
                is_walking = (zoom_ratio > 0.14) or (x_drift > 0.25)

                dy = np.diff(hip_y_arr)
                significant_dy = dy[np.abs(dy) > 0.0015]
                if len(significant_dy) >= 3:
                    reversals = int(np.sum(np.diff(np.sign(significant_dy)) != 0))
                else:
                    reversals = 0

                recent_bounce = np.ptp(hip_y_arr[-5:]) / mean_torso

                if is_walking:
                    posture_info = "Berjalan / Bergeser"
                    is_skipping = False
                    grace_counter = 0
                elif recent_bounce < 0.012:
                    is_skipping = False
                    grace_counter = 0
                else:
                    if not is_skipping:
                        if (prob_skipping >= 0.68) and (hip_bounce >= 0.025) and (reversals >= 2):
                            is_skipping = True
                            grace_counter = MAX_GRACE_FRAMES
                    else:
                        if (prob_skipping >= 0.50) and (hip_bounce >= 0.018) and (reversals >= 2):
                            grace_counter = MAX_GRACE_FRAMES
                        else:
                            grace_counter -= 1
                            if grace_counter <= 0:
                                is_skipping = False

                frame_buffer.pop(0)
                hip_y_buffer.pop(0)
                hip_x_buffer.pop(0)
                torso_buffer.pop(0)
        else:
            lost_frames += 1
            grace_counter -= 1
            if grace_counter <= 0:
                is_skipping = False
            if lost_frames > 10:
                prob_skipping = 0.0
                hip_bounce = 0.0
                reversals = 0
                frame_buffer.clear()
                hip_y_buffer.clear()
                hip_x_buffer.clear()
                torso_buffer.clear()
    else:
        lost_frames += 1
        grace_counter -= 1
        posture_info = "Orang Tidak Terdeteksi"
        if grace_counter <= 0:
            is_skipping = False
        if lost_frames > 10:
            prob_skipping = 0.0
            hip_bounce = 0.0
            reversals = 0
            frame_buffer.clear()
            hip_y_buffer.clear()
            hip_x_buffer.clear()
            torso_buffer.clear()

    if is_skipping and remaining_time > 0:
        remaining_time = max(0.0, remaining_time - dt)
        status_text = "SKIPPING (TIMER JALAN)"
        color = (0, 255, 100)
    elif remaining_time == 0:
        status_text = "SELESAI!"
        color = (0, 220, 255)
        if not finished_played:
            play_finish_sound()
            finished_played = True
    else:
        status_text = "BERHENTI (TIMER PAUSE)"
        color = (80, 80, 255)

    overlay = frame.copy()
    cv2.rectangle(overlay, (15, 15), (490, 185), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    cv2.putText(frame, f"Waktu: {remaining_time:.1f} dtk", (28, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(frame, status_text, (28, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, f"Postur: {posture_info}", (28, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (120, 220, 255), 1)
    cv2.putText(frame, f"AI: {prob_skipping*100:.0f}% | Pantulan: {hip_bounce:.3f} | Ritme: {reversals}", (28, 143), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1)

    progress = remaining_time / float(target_seconds)
    bar_width = int(434 * progress)
    cv2.rectangle(frame, (28, 160), (462, 172), (60, 60, 60), -1)
    cv2.rectangle(frame, (28, 160), (28 + bar_width, 172), color, -1)

    cv2.imshow(window_name, frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        remaining_time = float(target_seconds)
        finished_played = False
    elif key == ord('='):
        remaining_time += 15
        finished_played = False
    elif key == ord('-'):
        remaining_time = max(0, remaining_time - 15)

detector.close()
cap.release()
cv2.destroyAllWindows()