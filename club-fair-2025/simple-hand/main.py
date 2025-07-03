import time
import mediapipe as mp
import cv2
import numpy as np

mp_drawing         = mp.solutions.drawing_utils
BaseOptions        = mp.tasks.BaseOptions
HandLandmarker     = mp.tasks.vision.HandLandmarker
HandLandmarkerOpts = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerRes  = mp.tasks.vision.HandLandmarkerResult
RunningMode        = mp.tasks.vision.RunningMode
Connections        = mp.solutions.hands.HAND_CONNECTIONS

options = HandLandmarkerOpts(
    base_options = BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode = RunningMode.VIDEO
)

with HandLandmarker.create_from_options(options) as landmarker:

    cap         = cv2.VideoCapture(0)
    t0          = time.time()                    # reference point

    while cap.isOpened():

        ok, frame = cap.read()

        if not ok:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        res = landmarker.detect_for_video(mp_image, int((time.time() - t0) * 1000))
        rgb = frame_rgb.copy()

        if res.hand_landmarks:

            h, w, _ = rgb.shape
            for hand in res.hand_landmarks:
                
                for lm_pt in hand:
                    cv2.circle(
                        rgb,
                        (int(lm_pt.x * w), int(lm_pt.y * h)),
                        8, (0, 255, 0), -1
                    )

                for start_idx, end_idx in Connections:
                    start = hand[start_idx]
                    end   = hand[end_idx]
                    cv2.line(
                        rgb,
                        (int(start.x*w), int(start.y*h)),
                        (int(end.x*w),   int(end.y*h)),
                        (255, 0, 0),
                        2
                    )

        frame_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imshow("Hand Landmarks", frame_bgr)
        if cv2.waitKey(1) & 0xFF == 27:   # ESC to quit
            break

    cap.release()
    cv2.destroyAllWindows()
