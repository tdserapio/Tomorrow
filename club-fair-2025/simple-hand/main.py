import time
import mediapipe as mp
import cv2
import numpy as np
import pyautogui

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

    spaceDown   = False
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

        distanceSum = float('inf')

        if res.hand_landmarks:

            distanceSum = 0

            coords = []

            h, w, _ = rgb.shape

            for hand in res.hand_landmarks:

                for lm_pt in hand:
                    cv2.circle(
                        rgb,
                        (int(lm_pt.x * w), int(lm_pt.y * h)),
                        8, (0, 255, 0), -1
                    )
                    coords.append((lm_pt.x, lm_pt.y))

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
            
            for i in coords:
                for j in coords:
                    distanceSum += ((i[0] - j[0])**2 + (i[1] - j[1])**2)**.5

            x,*_ = Connections
            d1, d2 = [hand[x[0]], hand[x[1]]]
            distanceSum = distanceSum / (((d1.x - d2.x)**2 + (d1.y - d2.y)**2)**0.5)

        if distanceSum < 1000 and not spaceDown:
            spaceDown = True
            pyautogui.keyDown("SPACE")
        elif distanceSum >= 1000 and spaceDown:
            spaceDown = False
            pyautogui.keyUp("SPACE")

        frame_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imshow("Hand Landmarks", frame_bgr)
        if cv2.waitKey(1) & 0xFF == 27:   # ESC to quit
            break

    cap.release()
    cv2.destroyAllWindows()
