import numpy as np
import cv2

LANDMARK_NAMES = {
    0: "nose",
    1: "right eye inner",
    2: "right eye",
    3: "right eye outer",
    4: "left eye inner",
    5: "left eye",
    6: "left eye outer",
    7: "right ear",
    8: "left ear",
    9: "mouth right",
    10: "mouth left",
    11: "right shoulder",
    12: "left shoulder",
    13: "right elbow",
    14: "left elbow",
    15: "right wrist",
    16: "left wrist",
    17: "right pinky knuckle #1",
    18: "left pinky knuckle #1",
    19: "right index knuclke #1",
    20: "left index knuckle #1",
    21: "right thumb knuckle #2",
    22: "left thumb knuckle #2",
    23: "right hip",
    24: "left hip",
    25: "right knee",
    26: "left knee",
    27: "right ankle",
    28: "left ankle",
    29: "right heel",
    30: "left heel",
    31: "right foot index",
    32: "left foot index"
}

INDEX_LANDMARK = {LANDMARK_NAMES[i]: i for i in LANDMARK_NAMES}

def landmark_to_image_coords(landmark, frame):
    return (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]))

def draw_heart(image, center, size, color):
    t = np.arange(0, 2*np.pi, 0.1)
    x = size * (16*np.sin(t)**3)
    y = size * (13*np.cos(t) - 5*np.cos(2*t) - 2*np.cos(3*t) - np.cos(4*t))
    x = x + center[0]
    y = center[1] - y
    x = x.astype(int)
    y = y.astype(int)
    for i in range(len(x) - 1):
        cv2.line(image, (x[i], y[i]), (x[i+1], y[i+1]), color, 2)