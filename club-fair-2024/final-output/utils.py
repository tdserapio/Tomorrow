import numpy as np
import cv2
from screeninfo import get_monitors

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

def drawEyes(image, center, size):

    radius = int(size)
    border_thickness = int(size * 1.2)

    cv2.circle(image, center, radius + border_thickness, (255, 255, 255), -1)
    cv2.circle(image, center, radius, (0, 0, 0), -1)

def getScreenSize(ind = 0):
    monitors = get_monitors()
    return (monitors[ind].width, monitors[ind].height)