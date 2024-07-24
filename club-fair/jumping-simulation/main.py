"""

How to run:

1. Install the necessary packages:
    a. pip install opencv-python
    b. pip install mediapipe
    c. pip install pyautogui

2. python main.py

"""

# Import necessary packages
from utils import *
import pyautogui
import cv2
import mediapipe as mp

# Load mediapipe solutions
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.25, min_tracking_confidence=0.25)

# Open webcam
try:
    # Reminder: use same wifi as phone to computer
    # We use IP Web Cam to run this
    cap = cv2.VideoCapture("https://192.168.1.12:8080/video")
except:
    # Just get webcam feed
    cap = cv2.VideoCapture(0)

# Global variables to keep track of if you are jumping or not
isJumping = False
yJump = 100

# Process if person is jumping or not, and do the necessary actions
def processJump(landmarks, frame):

    global isJumping
    global yJump

    # Stores where eyes are
    eyes = []

    for i in [1, 3]:
        landmark = landmarks[i]
        print(LANDMARK_NAMES[i], 
                ":", 
                lm:=landmark_to_image_coords(landmark, frame)
             )
        draw_heart(frame, lm, 1, (0, 0, 255))
        eyes.append(lm[1])

    if max(eyes) < yJump and not isJumping: # High up in the air!!
        isJumping = True
        pyautogui.press("space")
    elif max(eyes) > yJump: # Fell back to the ground
        isJumping = False

    print("-----")

while cap.isOpened():    
    
    # Read frame from live video feed
    _, frame = cap.read()
    frameShape = frame.shape

    # Change the color channel to match BlazePose's specifications
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_results = pose.process(frame_rgb)

    # Draw line threshold
    frame = cv2.line(frame, (0, 60), (frameShape[1], 60), (0, 0, 0), 2)

    # If detected a person, process the jump
    if pose_results.pose_landmarks:
        processJump(pose_results.pose_landmarks.landmark, frame)

    # Flip the image
    frame = cv2.flip(frame, 1)

    # Show the output
    cv2.imshow('Output', frame)
        
    # If you press q, quit the program
    if cv2.waitKey(1) == ord('q'):
        break

# Destroy windows and end the program
cap.release()
cv2.destroyAllWindows()