"""

How to run:

1. Install the necessary packages:
    a. pip install opencv-python
    b. pip install mediapipe
    c. pip install pyautogui
    d. pip install screeninfo

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

# # Open webcam
# try:
#     # Reminder: use same wifi as phone to computer
#     # We use IP Web Cam to run this
#     cap = cv2.VideoCapture("http://192.168.1.21:8080/video")
# except:
#     # Just get webcam feed
cap = cv2.VideoCapture(0)

# Global variables to keep track of if you are jumping or not
isJumping = False
yJump = 100
showSkeleton = False

# Process if person is jumping or not, and do the necessary actions
def processJump(landmarks, frame):

    global isJumping
    global yJump

    # Stores where eyes are
    eyes = []

    for i in [3, 6]:
        landmark = landmarks[i]
        lm = landmark_to_image_coords(landmark, frame)

        drawEyes(frame, lm, 9)

        eyes.append(lm[1])

    if max(eyes) < yJump and not isJumping: # High up in the air!!
        isJumping = True
        pyautogui.press("space")
    elif max(eyes) > yJump: # Fell back to the ground
        isJumping = False



# Make OpenCV Camera work
cv2.namedWindow("Output", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Output", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while cap.isOpened():    
    
    # Read frame from live video feed
    _, frame = cap.read()
    frameShape = frame.shape

    # Change the color channel to match BlazePose's specifications
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_results = pose.process(frame_rgb)

    # Draw where joints are
    if showSkeleton:
        mp_drawing.draw_landmarks(
            frame, 
            pose_results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec = mp_drawing.DrawingSpec(color=(69, 40, 27), thickness=2, circle_radius=6),
            connection_drawing_spec = mp_drawing.DrawingSpec(color=(209, 204, 165), thickness=4, circle_radius=6)
        )

    # Draw line threshold
    frame = cv2.line(frame, (0, yJump), (frameShape[1], yJump), (60, 60, 60), 20)

    # If detected a person, process the jump
    if pose_results.pose_landmarks:
        processJump(pose_results.pose_landmarks.landmark, frame)

    # Flip the image
    frame = cv2.flip(frame, 1)

    # Resize image

    WIDTH, HEIGHT = getScreenSize()

    frame = cv2.copyMakeBorder(
        frame, 
        int((HEIGHT-frame.shape[0])/2), 
        int((HEIGHT-frame.shape[0])/2), 
        int((WIDTH-frame.shape[1])/2), 
        int((WIDTH-frame.shape[1])/2), 
    0)

    # Show the output
    cv2.imshow('Output', frame)
        
    # If you press q, quit the program

    waitedKey = cv2.waitKey(1)

    if waitedKey == ord('q'):
        break
    elif waitedKey == ord('d'):
        showSkeleton = not showSkeleton
    elif waitedKey == ord('w'):
        yJump -= 5
    elif waitedKey == ord('s'):
        yJump += 5

    # print(waitedKey)

# Destroy windows and end the program
cap.release()
cv2.destroyAllWindows()