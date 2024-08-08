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

# TODO: implement up, down, left, right

"""

GLOBAL VARIABLES

"""

pyautogui.PAUSE = 0.02

WIDTH, HEIGHT = getScreenSize()
WIDTH //= 2

xLeft = 0
xRight = 0
yUp = 0
yDown = 0

paddingX = 250
paddingY = 140

pressUp = False
pressDown = False
pressLeft = False
pressRight = False

showSkeleton = False
tutorial = False
setup = True

# Process if person is jumping or not, and do the necessary actions
def processFrame(landmarks, frame, currFrame):

    global xLeft
    global xRight
    global yUp
    global yDown

    global pressUp
    global pressDown
    global pressLeft
    global pressRight

    # Stores where eyes are
    eyes = []

    for i in [3, 6]:

        landmark = landmarks[i]
        lm = landmark_to_image_coords(landmark, frame)
        # print(landmark.z)
        zLandmark = abs(landmark.z)
        drawEyes(frame, lm, int(15 * zLandmark))

        eyes.append(lm)

    # maximum out of y-values of eyes
    bestEye = eyes[0]

    resString = ""

    if currFrame % 1 == 0:

        bestEye = min(eyes, key = lambda x: [x[1], x[0]])
        if bestEye[1] <= yUp and not pressUp:
            pyautogui.press("up")
            pressUp = True
        elif bestEye[1] > yUp:
            pressUp = False
        
        bestEye = max(eyes, key = lambda x: [x[1], x[0]])
        if bestEye[1] >= yDown and not pressDown:
            pyautogui.press("down")
            pressDown = True
        elif bestEye[1] < yDown:
            pressDown = False
        
        bestEye = min(eyes)
        if bestEye[0] <= xLeft and not pressLeft:
            pyautogui.press("right")
            pressRight = True
        elif bestEye[0] > xLeft:
            # pyautogui.keyUp("left")
            pressRight = False

        bestEye = max(eyes)
        if bestEye[0] >= xRight and not pressRight:
            pyautogui.press("left")
            pressLeft = True
        elif bestEye[0] < xRight:
            # pyautogui.keyUp("right")
            pressLeft = False

    if pressUp:
        resString += "UP "
    
    if pressDown:
        resString += "DOWN "

    if pressLeft:
        resString += "LEFT "
    
    if pressRight:
        resString += "RIGHT "

    return resString

"""

Start recording person 

"""

# Make OpenCV window work 
cv2.namedWindow("Output", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Output", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Load mediapipe solutions
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.25, min_tracking_confidence=0.25)

# Get webcam
cap = cv2.VideoCapture(0)

setupsetupsetup = True

currFrame = 0

while cap.isOpened():    
    
    # Read frame from live video feed
    _, frame = cap.read()
    
    # Resize image
    frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
    
    currFrame += 1

    if setupsetupsetup:
        setupsetupsetup = False
        
        yUp = int((HEIGHT-frame.shape[0])/2) + paddingY
        yDown = int((HEIGHT-frame.shape[0])/2) + frame.shape[0] - paddingY
        xLeft = int((WIDTH-frame.shape[1])/2) + paddingX
        xRight = int((WIDTH-frame.shape[1])/2) + frame.shape[1] - paddingX

    frame = cv2.copyMakeBorder(
        frame, 
        int((HEIGHT-frame.shape[0])/2), 
        int((HEIGHT-frame.shape[0])/2), 
        int((WIDTH-frame.shape[1])/2), 
        int((WIDTH-frame.shape[1])/2), 
    0)

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

    yessir = ""
    # If detected a person, process the jump
    if pose_results.pose_landmarks:
        yessir = processFrame(pose_results.pose_landmarks.landmark, frame, currFrame)

    # Flip the image
    frame = cv2.flip(frame, 1)

    # Draw line threshold
    if tutorial:
        frame = cv2.line(frame, (0, yUp), (5000, yUp), (250, 250, 0), 2)
        frame = cv2.line(frame, (0, yDown), (5000, yDown), (0, 0, 256), 2)
        frame = cv2.line(frame, (xLeft, 0), (xLeft, 5000), (256, 0, 0), 2)
        frame = cv2.line(frame, (xRight, 0), (xRight, 5000), (0, 256, 0), 2)



    if yessir:
        # font 
        font = cv2.FONT_HERSHEY_SIMPLEX 
        # org 
        org = (50, 50) 
        # fontScale 
        fontScale = 1
        # Blue color in BGR 
        color = (255, 255, 255) 
        # Line thickness of 2 px 
        thickness = 2
        # Using cv2.putText() method 
        image = cv2.putText(frame, yessir, org, font,  
                        fontScale, color, thickness, cv2.LINE_AA) 



    # Show the output
    cv2.imshow('Output', frame)
        
    # If you press q, quit the program

    waitedKey = cv2.waitKey(1)

    if waitedKey == ord('q'):
        break
    elif waitedKey == ord('e'):
        showSkeleton = not showSkeleton
    elif waitedKey == ord('t'):
        tutorial = not tutorial

# Destroy windows and end the program
cap.release()
cv2.destroyAllWindows()