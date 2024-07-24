"""

How to run:

1. Install the necessary packages:
    a. pip install opencv-python
    b. pip install mediapipe

2. python main.py

"""

# Import necessary packages
import cv2
import mediapipe as mp

# Load mediapipe solutions
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.25, min_tracking_confidence=0.25)

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():

    # Read frame from live video feed
    _, frame = cap.read()
    frameShape = frame.shape

    # Change the color channel to match BlazePose's specifications
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_results = pose.process(frame_rgb)
    
    # Draw where joints are
    mp_drawing.draw_landmarks(
        frame, 
        pose_results.pose_landmarks, 
        mp_pose.POSE_CONNECTIONS
    )

    # Flip the Frame
    frame = cv2.flip(frame, 1)

    # Show it in OpenCV
    cv2.imshow('Output', frame)
        
    # If you press q, quit the program
    if cv2.waitKey(1) == ord('q'):
        break

# Destroy windows and end the program
cap.release()
cv2.destroyAllWindows()