import requests
import cv2
import numpy as np


URL = "https://192.168.1.5:8080/shot.jpg"
while True:
    img_resp = requests.get(URL)
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    img = cv2.imdecode(img_arr, -1)
    cv2.imshow('IPWebcam', img)
    height, width, channels = img.shape
    print(height, width, channels)

    if cv2.waitKey(1) == 27:
        break