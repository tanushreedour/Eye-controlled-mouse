import cv2 as cv
import mediapipe as mp
import pyautogui as auto
import numpy as np 
import matplotlib.pyplot as plt
import itertools

cam = cv.VideoCapture(0)

mp_face_mesh = mp.solutions.face_mesh

mp_face_mesh_images = mp.solutions.face_mesh.FaceMesh(static_image_mode = True, max_num_faces = 1, min_detection_confidence = 0.9, min_tracking_confidence = 0.6, refine_landmarks = True) 

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

screen_width, screen_height = auto.size()

while True :
    isTrue, frame = cam.read()

    frame = cv.flip(frame, 1)  # X-axis : 0  and  Y-axis: 1

    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    output = mp_face_mesh_images.process(rgb_frame)

    landmark_points = output.multi_face_landmarks

    # print(landmark_points) :- This will print all the landmarks in the form of dictionary, the values of X, Y, Z axes of eye balls in the frame 
    
    frame_height, frame_width, _ = frame.shape

    if landmark_points:
        landmarks = landmark_points[0].landmark

        if landmarks[473]:
            landmark = landmarks[473]

            x = int(landmark.x * frame_width)
            y = int(landmark.y * frame_height)
            cv.circle(frame, (x, y), 3, (0, 255, 0), 2)
            screen_x = screen_width * landmark.x
            screen_y = screen_height * landmark.y
            auto.moveTo(screen_x, screen_y)
        left = [landmarks[145], landmarks[159]]
        for landmark in left:
            x = int(landmark.x * frame_width)
            y = int(landmark.y * frame_height)
            cv.circle(frame, (x, y), 3, (0, 255, 255))
        if (left[0].y - left[1].y) < 0.004:
            auto.click()
            auto.sleep(1)

    cv.imshow('Eye Controlled Mouse', frame)
    cv.waitKey(1)