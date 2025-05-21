import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
from win32com.client import Dispatch
from sklearn.neighbors import KNeighborsClassifier

# Function to speak text
def speak(text):
    speaker = Dispatch("SAPI.SpVoice")
    speaker.Speak(text)

# Initialize video capture and face detection
video = cv2.VideoCapture(0)
if not video.isOpened():
    raise Exception("Camera not found or not accessible.")

facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
if facedetect.empty():
    raise FileNotFoundError("Haarcascade file 'haarcascade_frontalface_default.xml' not found.")

# Load trained labels and face data
with open('data/names.pkl', 'rb') as w:
    LABELS = pickle.load(w)
with open('data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

# Validate data consistency
if len(LABELS) != FACES.shape[0]:
    raise ValueError("The number of faces and labels do not match!")

print("Length of LABELS:", len(LABELS))
print("Shape of FACES:", FACES.shape)

# Initialize KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

# Load background image and ensure its size
imgBackground = cv2.imread("bg1.png")
imgBackground = cv2.resize(imgBackground, (695, 642))  # Adjust to required size
COL_NAMES = ['NAME', 'TIME']

# Define the correct face size used for training
face_size = (50, 50)  # Match training size

# Start the video feed loop
while True:
    ret, frame = video.read()
    if not ret:
        print("Failed to capture a frame. Exiting...")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Crop and resize the detected face to match the training data size (50x50)
        crop_img = frame[y:y + h, x:x + w]
        resized_img = cv2.resize(crop_img, dsize=face_size).flatten().reshape(1, -1)

        # Predict the label using KNN
        output = knn.predict(resized_img)

        # Get timestamp
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")

        # Check if the attendance file exists
        os.makedirs("Attendance", exist_ok=True)
        attendance_file = f"Attendance/Attendance_{date}.csv"
        exist = os.path.isfile(attendance_file)

        # Draw rectangles and labels around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
        cv2.putText(frame, output[0], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Prepare attendance record
        attendance = [str(output[0]), str(timestamp)]

        # Update background with frame
        imgBackground[162:162 + 480, 55:55 + 640] = cv2.resize(frame, (640, 480))
        cv2.imshow("Frame", imgBackground)

    # Handle keyboard inputs
    k = cv2.waitKey(1)
    if k == ord('o'):
        speak("Attendance Taken.")
        time.sleep(2)
        with open(attendance_file, "a" if exist else "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if not exist:
                writer.writerow(COL_NAMES)
            writer.writerow(attendance)

    if k == ord('q'):
        print("Exiting program.")
        break

# Release video capture and close all windows
video.release()
cv2.destroyAllWindows()

