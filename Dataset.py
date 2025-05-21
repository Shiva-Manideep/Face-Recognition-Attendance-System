import cv2
import pickle
import numpy as np
import os

# Check camera functionality
video = cv2.VideoCapture(0)
if not video.isOpened():
    raise Exception("Camera not found or not accessible.")

# Load face detection model
facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
if facedetect.empty():
    raise FileNotFoundError("Haarcascade file 'haarcascade_frontalface_default.xml' not found.")

faces_data = []
i = 0

name = input("Enter Your Name: ")

while True:
    ret, frame = video.read()
    if not ret:
        print("Failed to capture a frame. Exiting...")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y + h, x:x + w, :]
        resized_img = cv2.resize(crop_img, (50, 50))  # Fixed size (50x50)

        if len(faces_data) < 100 and i % 10 == 0:
            faces_data.append(resized_img)

        i += 1
        cv2.putText(frame, f"Collected: {len(faces_data)}", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)

    cv2.imshow("Frame", frame)

    if len(faces_data) == 100:  # Collect exactly 100 images
        print("Data collection complete.")
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Quit with 'q'
        print("Data collection interrupted by user.")
        break

video.release()
cv2.destroyAllWindows()

# Save face data
faces_data = np.asarray(faces_data)
faces_data = faces_data.reshape(len(faces_data), -1)

# Ensure data directory exists
os.makedirs('data', exist_ok=True)

if not os.path.exists('data/names.pkl'):
    names = [name] * len(faces_data)
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)
else:
    with open('data/names.pkl', 'rb') as f:
        names = pickle.load(f)
    names.extend([name] * len(faces_data))
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)

if not os.path.exists('data/faces_data.pkl'):
    with open('data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces_data, f)
else:
    with open('data/faces_data.pkl', 'rb') as f:
        faces = pickle.load(f)
    faces = np.vstack((faces, faces_data))
    with open('data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces, f)

print("Data saved successfully.")
