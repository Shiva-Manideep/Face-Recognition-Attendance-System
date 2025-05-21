# FACE-RECOGNITION-ATTENDANCE-SYSTEM
**1. Data Collection Phase (Training)**
The webcam is activated, and the user is prompted to input their name. Using a Haarcascade classifier, the system detects the face in each frame. The face region is cropped and resized to 50×50 pixels to ensure uniformity. Every 10th frame is used to collect a total of 100 images per user. Each image is flattened into a one-dimensional array and stored in `faces_data.pkl`, while the corresponding user names are stored in `names.pkl`, both using Python's `pickle` module.



**2. Recognition & Attendance Phase**
The system loads the previously saved face data and labels, and trains a K-Nearest Neighbors (KNN) classifier for recognition. The webcam is then used to detect faces in real-time. For each detected face, it is preprocessed (cropped, resized, and flattened) and passed to the trained KNN model to predict the name. A rectangle and name label are drawn on the video frame around each recognized face. When the user presses the "o" key, the system records the recognized name along with the current timestamp in a CSV file and announces "Attendance Taken" using text-to-speech. The live camera feed is displayed over a background image, and the system exits when the "q" key is pressed.
