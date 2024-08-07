import cv2
import mediapipe as mp
import time

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize MediaPipe Drawing
mp_drawing = mp.solutions.drawing_utils

# Capture video from the standard (default) camera
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam

# Variables to keep track of curl-up count
curl_up_count = 0
is_up = False
is_down = True

# Threshold for detecting curl-up
curl_up_threshold = 0.1  # Allowable difference in y-coordinates to consider it a curl-up

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Pose
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        # Get the landmarks
        landmarks = results.pose_landmarks.landmark

        # Get the y-coordinates of the chest and knees
        left_shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
        right_shoulder_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
        left_knee_y = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y
        right_knee_y = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y

        # Compute the average y-coordinates
        avg_shoulder_y = (left_shoulder_y + right_shoulder_y) / 2
        avg_knee_y = (left_knee_y + right_knee_y) / 2

        # Check if the user is in an up position
        if avg_shoulder_y < avg_knee_y - curl_up_threshold:
            if is_down:
                is_up = True
                is_down = False
        elif avg_shoulder_y > avg_knee_y + curl_up_threshold:
            if is_up:
                is_up = False
                is_down = True
                curl_up_count += 1

        # Display the curl-up count on the frame
        cv2.putText(frame, f'Curl-Ups: {curl_up_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Draw the pose landmarks on the frame
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Display the frame
    cv2.imshow('Curl-Up Counter', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()