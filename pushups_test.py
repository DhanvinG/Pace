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

# Variables to keep track of push-up count
push_up_count = 0
is_down = False

# Threshold for detecting push-up
push_up_threshold = 0.1  # Allowable difference in y-coordinates to consider it a push-up

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

        # Get the y-coordinates of the shoulders and elbows
        left_shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
        right_shoulder_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
        left_elbow_y = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y
        right_elbow_y = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y

        # Compute the average y-coordinates
        avg_shoulder_y = (left_shoulder_y + right_shoulder_y) / 2
        avg_elbow_y = (left_elbow_y + right_elbow_y) / 2

        # Check if the user is in a push-up position
        if avg_elbow_y - avg_shoulder_y > push_up_threshold:
            if not is_down:
                is_down = True
        else:
            if is_down:
                is_down = False
                push_up_count += 1

        # Display the push-up count on the frame
        cv2.putText(frame, f'Push-Ups: {push_up_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Draw the pose landmarks on the frame
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Display the frame
    cv2.imshow('Push-Up Counter', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
