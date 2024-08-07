import cv2
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize MediaPipe Drawing
mp_drawing = mp.solutions.drawing_utils

# Capture video from the standard (default) camera
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam

# Thresholds to detect running in place
run_threshold = 0.017  # Adjusted threshold for detecting vertical foot movement
previous_left_foot_y = None
previous_right_foot_y = None

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

        # Get the y-coordinate of the left and right feet
        left_foot_y = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y
        right_foot_y = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y

        # Check if the user is running in place
        if previous_left_foot_y is not None and previous_right_foot_y is not None:
            left_foot_movement = abs(left_foot_y - previous_left_foot_y)
            right_foot_movement = abs(right_foot_y - previous_right_foot_y)
            
            if left_foot_movement > run_threshold or right_foot_movement > run_threshold:
                cv2.putText(frame, 'Running in Place Detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                print("Running in Place Detected")

        # Update the previous foot y-coordinates
        previous_left_foot_y = left_foot_y
        previous_right_foot_y = right_foot_y

        # Draw the pose landmarks on the frame
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Display the frame
    cv2.imshow('Running Detection', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()