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

# Adjusted thresholds for detecting squats
squat_threshold_down = 0.05  # Adjusted threshold for detecting the downward movement
squat_threshold_up = 0.02    # Adjusted threshold for detecting the upward movement

# Flags and counters
squat_down = False
squat_count = 0
last_squat_time = 0
message_duration = 2  # Duration to prevent multiple detections in seconds

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

        # Get the y-coordinate of the hips and knees
        left_hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
        right_hip_y = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y
        left_knee_y = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y
        right_knee_y = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y

        # Compute the average y-coordinate of hips and knees
        avg_hip_y = (left_hip_y + right_hip_y) / 2
        avg_knee_y = (left_knee_y + right_knee_y) / 2

        current_time = time.time()

        # Check if the user is squatting down
        if avg_hip_y >= ((80 * (avg_knee_y + squat_threshold_down)) / 100):
            if not squat_down and (current_time - last_squat_time) > message_duration:
                squat_down = True
                squat_count += 1
                last_squat_time = current_time

        # Check if the user is standing up from the squat
        if avg_hip_y <= ((80 * (avg_knee_y + squat_threshold_down)) / 100):
            if squat_down:
                squat_down = False

        # Draw the pose landmarks on the frame
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Display the squat count on the frame
    cv2.putText(frame, f'Squat Count: {squat_count}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Squat Detection', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
