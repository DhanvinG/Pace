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

# Threshold to detect lunges
lunge_threshold = 0.1  # Adjusted threshold for detecting lunges
previous_left_knee_y = None
previous_right_knee_y = None
lunge_detected = False
lunge_count = 0

# Cooldown variables
cooldown_time = 1  # 1 second cooldown
last_lunge_time = 0

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

        # Get the y-coordinate of the left and right knees
        left_knee_y = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y
        right_knee_y = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y

        # Check if the user is performing a lunge
        if previous_left_knee_y is not None and previous_right_knee_y is not None:
            left_knee_movement = abs(left_knee_y - previous_left_knee_y)
            right_knee_movement = abs(right_knee_y - previous_right_knee_y)

            current_time = time.time()

            if current_time - last_lunge_time > cooldown_time:
                if left_knee_movement > lunge_threshold or right_knee_movement > lunge_threshold:
                    lunge_count += 1
                    print(f"Lunge Detected - Count: {lunge_count}")
                    last_lunge_time = current_time

        # Update the previous knee y-coordinates
        previous_left_knee_y = left_knee_y
        previous_right_knee_y = right_knee_y

        # Draw the pose landmarks on the frame
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Display the lunge count on the frame
        cv2.putText(frame, f'Lunges: {lunge_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Lunge Detection', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
