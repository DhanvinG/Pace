import cv2
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize MediaPipe Drawing
mp_drawing = mp.solutions.drawing_utils

# Capture video from the standard (default) camera
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam

# Thresholds to detect jumping jacks
jumping_jack_threshold = 0.1  # Adjusted threshold for detecting hand movements
previous_left_hand_y = None
previous_right_hand_y = None
previous_foot_distance = None
jumping_jacks_count = 0
hand_up = False

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

        # Get the y-coordinate of the left and right hands
        left_hand_y = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y
        right_hand_y = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y

        # Get the x-coordinates of the left and right feet
        left_foot_x = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x
        right_foot_x = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x

        # Compute the distance between the feet
        current_foot_distance = abs(left_foot_x - right_foot_x)

        # Check if the hands are moving up and feet are apart
        if previous_left_hand_y is not None and previous_right_hand_y is not None and previous_foot_distance is not None:
            hand_movement = (left_hand_y - previous_left_hand_y) + (right_hand_y - previous_right_hand_y)

            if hand_movement < -jumping_jack_threshold and current_foot_distance > previous_foot_distance:
                if not hand_up:
                    hand_up = True
                    print("Hands Up Detected")

            if hand_movement > jumping_jack_threshold and current_foot_distance < previous_foot_distance:
                if hand_up:
                    hand_up = False
                    jumping_jacks_count += 1
                    print(f"Jumping Jack Detected - Count: {jumping_jacks_count}")

        # Update the previous hand and foot coordinates
        previous_left_hand_y = left_hand_y
        previous_right_hand_y = right_hand_y
        previous_foot_distance = current_foot_distance

        # Draw the pose landmarks on the frame
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Display the jumping jacks count on the frame
        cv2.putText(frame, f'Jumping Jacks: {jumping_jacks_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Jumping Jacks Detection', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
