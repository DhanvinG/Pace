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

# Threshold for detecting plank
plank_threshold = 0.1  # Allowable difference in y-coordinates to consider it a plank

# Variables to keep track of plank duration
plank_start_time = None
plank_duration = 0

# Threshold for detecting floor touch
floor_touch_threshold = 0.4  # Larger threshold to detect when the body touches the floor

# Threshold for horizontal alignment
horizontal_alignment_threshold = 0.4  # Allowable difference in x-coordinates to consider horizontal alignment

def is_touching_floor(landmarks):
    # Get the y-coordinates of the hips, shoulders, nose, and knees
    left_hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
    right_hip_y = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y
    left_shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
    right_shoulder_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
    nose_y = landmarks[mp_pose.PoseLandmark.NOSE.value].y
    left_knee_y = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y
    right_knee_y = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y

    # Compute the average y-coordinates
    avg_hip_y = (left_hip_y + right_hip_y) / 2
    avg_shoulder_y = (left_shoulder_y + right_shoulder_y) / 2

    # Check if the hips are significantly lower than the shoulders or other parts are touching the floor
    return (avg_hip_y - avg_shoulder_y > floor_touch_threshold or 
            nose_y > 1 - floor_touch_threshold or 
            left_knee_y > 1 - floor_touch_threshold or 
            right_knee_y > 1 - floor_touch_threshold)

def is_horizontal(landmarks):
    # Get the x-coordinates of the shoulders and hips
    left_shoulder_x = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x
    right_shoulder_x = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x
    left_hip_x = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x
    right_hip_x = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x

    # Compute the average x-coordinates
    avg_shoulder_x = (left_shoulder_x + right_shoulder_x) / 2
    avg_hip_x = (left_hip_x + right_hip_x) / 2

    # Check if the shoulders and hips are aligned horizontally
    return (abs(left_shoulder_x - right_shoulder_x) < horizontal_alignment_threshold and
            abs(left_hip_x - right_hip_x) < horizontal_alignment_threshold and
            abs(avg_shoulder_x - avg_hip_x) < horizontal_alignment_threshold)

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

        # Get the y-coordinates of the hips, shoulders, and ankles
        left_hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
        right_hip_y = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y
        left_shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
        right_shoulder_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
        left_ankle_y = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y
        right_ankle_y = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y

        # Compute the average y-coordinates
        avg_hip_y = (left_hip_y + right_hip_y) / 2
        avg_shoulder_y = (left_shoulder_y + right_shoulder_y) / 2
        avg_ankle_y = (left_ankle_y + right_ankle_y) / 2

        # Check if the user is in a plank position
        if (abs(avg_hip_y - avg_shoulder_y) < plank_threshold and
                abs(avg_hip_y - avg_ankle_y) < plank_threshold and
                not is_touching_floor(landmarks) and
                is_horizontal(landmarks)):
            if plank_start_time is None:
                plank_start_time = time.time()
            plank_duration = int(time.time() - plank_start_time)
        else:
            plank_start_time = None
            plank_duration = 0

        # Display the plank duration on the frame        
        cv2.putText(frame, f'Time: {plank_duration} sec', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Draw the pose landmarks on the frame
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Display the frame
    cv2.imshow('Plank Detection', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
