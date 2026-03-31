# Pace

Pace is a computer-vision fitness prototype that uses **OpenCV**, **MediaPipe Pose**, and a webcam feed to detect and track bodyweight exercises in real time.

The repository explores how pose estimation can be used to recognize workout movements such as **squats, push-ups, jumping jacks, planks, curl-ups, lunges, and running in place** through lightweight rule-based logic.

## Overview

This project was built as an early-stage prototype to test whether webcam-based exercise recognition could work without requiring wearables or specialized hardware.

Each script in this repository opens a webcam stream, extracts body landmarks using MediaPipe Pose, and applies exercise-specific heuristics to detect movement patterns or count repetitions.

## Features

- Real-time pose tracking using **MediaPipe Pose**
- Webcam-based exercise recognition with **OpenCV**
- Prototype support for multiple exercises:
  - Squats
  - Push-ups
  - Jumping jacks
  - Planks
  - Curl-ups
  - Lunges
  - Running in place
- Lightweight rule-based detection logic for repetition counting and state changes

## Repository Structure

- `squat_test.py` — detects squat down/up motion and counts repetitions
- `pushups_test.py` — tracks push-up motion using shoulder and elbow landmarks
- `jumpingjacks_test.py` — detects jumping jack cycles using hand and foot movement
- `plank_test.py` — estimates whether a plank position is being held and tracks duration
- `curlup_test.py` — counts curl-up repetitions using upper-body and leg landmark movement
- `lunge_test.py` — detects lunge-like motion using knee position changes
- `running_test.py` — detects running-in-place movement from foot motion

## Tech Stack

- Python
- OpenCV
- MediaPipe

## How It Works

1. Capture live video from the webcam.
2. Run pose estimation on each frame using MediaPipe Pose.
3. Extract relevant landmarks for the selected exercise.
4. Apply threshold-based logic to determine whether the user is in an “up,” “down,” or “hold” state.
5. Count repetitions or display exercise status in real time.

## Installation

```bash
pip install opencv-python mediapipe
```

## Running the Scripts

Run any exercise script directly:

```bash
python squat_test.py
```

You can replace `squat_test.py` with any other script in the repository, such as:

```bash
python pushups_test.py
python plank_test.py
python jumpingjacks_test.py
```

## Current Limitations

This repository is a **prototype** and the current exercise detection logic is based on simple handcrafted thresholds. That means performance may vary depending on:

- Camera angle
- Lighting conditions
- Distance from the camera
- Body proportions
- Pose estimation stability

Some exercises are tracked more reliably than others, and the project has not yet been calibrated across different users or environments.

## Future Improvements

- Refactor repeated webcam and pose-tracking logic into shared utility modules
- Create a single interface for selecting exercises
- Replace simple y-threshold heuristics with more robust angle-based biomechanics
- Add per-user calibration
- Improve repetition accuracy and reduce false positives

## Why I Built This

I built Pace to explore how computer vision could make fitness tracking more accessible and interactive using only a webcam. The project was also a way to experiment with pose estimation, real-time movement analysis, and lightweight human activity recognition.

## Notes

Pace is best understood as an **experimental proof of concept** rather than a finished production application. Its main value is in demonstrating early exploration of webcam-based workout recognition using pose landmarks and real-time computer vision.
