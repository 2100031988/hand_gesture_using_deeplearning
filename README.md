# Hand Gesture Identification

This project is a Python application for real-time hand gesture identification using the MediaPipe Hands library for hand landmark detection and OpenCV for video capture and processing. It can detect hand gestures in a live video stream from a webcam and count the number of fingers extended in each detected hand.

## Introduction

Hand gesture identification has various applications, including sign language recognition, gesture-based control systems, and virtual reality interactions. This project provides a simple yet effective solution for real-time hand gesture identification using computer vision techniques.

## Features

- Detects hand landmarks using MediaPipe Hands library
- Counts the number of fingers extended in each hand gesture
- Displays the finger count on the video feed in real-time
- Supports various hand gestures, including open hand, fist, and different finger configurations

## Installation

To run the project locally, follow these steps:

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/2100031988/hand_gesture_identification.git
2. download the needed files

   pip install -r requirements.txt

3. run the final code
   
   python main.py

Usage
Once the script is running, it will open a window displaying the live video feed from your webcam. It will detect hand gestures and count the number of fingers extended in each hand.

Extend your hand in front of the camera to see the detected hand landmarks and finger count.
Try different hand gestures, such as open hand, fist, and various finger configurations, to observe the detection accuracy.

