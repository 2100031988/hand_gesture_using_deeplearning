import cv2
import mediapipe as mp

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize MediaPipe Drawing module for drawing landmarks
mp_drawing = mp.solutions.drawing_utils


def count_fingers(hand_landmarks, frame_width, frame_height):
    tip_ids = [4, 8, 12, 16, 20]  # Finger landmark indices
    finger_count = 0

    # Thumb detection
    if hand_landmarks.landmark[tip_ids[0]].x * frame_width < hand_landmarks.landmark[tip_ids[1]].x * frame_width:
        finger_count += 1

    # Other fingers detection
    for id in range(1, 5):
        if hand_landmarks.landmark[tip_ids[id]].y * frame_height < hand_landmarks.landmark[
            tip_ids[id] - 2].y * frame_height:
            finger_count += 1

    # Check for fist (all fingers folded)
    if finger_count == 0:
        thumb_tip_y = hand_landmarks.landmark[4].y * frame_height
        pinky_tip_y = hand_landmarks.landmark[20].y * frame_height
        if thumb_tip_y > pinky_tip_y:
            finger_count = 5  # All fingers folded, consider it as a fist

    return finger_count


# Open a video capture object (0 for the default camera)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        continue

    # Convert the frame to RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hands
    results = hands.process(frame_rgb)

    # Check if hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get frame dimensions
            frame_height, frame_width, _ = frame.shape

            # Count fingers
            finger_count = count_fingers(hand_landmarks, frame_width, frame_height)

            # Display finger count
            cv2.putText(frame, "Finger Count: " + str(finger_count), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                        2, cv2.LINE_AA)

    # Display the frame with hand landmarks
    cv2.imshow('Hand Recognition', frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()
