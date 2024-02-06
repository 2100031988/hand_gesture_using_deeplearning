import cv2
import numpy as np
from sklearn.metrics import pairwise

accumulated_weight = 0.5
roi_top = 20
roi_bottom = 300
roi_right = 300
roi_left = 600
shift = (roi_right, roi_top)


def calc_accum_avg(frame, background):
    if background is None:
        return frame.copy()

    cv2.accumulateWeighted(frame, background.astype("float"), accumulated_weight)
    return background


def segment(frame, background, threshold=25):
    diff = cv2.absdiff(background, frame)
    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None
    else:
        hand_segment = max(contours, key=cv2.contourArea)
        return (thresh, hand_segment)


def count_fingers(thresh, hand_segment):
    conv_hull = cv2.convexHull(hand_segment, returnPoints=False)
    defects = cv2.convexityDefects(hand_segment, conv_hull)

    if defects is not None:
        count = 0
        fingerTips = []
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(hand_segment[s][0])
            end = tuple(hand_segment[e][0])
            far = tuple(hand_segment[f][0])

            a = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)

            angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))

            if angle <= np.pi / 2:
                count += 1
                fingerTips.append(far)

        return count, fingerTips, tuple(hand_segment[0][0])
    else:
        return 0, [], (0, 0)


def real_time_feed():
    background = None
    cam = cv2.VideoCapture(0)
    num_frames = 0
    while True:
        num_frames += 1
        ret, frame = cam.read()
        frame = cv2.flip(frame, 1)
        frame_copy = frame.copy()
        fingerTips = []
        roi = frame[roi_top:roi_bottom, roi_right:roi_left]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if num_frames <= 60:
            background = calc_accum_avg(gray, background)
            if num_frames <= 60:
                cv2.putText(frame_copy, "WAIT! GETTING BACKGROUND AVG.", (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 2)
                cv2.imshow("Finger Count", frame_copy)

        else:
            hand = segment(gray, background)

            if hand is not None:
                thresholded, hand_segment = hand
                res = cv2.drawContours(frame_copy, [hand_segment + (roi_right, roi_top)], -1, (255, 0, 0), 1)
                fingers, fingerTips, centrePt = count_fingers(thresholded, hand_segment)
                centrePt = tuple(map(sum, zip(centrePt, shift)))
                cv2.putText(frame_copy, "Count = " + str(fingers), (100, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),
                            2)
                cv2.imshow("Thresholded", thresholded)

        res = cv2.rectangle(frame_copy, (roi_left, roi_top), (roi_right, roi_bottom), (0, 0, 255), 5)
        for tip in fingerTips:
            tip = tuple(map(sum, zip(tip, shift)))
            res = cv2.line(frame_copy, tip, centrePt, (0, 0, 255), 2)
        cv2.imshow("Finger Count", frame_copy)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    real_time_feed()
