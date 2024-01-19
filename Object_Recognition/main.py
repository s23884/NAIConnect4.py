"""
Program wykorzystuje moduły OpenCV oraz MediaPipe
do śledzenia i detekcji ruchu człowieka na podstawie obrazu z kamerki.

Authorzy: Wiktor Krieger i Sebastian Augustyniak

Przykład użycia:
1. Uruchom program, który uruchamia kamerę (upewnij się że kamerka jest dostępna).
2. Program automatycznie wykrywa człowieka i reaguje na jego ruch.
3. Jeśli zostanie wykryty ruch, pojawi się celownik na twarzy, oraz adekwatny komunikat
4. Jeśli człowiek się podda (podniesie obie dłonie powyżej głowy),
   na ekranie pojawi się kwadrat wokół głowy, oraz adekwatny komunikat
5. Aby zakończyć działanie programu kliknij "q"
"""

import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic


def detect_movement(previous_frame, current_frame):
    frame_diff = cv2.absdiff(previous_frame, current_frame)
    threshold = 30
    _, threshold_diff = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)
    diff_sum = np.sum(threshold_diff)

    return diff_sum > 5000


def calculate_head_target(frame, pose_landmarks):
    nose_point = np.array([pose_landmarks[mp_holistic.PoseLandmark.NOSE].x,
                           pose_landmarks[mp_holistic.PoseLandmark.NOSE].y])
    left_shoulder_point = np.array([pose_landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER].x,
                                    pose_landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER].y])
    right_shoulder_point = np.array([pose_landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER].x,
                                     pose_landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER].y])

    neck_point = (left_shoulder_point + right_shoulder_point) / 2
    head_height = int(frame.shape[0] * np.linalg.norm(neck_point - nose_point))
    center_x, center_y = int(frame.shape[1] * nose_point[0]), int(frame.shape[0] * nose_point[1])

    return center_x, center_y, head_height


def draw_target(frame, pose_landmarks):
    center_x, center_y, head_height = calculate_head_target(frame, pose_landmarks)

    if head_height > 0:
        radius = int(head_height / 2)
        cv2.circle(frame, (center_x, center_y), radius, (0, 0, 0), 3)
        cv2.line(frame, (center_x - radius, center_y), (center_x + radius, center_y), (0, 0, 255), 3)
        cv2.line(frame, (center_x, center_y - radius), (center_x, center_y + radius), (0, 0, 255), 3)


def draw_surrender(frame, pose_landmarks):
    center_x, center_y, head_height = calculate_head_target(frame, pose_landmarks)

    if head_height > 0:
        rect_size = int(head_height)
        cv2.rectangle(frame, (center_x - rect_size // 2, center_y - rect_size // 2),
                      (center_x + rect_size // 2, center_y + rect_size // 2), (0, 255, 0), 3)


def detect_surrender(results):
    if results.right_hand_landmarks and results.left_hand_landmarks:
        right_hand_y = results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.WRIST].y
        left_hand_y = results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.WRIST].y
        head_y = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].y

        return right_hand_y <= head_y and left_hand_y <= head_y
    return False


cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    previous_frame = None
    surrender_detected = False
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = holistic.process(frame)

        if results.pose_landmarks:
            current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if previous_frame is not None:
                if detect_movement(previous_frame, current_frame):
                    surrender_detected = detect_surrender(results)
                    if surrender_detected:
                        cv2.putText(frame, "Cel sie poddaje! Nie strzelac!", (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                        draw_surrender(frame, results.pose_landmarks.landmark)
                    else:
                        cv2.putText(frame, "Cel sie poruszyl, oznaczono", (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                        draw_target(frame, results.pose_landmarks.landmark)

            previous_frame = current_frame

        # Rysuj połączenia na ciele (zakomentowane, w celu dodania przejrzystości)
        # mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        # mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        # mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        cv2.imshow('Baba Jaga Patrzy', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
