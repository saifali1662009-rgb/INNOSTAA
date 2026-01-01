import cv2
import mediapipe as mp
import numpy as np

class GestureWhiteboard:
    def __init__(self):
        self.running = False
        self.cap = None   # camera created later

        self.mp_hands = mp.solutions.hands
        self.hands = None
        self.mp_draw = mp.solutions.drawing_utils

        self.whiteboard = np.ones((480, 640, 3), dtype=np.uint8) * 255
        self.prev_x, self.prev_y = 0, 0
        self.brush_color = (0, 0, 0)
        self.brush_thickness = 8
        self.eraser_thickness = 40

    def start(self):
        # ✅ Create camera ONLY here
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        self.running = True

    def stop(self):
        self.running = False

        if self.hands:
            self.hands.close()
            self.hands = None

        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None

    def count_fingers(self, lm):
        tips = [8, 12, 16, 20]
        count = 0
        for tip in tips:
            if lm.landmark[tip].y < lm.landmark[tip - 2].y:
                count += 1
        return count

    def get_frame(self):
        if not self.running or self.cap is None:
            return None

        ret, frame = self.cap.read()
        if not ret:
            return None

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand in result.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame, hand, self.mp_hands.HAND_CONNECTIONS
                )

                x = int(hand.landmark[8].x * 640)
                y = int(hand.landmark[8].y * 480)
                fingers = self.count_fingers(hand)

                # ✍ DRAW → 2 fingers
                if fingers == 2:
                    if self.prev_x != 0:
                        cv2.line(
                            self.whiteboard,
                            (self.prev_x, self.prev_y),
                            (x, y),
                            self.brush_color,
                            self.brush_thickness
                        )
                    self.prev_x, self.prev_y = x, y

                # 🧽 ERASE → fist
                elif fingers == 0:
                    cv2.circle(
                        self.whiteboard,
                        (x, y),
                        self.eraser_thickness,
                        (255, 255, 255),
                        -1
                    )
                    self.prev_x, self.prev_y = 0, 0

                else:
                    self.prev_x, self.prev_y = 0, 0

        combined = cv2.addWeighted(frame, 0.4, self.whiteboard, 0.6, 0)
        return combined
