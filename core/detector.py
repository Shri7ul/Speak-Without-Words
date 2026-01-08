import cv2
import mediapipe as mp

class HandDetector:
    def __init__(self, max_hands=1, det_conf=0.6, track_conf=0.6):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=det_conf,
            min_tracking_confidence=track_conf
        )
        self.drawer = mp.solutions.drawing_utils

    def process(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.hands.process(frame_rgb)

        hands_out = []  # each: {"lms": [...], "label": "Left"/"Right"}
        if res.multi_hand_landmarks:
            for i, hand_lms in enumerate(res.multi_hand_landmarks):
                one_hand = [(lm.x, lm.y, lm.z) for lm in hand_lms.landmark]

                label = "Unknown"
                if res.multi_handedness and len(res.multi_handedness) > i:
                    label = res.multi_handedness[i].classification[0].label  # "Left"/"Right"

                hands_out.append({"lms": one_hand, "label": label})

                self.drawer.draw_landmarks(
                    frame_bgr, hand_lms, self.mp_hands.HAND_CONNECTIONS
                )

        return frame_bgr, hands_out
