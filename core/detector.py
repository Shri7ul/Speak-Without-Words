# gesture detection logic
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
        """Returns: (annotated_frame, landmarks_list)
        landmarks_list: list of hands; each hand = list of (x,y,z) normalized [0..1]
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.hands.process(frame_rgb)

        landmarks_all = []
        if res.multi_hand_landmarks:
            for hand_lms in res.multi_hand_landmarks:
                # collect landmarks
                one_hand = []
                for lm in hand_lms.landmark:
                    one_hand.append((lm.x, lm.y, lm.z))
                landmarks_all.append(one_hand)

                # draw on frame
                self.drawer.draw_landmarks(
                    frame_bgr, hand_lms, self.mp_hands.HAND_CONNECTIONS
                )
        return frame_bgr, landmarks_all
