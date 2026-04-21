import numpy as np
from utils import calculate_distance
import config

class GestureRecognizer:
    def __init__(self):

        self.gesture_history = []
        self.history_length = 5  # Smoothing window
        
    def recognize(self, hand_landmarks, finger_count, handedness):
        landmarks = hand_landmarks.landmark
        
        # STOP gesture: Open palm (all fingers raised)
        if finger_count == 5:
            # Check if fingers are spread (open palm)
            if self._is_open_palm(landmarks):
                gesture = "STOP"
            else:
                gesture = "OPEN_HAND"
        
        # GO gesture: Pointing index finger
        elif finger_count == 1:
            # Check if only index finger is raised
            if self._is_pointing(landmarks):
                gesture = "GO"
            else:
                gesture = "ONE"
        
        # PEACE/VICTORY gesture
        elif finger_count == 2:
            if self._is_peace_sign(landmarks, handedness):
                gesture = "PEACE"
            else:
                gesture = "TWO"
        
        # OK gesture: Thumb and index forming circle
        elif finger_count == 3:
            if self._is_ok_gesture(landmarks):
                gesture = "OK"
            else:
                gesture = "THREE"
        
        # THUMBS UP gesture
        elif finger_count == 1 and self._is_thumbs_up(landmarks, handedness):
            gesture = "THUMBS_UP"
        
        # PINCH gesture (thumb and index touching)
        elif self._is_pinch(landmarks):
            gesture = "PINCH"
        
        else:
            gesture = f"FINGER_COUNT_{finger_count}"
        
        # Smoothing: Store in history and return most common
        self.gesture_history.append(gesture)
        if len(self.gesture_history) > self.history_length:
            self.gesture_history.pop(0)
        
        # Return most frequent gesture in history
        from collections import Counter
        most_common = Counter(self.gesture_history).most_common(1)[0][0]
        return most_common
    
    def _is_open_palm(self, landmarks):
        # Check if fingers are spread apart
        index_tip = landmarks[config.LANDMARKS['INDEX_TIP']]
        pinky_tip = landmarks[config.LANDMARKS['PINKY_TIP']]
        
        # If horizontal distance is large, fingers are spread
        distance = abs(index_tip.x - pinky_tip.x)
        return distance > 0.3  # Threshold for spread fingers
    
    def _is_pointing(self, landmarks):
        index_tip = landmarks[config.LANDMARKS['INDEX_TIP']]
        middle_tip = landmarks[config.LANDMARKS['MIDDLE_TIP']]
        
        # Index should be higher than middle finger
        return index_tip.y < middle_tip.y
    
    def _is_peace_sign(self, landmarks, handedness):
        # Index and middle fingers up
        index_up = landmarks[config.LANDMARKS['INDEX_TIP']].y < landmarks[config.LANDMARKS['INDEX_DIP']].y
        middle_up = landmarks[config.LANDMARKS['MIDDLE_TIP']].y < landmarks[config.LANDMARKS['MIDDLE_DIP']].y
        
        # Ring and pinky down
        ring_down = landmarks[config.LANDMARKS['RING_TIP']].y > landmarks[config.LANDMARKS['RING_DIP']].y
        pinky_down = landmarks[config.LANDMARKS['PINKY_TIP']].y > landmarks[config.LANDMARKS['PINKY_DIP']].y
        
        # Thumb position depends on handedness
        if handedness == "Right":
            thumb_side = landmarks[config.LANDMARKS['THUMB_TIP']].x > landmarks[config.LANDMARKS['THUMB_IP']].x
        else:
            thumb_side = landmarks[config.LANDMARKS['THUMB_TIP']].x < landmarks[config.LANDMARKS['THUMB_IP']].x
        
        return index_up and middle_up and ring_down and pinky_down and not thumb_side
    
    def _is_ok_gesture(self, landmarks):
        thumb_tip = landmarks[config.LANDMARKS['THUMB_TIP']]
        index_tip = landmarks[config.LANDMARKS['INDEX_TIP']]
        
        distance = calculate_distance(thumb_tip, index_tip)
        return distance < config.PINCH_THRESHOLD
    
    def _is_thumbs_up(self, landmarks, handedness):
        # Thumb raised
        if handedness == "Right":
            thumb_up = landmarks[config.LANDMARKS['THUMB_TIP']].x > landmarks[config.LANDMARKS['THUMB_IP']].x
        else:
            thumb_up = landmarks[config.LANDMARKS['THUMB_TIP']].x < landmarks[config.LANDMARKS['THUMB_IP']].x
        
        # All other fingers closed
        fingers_closed = all([
            landmarks[config.LANDMARKS['INDEX_TIP']].y > landmarks[config.LANDMARKS['INDEX_DIP']].y,
            landmarks[config.LANDMARKS['MIDDLE_TIP']].y > landmarks[config.LANDMARKS['MIDDLE_DIP']].y,
            landmarks[config.LANDMARKS['RING_TIP']].y > landmarks[config.LANDMARKS['RING_DIP']].y,
            landmarks[config.LANDMARKS['PINKY_TIP']].y > landmarks[config.LANDMARKS['PINKY_DIP']].y
        ])
        
        return thumb_up and fingers_closed
    
    def _is_pinch(self, landmarks):
        thumb_tip = landmarks[config.LANDMARKS['THUMB_TIP']]
        index_tip = landmarks[config.LANDMARKS['INDEX_TIP']]
        
        distance = calculate_distance(thumb_tip, index_tip)
        return distance < config.PINCH_THRESHOLD