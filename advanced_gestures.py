import numpy as np
from collections import Counter

class AdvancedGestureRecognizer:
    def __init__(self):
        # Finger landmark indices
        self.tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        self.dips = [3, 6, 10, 14, 18]  # Joints
        self.mcps = [2, 5, 9, 13, 17]   # Base joints
        self.finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
        
        # Gesture history for smoothing
        self.gesture_history = []
        self.history_size = 5
        
    def get_finger_states(self, landmarks, handedness):
        """Get detailed state of each finger (raised/lowered)"""
        fingers = []
        
        # Thumb (horizontal movement)
        if handedness == "Right":
            fingers.append(1 if landmarks[self.tips[0]].x > landmarks[self.dips[0]].x else 0)
        else:
            fingers.append(1 if landmarks[self.tips[0]].x < landmarks[self.dips[0]].x else 0)
        
        # Other fingers (vertical movement)
        for i in range(1, 5):
            if landmarks[self.tips[i]].y < landmarks[self.dips[i]].y:
                fingers.append(1)
            else:
                fingers.append(0)
        
        return fingers
    
    def get_distance_between(self, landmarks, point1, point2):
        """Calculate distance between two landmarks"""
        p1 = np.array([landmarks[point1].x, landmarks[point1].y])
        p2 = np.array([landmarks[point2].x, landmarks[point2].y])
        return np.linalg.norm(p1 - p2)
    
    def get_angle(self, landmarks, point1, point2, point3):
        """Calculate angle between three points"""
        p1 = np.array([landmarks[point1].x, landmarks[point1].y])
        p2 = np.array([landmarks[point2].x, landmarks[point2].y])
        p3 = np.array([landmarks[point3].x, landmarks[point3].y])
        
        v1 = p1 - p2
        v2 = p3 - p2
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        return np.degrees(angle)
    
    def recognize_number_gestures(self, fingers):
        """Recognize number gestures (1-10)"""
        finger_count = sum(fingers)
        
        # Specific number gestures
        if fingers == [0, 1, 0, 0, 0]:  # Only index up
            return "ONE (1)"
        elif fingers == [0, 1, 1, 0, 0]:  # Index and middle
            return "TWO (2)"
        elif fingers == [0, 1, 1, 1, 0]:  # Index, middle, ring
            return "THREE (3)"
        elif fingers == [0, 1, 1, 1, 1]:  # All except thumb
            return "FOUR (4)"
        elif fingers == [1, 1, 1, 1, 1]:  # All fingers
            return "FIVE (5)"
        elif fingers == [1, 0, 0, 0, 0]:  # Only thumb
            return "SIX (6)"
        elif fingers == [1, 1, 0, 0, 0]:  # Thumb and index
            return "SEVEN (7)"
        elif fingers == [1, 1, 1, 0, 0]:  # Thumb, index, middle
            return "EIGHT (8)"
        elif fingers == [0, 0, 0, 0, 0]:  # Fist
            return "ZERO (0)"
        
        return None
    
    def recognize_letter_gestures(self, landmarks, fingers):
        """Recognize ASL letter gestures"""
        
        # A: Fist with thumb to side
        if sum(fingers) == 1 and fingers[0] == 1 and all(f == 0 for f in fingers[1:]):
            return "LETTER A"
        
        # B: All fingers straight, thumb folded
        if fingers[0] == 0 and all(f == 1 for f in fingers[1:]):
            return "LETTER B"
        
        # C: Thumb and index curved like C
        thumb_index_dist = self.get_distance_between(landmarks, self.tips[0], self.tips[1])
        if 0.05 < thumb_index_dist < 0.15 and fingers[2] == 0:
            return "LETTER C 🇨"
        
        # L: Thumb and index at 90 degrees
        if fingers[0] == 1 and fingers[1] == 1 and sum(fingers[2:]) == 0:
            angle = self.get_angle(landmarks, self.mcps[0], self.tips[0], self.tips[1])
            if 60 < angle < 120:
                return "LETTER L 🇱"
        
        # OK sign: Thumb and index touching
        ok_distance = self.get_distance_between(landmarks, self.tips[0], self.tips[1])
        if ok_distance < 0.05 and fingers[2] == 1:
            return "OK"
        
        # Peace/Victory
        if fingers[1] == 1 and fingers[2] == 1 and sum(fingers[3:]) == 0 and fingers[0] == 0:
            index_mid_dist = self.get_distance_between(landmarks, self.tips[1], self.tips[2])
            if index_mid_dist > 0.1:
                return "PEACE"
        
        return None
    
    def recognize_shape_gestures(self, landmarks, fingers):
        """Recognize hand shapes and symbols"""
        
        # Heart shape
        thumb_index_dist = self.get_distance_between(landmarks, self.tips[0], self.tips[1])
        index_mid_dist = self.get_distance_between(landmarks, self.tips[1], self.tips[2])
        
        if thumb_index_dist < 0.05 and index_mid_dist < 0.05:
            return "HEART"
        
        # Star/Spider shape (all fingers spread wide)
        if all(f == 1 for f in fingers):
            spread = all(
                self.get_distance_between(landmarks, self.tips[i], self.tips[i+1]) > 0.1
                for i in range(1, 4)
            )
            if spread:
                return "STAR"
        
        # Gun gesture
        if fingers[0] == 1 and fingers[1] == 1 and sum(fingers[2:]) == 0:
            angle = self.get_angle(landmarks, self.mcps[1], self.tips[1], self.tips[0])
            if angle > 150:
                return "GUN"
        
        # Call me gesture
        if fingers[0] == 1 and fingers[4] == 1 and sum(fingers[1:4]) == 0:
            return "CALL ME"
        
        # Rock On
        if fingers[1] == 1 and fingers[4] == 1 and sum(fingers[2:4]) == 0:
            return "ROCK ON"
        
        # Spider-Man web shooter
        if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1 and fingers[0] == 0 and fingers[4] == 0:
            return "WEB SHOOTER"
        
        return None
    
    def recognize_control_gestures(self, landmarks, fingers):
        """Recognize control-oriented gestures"""
        
        # Pinch gesture
        pinch_dist = self.get_distance_between(landmarks, self.tips[0], self.tips[1])
        if pinch_dist < 0.05:
            return "PINCH"
        
        # Pointing
        if fingers[1] == 1 and sum([fingers[0], fingers[2], fingers[3], fingers[4]]) <= 1:
            return "POINTING"
        
        # Double point
        if fingers[1] == 1 and fingers[2] == 1:
            index_mid_dist = self.get_distance_between(landmarks, self.tips[1], self.tips[2])
            if index_mid_dist < 0.05:
                return "DOUBLE POINT"
        
        # Grab/Claw
        if all(f == 1 for f in fingers):
            avg_tip_dist = np.mean([
                self.get_distance_between(landmarks, self.tips[i], self.mcps[i])
                for i in range(5)
            ])
            if avg_tip_dist < 0.15:
                return "GRAB"
        
        return None
    
    def recognize_emotion_gestures(self, landmarks, fingers, handedness):
        """Recognize emotion gestures"""
        
        # Thumbs up/down
        if fingers[0] == 1 and sum(fingers[1:]) == 0:
            if handedness == "Right":
                if landmarks[self.tips[0]].x > landmarks[self.dips[0]].x:
                    return "THUMBS UP"
                else:
                    return "THUMBS DOWN"
            else:
                if landmarks[self.tips[0]].x < landmarks[self.dips[0]].x:
                    return "THUMBS UP"
                else:
                    return "THUMBS DOWN"
        
        # High five
        if all(f == 1 for f in fingers):
            spread = all(
                self.get_distance_between(landmarks, self.tips[i], self.tips[i+1]) > 0.08
                for i in range(1, 4)
            )
            if spread:
                return "HIGH FIVE"
        
        # Fist bump
        if sum(fingers) == 0:
            return "FIST BUMP"
        
        return None
    
    def recognize_gesture(self, landmarks, handedness):
        """Main method to recognize all gestures"""
        
        # Get finger states
        fingers = self.get_finger_states(landmarks, handedness)
        
        # Try each gesture category (order matters - more specific first)
        gesture = self.recognize_shape_gestures(landmarks, fingers)
        if gesture: 
            return self.smooth_gesture(gesture), sum(fingers)
        
        gesture = self.recognize_letter_gestures(landmarks, fingers)
        if gesture: 
            return self.smooth_gesture(gesture), sum(fingers)
        
        gesture = self.recognize_control_gestures(landmarks, fingers)
        if gesture: 
            return self.smooth_gesture(gesture), sum(fingers)
        
        gesture = self.recognize_emotion_gestures(landmarks, fingers, handedness)
        if gesture: 
            return self.smooth_gesture(gesture), sum(fingers)
        
        gesture = self.recognize_number_gestures(fingers)
        if gesture: 
            return self.smooth_gesture(gesture), sum(fingers)
        
        # Default: Show finger count
        finger_count = sum(fingers)
        if finger_count == 0:
            return self.smooth_gesture("FIST"), 0
        else:
            return self.smooth_gesture(f"{finger_count} FINGERS"), finger_count
    
    def smooth_gesture(self, gesture):
        """Smooth gesture recognition to avoid flickering"""
        self.gesture_history.append(gesture)
        if len(self.gesture_history) > self.history_size:
            self.gesture_history.pop(0)
        
        # Return most common gesture in history
        most_common = Counter(self.gesture_history).most_common(1)[0][0]
        return most_common