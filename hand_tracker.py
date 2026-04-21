import cv2
import mediapipe as mp
import config

class HandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize hand detection model
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=config.MAX_NUM_HANDS,
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE
        )
        
    def find_hands(self, frame, draw=True):
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        self.results = self.hands.process(frame_rgb)
        
        # Draw landmarks if hands are detected
        if draw and self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                # Draw hand landmarks and connections
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        
        return frame, self.results
    
    def get_hand_position(self, frame):
        hand_info = []
        h, w = frame.shape[:2]
        
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                # Get all landmark coordinates
                landmarks = []
                x_min, y_min = w, h
                x_max, y_max = 0, 0
                
                for lm in hand_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    landmarks.append((x, y))
                    
                    # Update bounding box
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)
                
                # Add padding
                padding = 20
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(w, x_max + padding)
                y_max = min(h, y_max + padding)
                
                # Calculate center
                center_x = (x_min + x_max) // 2
                center_y = (y_min + y_max) // 2
                
                hand_info.append({
                    'landmarks': hand_landmarks.landmark,
                    'bbox': (x_min, y_min, x_max, y_max),
                    'center': (center_x, center_y)
                })
        
        return hand_info
    
    def get_handedness(self):
        handedness = []
        if self.results.multi_handedness:
            for hand in self.results.multi_handedness:
                handedness.append(hand.classification[0].label)
        return handedness