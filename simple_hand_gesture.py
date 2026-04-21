import cv2
import mediapipe as mp
import numpy as np

class SimpleHandGesture:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Finger tip and joint indices
        self.finger_tips = [4, 8, 12, 16, 20]
        self.finger_dips = [2, 6, 10, 14, 18]
        self.finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
        
        # Start camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
    def count_fingers(self, landmarks, handedness):
        """Count raised fingers"""
        fingers = []
        
        # Thumb (horizontal movement)
        if handedness == "Right":
            if landmarks[self.finger_tips[0]].x > landmarks[self.finger_dips[0]].x:
                fingers.append(1)
            else:
                fingers.append(0)
        else:  # Left hand
            if landmarks[self.finger_tips[0]].x < landmarks[self.finger_dips[0]].x:
                fingers.append(1)
            else:
                fingers.append(0)
        
        # Other fingers (vertical movement)
        for i in range(1, 5):
            if landmarks[self.finger_tips[i]].y < landmarks[self.finger_dips[i]].y:
                fingers.append(1)
            else:
                fingers.append(0)
        
        return sum(fingers), fingers
    
    def recognize_gesture(self, finger_count, finger_states, handedness):
        """Recognize common gestures"""
        if finger_count == 5:
            return "STOP"
        elif finger_count == 1 and finger_states[1] == 1:  # Only index up
            return "GO"
        elif finger_count == 2 and finger_states[1] == 1 and finger_states[2] == 1:
            return "PEACE"
        elif finger_count == 0:
            return "FIST"
        elif finger_count == 3:
            return "THREE"
        else:
            return f"COUNT: {finger_count}"
    
    def draw_info(self, frame, handedness, finger_count, gesture, finger_states):
        """Draw information on frame"""
        h, w = frame.shape[:2]
        
        # Background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 180), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        # Display information
        y_pos = 40
        cv2.putText(frame, f"Hand: {handedness}", (20, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(frame, f"Fingers: {finger_count}", (20, y_pos + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(frame, f"Gesture: {gesture}", (20, y_pos + 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Show individual finger states
        finger_text = " | ".join([f"{self.finger_names[i]}:{finger_states[i]}" 
                                 for i in range(5)])
        cv2.putText(frame, finger_text, (20, y_pos + 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame
    
    def run(self):
        """Main loop"""
        print("Hand Gesture Recognition Started!")
        print("Press 'q' to quit")
        print("Press 's' to save screenshot")
        
        while True:
            success, frame = self.cap.read()
            if not success:
                break
            
            # Flip horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)
            
            if results.multi_hand_landmarks:
                for hand_landmarks, handedness_obj in zip(
                    results.multi_hand_landmarks,
                    results.multi_handedness
                ):
                    # Get handedness
                    handedness = handedness_obj.classification[0].label
                    
                    # Count fingers
                    landmarks = hand_landmarks.landmark
                    finger_count, finger_states = self.count_fingers(landmarks, handedness)
                    
                    # Recognize gesture
                    gesture = self.recognize_gesture(finger_count, finger_states, handedness)
                    
                    # Draw landmarks
                    self.mp_draw.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # Draw information
                    frame = self.draw_info(frame, handedness, finger_count, 
                                          gesture, finger_states)
                    
                    # Draw bounding box
                    x_coords = [int(lm.x * frame.shape[1]) for lm in landmarks]
                    y_coords = [int(lm.y * frame.shape[0]) for lm in landmarks]
                    cv2.rectangle(frame, 
                                (min(x_coords)-20, min(y_coords)-20),
                                (max(x_coords)+20, max(y_coords)+20),
                                (255, 0, 0), 2)
            
            # Show instructions
            cv2.putText(frame, "Q: Quit | S: Screenshot", (10, frame.shape[0]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display
            cv2.imshow("Hand Gesture Recognition", frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = cv2.getTickCount()
                cv2.imwrite(f"screenshot_{timestamp}.jpg", frame)
                print("Screenshot saved!")
        
        self.cap.release()
        cv2.destroyAllWindows()
        print("Application closed!")

if __name__ == "__main__":
    app = SimpleHandGesture()
    app.run()