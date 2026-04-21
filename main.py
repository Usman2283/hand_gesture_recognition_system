import cv2
import mediapipe as mp
from advanced_gestures import AdvancedGestureRecognizer

class EnhancedHandGesture:
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
        
        # Initialize advanced gesture recognizer
        self.gesture_recognizer = AdvancedGestureRecognizer()
        
        # Start camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
    def draw_fancy_info_panel(self, frame, handedness, finger_count, gesture):
        """Draw beautiful info panel with gesture details"""
        h, w = frame.shape[:2]
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 180), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        # Draw information
        y = 40
        cv2.putText(frame, f"Hand: {handedness}", (20, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(frame, f"Fingers: {finger_count}", (20, y + 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Gesture with colored background
        cv2.rectangle(frame, (15, y + 70), (395, y + 115), (100, 0, 100), -1)
        
        # Split gesture text if too long
        if len(gesture) > 25:
            gesture = gesture[:25]
        
        cv2.putText(frame, f"Gesture:", (20, y + 95), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        cv2.putText(frame, f"{gesture}", (110, y + 95), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return frame
    
    def draw_gesture_legend(self, frame):
        """Draw gesture legend on the right side"""
        h, w = frame.shape[:2]
        
        # Legend background
        overlay = frame.copy()
        cv2.rectangle(overlay, (w - 220, 10), (w - 10, h - 10), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
        
        # Legend title
        cv2.putText(frame, "GESTURES", (w - 200, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Common gestures list
        gestures = [
            "Thumbs Up/Down",
            "Peace/Victory",
            "OK Sign",
            "Rock On",
            "High Five",
            "Fist Bump",
            "Heart Shape",
            "Pinch",
            "Pointing",
            "Spider-Man"
        ]
        
        y = 70
        for i, gesture in enumerate(gestures):
            if y > h - 50:
                break
            cv2.putText(frame, gesture, (w - 200, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            y += 25
        
        return frame
    
    def run(self):
        """Main loop"""
        print("=" * 60)
        print("ENHANCED HAND GESTURE RECOGNITION SYSTEM")
        print("=" * 60)
        print("20+ Gestures Available!")
        print("Real-time Recognition")
        print("Left/Right Hand Detection")
        print("=" * 60)
        print("Try these gestures:")
        print("   • Numbers 0-5")
        print("   • Letters A, B, C, L, O")
        print("   • Shapes: Heart, Star, Gun")
        print("   • Actions: Thumbs Up/Down, High Five")
        print("   • Controls: Pinch, Point, Grab")
        print("=" * 60)
        print("Controls:")
        print("   'q' - Quit")
        print("   's' - Screenshot")
        print("   'h' - Hide/Show Legend")
        print("=" * 60)
        
        show_legend = True
        
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
                    
                    # Recognize advanced gesture
                    gesture, finger_count = self.gesture_recognizer.recognize_gesture(
                        hand_landmarks.landmark, handedness
                    )
                    
                    # Draw hand landmarks with custom style
                    self.mp_draw.draw_landmarks(
                        frame, 
                        hand_landmarks, 
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)
                    )
                    
                    # Draw bounding box
                    h, w = frame.shape[:2]
                    x_coords = [int(lm.x * w) for lm in hand_landmarks.landmark]
                    y_coords = [int(lm.y * h) for lm in hand_landmarks.landmark]
                    
                    cv2.rectangle(frame, 
                                (min(x_coords)-20, min(y_coords)-20),
                                (max(x_coords)+20, max(y_coords)+20),
                                (255, 0, 0), 2)
                    
                    # Draw info panel
                    frame = self.draw_fancy_info_panel(frame, handedness, finger_count, gesture)
            
            # Draw legend (toggle with 'h')
            if show_legend:
                frame = self.draw_gesture_legend(frame)
            
            # Draw instructions
            cv2.putText(frame, "Q:Quit S:Screenshot H:Hide Legend", (10, frame.shape[0]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display FPS (simple calculation)
            cv2.putText(frame, "Gesture Recognition Active", (frame.shape[1]-250, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Show the frame
            cv2.imshow("Enhanced Hand Gesture Recognition", frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = cv2.getTickCount()
                filename = f"gesture_screenshot_{int(timestamp)}.jpg"
                cv2.imwrite(filename, frame)
                print(f"📸 Screenshot saved: {filename}")
            elif key == ord('h'):
                show_legend = not show_legend
                print(f"Legend {'Hidden' if not show_legend else 'Shown'}")
        
        self.cap.release()
        cv2.destroyAllWindows()
        print("\nApplication closed successfully!")
        print("Thanks for using Enhanced Hand Gesture Recognition!")

if __name__ == "__main__":
    app = EnhancedHandGesture()
    app.run()