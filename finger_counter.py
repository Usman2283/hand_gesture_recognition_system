import math
import config

class FingerCounter:
    def __init__(self):
        # Finger tip and dip (joint) indices
        self.finger_tips = [
            config.LANDMARKS['THUMB_TIP'],
            config.LANDMARKS['INDEX_TIP'],
            config.LANDMARKS['MIDDLE_TIP'],
            config.LANDMARKS['RING_TIP'],
            config.LANDMARKS['PINKY_TIP']
        ]
        
        self.finger_dips = [
            config.LANDMARKS['THUMB_IP'],
            config.LANDMARKS['INDEX_DIP'],
            config.LANDMARKS['MIDDLE_DIP'],
            config.LANDMARKS['RING_DIP'],
            config.LANDMARKS['PINKY_DIP']
        ]
        
        self.finger_mcps = [
            config.LANDMARKS['THUMB_MCP'],
            config.LANDMARKS['INDEX_MCP'],
            config.LANDMARKS['MIDDLE_MCP'],
            config.LANDMARKS['RING_MCP'],
            config.LANDMARKS['PINKY_MCP']
        ]
    
    def count_fingers(self, hand_landmarks, handedness):
        fingers = []
        landmarks = hand_landmarks.landmark
        
        # Thumb detection (special case - horizontal movement)
        if handedness == "Right":
            # Right hand: thumb raised if tip is to the right of joint
            if landmarks[self.finger_tips[0]].x > landmarks[self.finger_dips[0]].x:
                fingers.append(1)
            else:
                fingers.append(0)
        else:  # Left hand
            # Left hand: thumb raised if tip is to the left of joint
            if landmarks[self.finger_tips[0]].x < landmarks[self.finger_dips[0]].x:
                fingers.append(1)
            else:
                fingers.append(0)
        
        # Other 4 fingers (vertical movement)
        for i in range(1, 5):
            # Finger raised if tip is above the dip joint
            if landmarks[self.finger_tips[i]].y < landmarks[self.finger_dips[i]].y:
                fingers.append(1)
            else:
                fingers.append(0)
        
        return sum(fingers)
    
    def get_finger_states(self, hand_landmarks, handedness):
        finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
        finger_states = {}
        
        landmarks = hand_landmarks.landmark
        
        # Thumb
        if handedness == "Right":
            finger_states['Thumb'] = landmarks[self.finger_tips[0]].x > landmarks[self.finger_dips[0]].x
        else:
            finger_states['Thumb'] = landmarks[self.finger_tips[0]].x < landmarks[self.finger_dips[0]].x
        
        # Other fingers
        for i in range(1, 5):
            finger_states[finger_names[i]] = landmarks[self.finger_tips[i]].y < landmarks[self.finger_dips[i]].y
        
        return finger_states
    
    def calculate_finger_angles(self, hand_landmarks):
        landmarks = hand_landmarks.landmark
        finger_angles = {}
        
        # Calculate angle for each finger
        finger_joints = [
            (config.LANDMARKS['THUMB_CMC'], config.LANDMARKS['THUMB_MCP'], config.LANDMARKS['THUMB_IP']),
            (config.LANDMARKS['INDEX_MCP'], config.LANDMARKS['INDEX_PIP'], config.LANDMARKS['INDEX_DIP']),
            (config.LANDMARKS['MIDDLE_MCP'], config.LANDMARKS['MIDDLE_PIP'], config.LANDMARKS['MIDDLE_DIP']),
            (config.LANDMARKS['RING_MCP'], config.LANDMARKS['RING_PIP'], config.LANDMARKS['RING_DIP']),
            (config.LANDMARKS['PINKY_MCP'], config.LANDMARKS['PINKY_PIP'], config.LANDMARKS['PINKY_DIP'])
        ]
        
        finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
        
        for i, (joint1, joint2, joint3) in enumerate(finger_joints):
            # Calculate angle (simplified - would need proper 3D angle calculation)
            # This is a placeholder for actual angle calculation
            finger_angles[finger_names[i]] = 0
        
        return finger_angles