import cv2

# Camera settings
CAMERA_ID = 0  # 0 for built-in webcam, 1 for external
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS_TARGET = 30

# MediaPipe settings
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
MAX_NUM_HANDS = 2

# Finger landmarks (MediaPipe indices)
LANDMARKS = {
    'WRIST': 0,
    'THUMB_CMC': 1,
    'THUMB_MCP': 2,
    'THUMB_IP': 3,
    'THUMB_TIP': 4,
    'INDEX_MCP': 5,
    'INDEX_PIP': 6,
    'INDEX_DIP': 7,
    'INDEX_TIP': 8,
    'MIDDLE_MCP': 9,
    'MIDDLE_PIP': 10,
    'MIDDLE_DIP': 11,
    'MIDDLE_TIP': 12,
    'RING_MCP': 13,
    'RING_PIP': 14,
    'RING_DIP': 15,
    'RING_TIP': 16,
    'PINKY_MCP': 17,
    'PINKY_PIP': 18,
    'PINKY_DIP': 19,
    'PINKY_TIP': 20
}

# Gesture thresholds
PINCH_THRESHOLD = 0.05  # Distance for pinch detection
FINGER_RAISED_THRESHOLD = 0.1  # For vertical finger detection

# Display settings
WINDOW_NAME = "Hand Gesture Recognition"
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1
COLORS = {
    'WHITE': (255, 255, 255),
    'GREEN': (0, 255, 0),
    'RED': (0, 0, 255),
    'BLUE': (255, 0, 0),
    'YELLOW': (0, 255, 255)
}