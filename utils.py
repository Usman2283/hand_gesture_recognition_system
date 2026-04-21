import cv2
import numpy as np
from datetime import datetime

def calculate_distance(point1, point2):
    return np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def normalize_coordinates(landmark, frame_width, frame_height):
    x = int(landmark.x * frame_width)
    y = int(landmark.y * frame_height)
    return (x, y)

def draw_fancy_hand(frame, hand_landmarks, connections, color=(0, 255, 0)):
    h, w = frame.shape[:2]
    
    # Draw connections (bones)
    for connection in connections:
        start_idx, end_idx = connection
        start_point = normalize_coordinates(hand_landmarks.landmark[start_idx], w, h)
        end_point = normalize_coordinates(hand_landmarks.landmark[end_idx], w, h)
        cv2.line(frame, start_point, end_point, color, 2)
    
    # Draw landmarks (joints)
    for landmark in hand_landmarks.landmark:
        x, y = normalize_coordinates(landmark, w, h)
        cv2.circle(frame, (x, y), 5, color, -1)
        cv2.circle(frame, (x, y), 7, (255, 255, 255), 1)

def put_text_with_background(frame, text, position, font_scale=0.7, 
                            font_color=(255, 255, 255), 
                            bg_color=(0, 0, 0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    
    # Get text size
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Draw background rectangle
    x, y = position
    cv2.rectangle(frame, (x - 5, y - text_h - 5), 
                  (x + text_w + 5, y + 5), bg_color, -1)
    
    # Draw text
    cv2.putText(frame, text, (x, y), font, font_scale, font_color, thickness)

def save_screenshot(frame, filename=None):
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"output/screenshot_{timestamp}.jpg"
    
    cv2.imwrite(filename, frame)
    return filename

def calculate_angle(a, b, c):
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    
    return np.degrees(angle)