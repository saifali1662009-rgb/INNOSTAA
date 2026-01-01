import cv2
import mediapipe as mp
import pyautogui
import math
import time

frame_callback = None

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mouse_running = False


# Initialize PyAutoGUI
pyautogui.FAILSAFE = False
screen_w, screen_h = pyautogui.size()

# Smoothing variables
smooth_x, smooth_y = 0, 0
smooth_factor = 0.5

# Click detection variables
click_cooldown = 0.3
last_click_time = 0

# Scroll variables
prev_scroll_y = 0
scroll_sensitivity = 2

def set_frame_callback(cb):
    global frame_callback
    frame_callback = cb

def get_distance(p1, p2):
    """Calculate Euclidean distance between two points"""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def is_finger_up(landmarks, finger_tip, finger_pip):
    """Check if a finger is extended"""
    return landmarks[finger_tip].y < landmarks[finger_pip].y

def detect_gesture(hand_landmarks, frame_h, frame_w):
    """Detect hand gesture and return action"""
    landmarks = hand_landmarks.landmark
    
    # Get key points
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]
    
    # Check which fingers are up
    index_up = is_finger_up(landmarks, 8, 6)
    middle_up = is_finger_up(landmarks, 12, 10)
    ring_up = is_finger_up(landmarks, 16, 14)
    pinky_up = is_finger_up(landmarks, 20, 18)
    
    # Calculate distances
    thumb_index_dist = get_distance(
        (thumb_tip.x * frame_w, thumb_tip.y * frame_h),
        (index_tip.x * frame_w, index_tip.y * frame_h)
    )
    
    thumb_middle_dist = get_distance(
        (thumb_tip.x * frame_w, thumb_tip.y * frame_h),
        (middle_tip.x * frame_w, middle_tip.y * frame_h)
    )
    
    # Get index finger position for cursor
    index_x = int(index_tip.x * frame_w)
    index_y = int(index_tip.y * frame_h)
    
    # Gesture detection logic
    if index_up and not middle_up and not ring_up and not pinky_up:
        # One finger up - Move cursor
        if thumb_index_dist < 40:
            return "left_click", index_x, index_y
        return "move", index_x, index_y
    
    elif index_up and middle_up and not ring_up and not pinky_up:
        # Two fingers up
        if thumb_middle_dist < 40:
            return "right_click", index_x, index_y
        return "scroll", index_x, index_y
    
    return "none", index_x, index_y

def start_mouse():
    global smooth_x, smooth_y, last_click_time, prev_scroll_y
    global mouse_running
    mouse_running = True

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        
        print("Virtual Mouse Started!")
        print("Controls:")
        print("- One finger: Move cursor")
        print("- One finger + thumb touch: Left click")
        print("- Two fingers: Scroll mode")
        print("- Two fingers + thumb touch: Right click")
        print("Press 'q' to quit")
        
        while cap.isOpened() and mouse_running:
            success, frame = cap.read()
            if not success:
                continue
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            frame_h, frame_w, _ = frame.shape
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process hand detection
            results = hands.process(rgb_frame)

            if frame_callback:
               frame_callback(frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )                   
                    # Detect gesture
                    gesture, x, y = detect_gesture(hand_landmarks, frame_h, frame_w)
                    
                    # Map coordinates to screen
                    screen_x = int(x * screen_w / frame_w)
                    screen_y = int(y * screen_h / frame_h)
                    
                    # Smooth cursor movement
                    smooth_x = smooth_x * smooth_factor + screen_x * (1 - smooth_factor)
                    smooth_y = smooth_y * smooth_factor + screen_y * (1 - smooth_factor)
                    
                    current_time = time.time()
                    
                    if gesture == "move":
                        pyautogui.moveTo(smooth_x, smooth_y)
                        cv2.putText(frame, "MOVE", (10, 50), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    elif gesture == "left_click":
                        if current_time - last_click_time > click_cooldown:
                            pyautogui.click()
                            last_click_time = current_time
                        cv2.putText(frame, "LEFT CLICK", (10, 50), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    elif gesture == "right_click":
                        if current_time - last_click_time > click_cooldown:
                            pyautogui.rightClick()
                            last_click_time = current_time
                        cv2.putText(frame, "RIGHT CLICK", (10, 50), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    
                    elif gesture == "scroll":
                        scroll_delta = (prev_scroll_y - screen_y) // scroll_sensitivity
                        if abs(scroll_delta) > 5:
                            pyautogui.scroll(scroll_delta)
                        prev_scroll_y = screen_y
                        cv2.putText(frame, "SCROLL", (10, 50), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2) 
            

            # Display frame
            #cv2.imshow('Virtual Mouse - Hand Gesture Control', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

def stop_mouse():
    global mouse_running
    mouse_running = False

    