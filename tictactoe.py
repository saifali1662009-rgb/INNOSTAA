import cv2
import mediapipe as mp
import numpy as np
import random
import time

class HandGestureTicTacToe:
    def __init__(self):
        # Initialize MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            max_num_hands=1
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Game board (3x3 grid)
        self.board = [['' for _ in range(3)] for _ in range(3)]
        self.player = 'O'  # Player 1 is O
        self.computer = 'X'  # Computer is X
        self.current_turn = self.player
        
        # Game state
        self.game_over = False
        self.winner = None
        self.selected_cell = [0, 0]  # [row, col]
        
        # Gesture detection
        self.hand_open = False
        self.last_gesture_time = 0
        self.gesture_cooldown = 1.0  # 1 second cooldown for placement
        
        # Camera
        self.cap = None
        self.running = False
        self.hands = None

        self.game_over_time = None
        
    def count_fingers(self, hand_landmarks):
        """Count extended fingers to detect open/closed hand"""
        fingers = []
        
        # Thumb
        if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
            fingers.append(1)
        else:
            fingers.append(0)
            
        # Other fingers
        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]
        
        for tip, pip in zip(finger_tips, finger_pips):
            if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
                fingers.append(1)
            else:
                fingers.append(0)
                
        return sum(fingers)
    
    def get_hand_position(self, hand_landmarks, frame_shape):
        """Get hand center position"""
        h, w, _ = frame_shape
        x = int(hand_landmarks.landmark[9].x * w)
        y = int(hand_landmarks.landmark[9].y * h)
        return x, y
    
    def map_position_to_cell(self, x, y, frame_shape):
        """Map hand position to grid cell"""
        h, w, _ = frame_shape
        
        # Grid boundaries (centered on screen)
        grid_size = min(h, w) - 100
        start_x = (w - grid_size) // 2
        start_y = (h - grid_size) // 2
        
        cell_size = grid_size // 3
        
        if start_x <= x <= start_x + grid_size and start_y <= y <= start_y + grid_size:
            col = (x - start_x) // cell_size
            row = (y - start_y) // cell_size
            return [min(row, 2), min(col, 2)]
        
        return self.selected_cell
    
    def draw_grid(self, frame):
        """Draw tic-tac-toe grid"""
        h, w, _ = frame.shape
        grid_size = min(h, w) - 100
        start_x = (w - grid_size) // 2
        start_y = (h - grid_size) // 2
        cell_size = grid_size // 3
        
        # Draw grid lines
        for i in range(4):
            # Vertical lines
            x = start_x + i * cell_size
            cv2.line(frame, (x, start_y), (x, start_y + grid_size), (255, 255, 255), 3)
            # Horizontal lines
            y = start_y + i * cell_size
            cv2.line(frame, (start_x, y), (start_x + grid_size, y), (255, 255, 255), 3)
        
        # Highlight selected cell
        if not self.game_over and self.current_turn == self.player:
            row, col = self.selected_cell
            x1 = start_x + col * cell_size + 5
            y1 = start_y + row * cell_size + 5
            x2 = x1 + cell_size - 10
            y2 = y1 + cell_size - 10
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 3)
        
        # Draw X's and O's
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == 'O':
                    cx = start_x + j * cell_size + cell_size // 2
                    cy = start_y + i * cell_size + cell_size // 2
                    cv2.circle(frame, (cx, cy), cell_size // 3, (0, 255, 0), 5)
                elif self.board[i][j] == 'X':
                    x1 = start_x + j * cell_size + cell_size // 4
                    y1 = start_y + i * cell_size + cell_size // 4
                    x2 = x1 + cell_size // 2
                    y2 = y1 + cell_size // 2
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 5)
                    cv2.line(frame, (x2, y1), (x1, y2), (255, 0, 0), 5)
        
        return frame, start_x, start_y, cell_size
    
    def check_winner(self):
        """Check if there's a winner"""
        # Check rows
        for row in self.board:
            if row[0] == row[1] == row[2] != '':
                return row[0]
        
        # Check columns
        for col in range(3):
            if self.board[0][col] == self.board[1][col] == self.board[2][col] != '':
                return self.board[0][col]
        
        # Check diagonals
        if self.board[0][0] == self.board[1][1] == self.board[2][2] != '':
            return self.board[0][0]
        if self.board[0][2] == self.board[1][1] == self.board[2][0] != '':
            return self.board[0][2]
        
        # Check for draw
        if all(self.board[i][j] != '' for i in range(3) for j in range(3)):
            return 'Draw'
        
        return None
    
    def computer_move(self):
        """Simple AI for computer move"""
        # Check if computer can win
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == '':
                    self.board[i][j] = self.computer
                    if self.check_winner() == self.computer:
                        return
                    self.board[i][j] = ''
        
        # Check if player can win and block
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == '':
                    self.board[i][j] = self.player
                    if self.check_winner() == self.player:
                        self.board[i][j] = self.computer
                        return
                    self.board[i][j] = ''
        
        # Take center if available
        if self.board[1][1] == '':
            self.board[1][1] = self.computer
            return
        
        # Take corners
        corners = [(0, 0), (0, 2), (2, 0), (2, 2)]
        random.shuffle(corners)
        for i, j in corners:
            if self.board[i][j] == '':
                self.board[i][j] = self.computer
                return
        
        # Take any available cell
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == '':
                    self.board[i][j] = self.computer
                    return

    def get_frame(self):
        if not self.running or self.cap is None:
           return None

        if not hasattr(self, "running") or not self.running:
            return None
    
        ret, frame = self.cap.read()
        if not ret:
            return None
    
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
    
        # Draw grid
        frame, start_x, start_y, cell_size = self.draw_grid(frame)
        
        # 🔁 Auto reset after 2 seconds
        if self.game_over and self.game_over_time:
            if time.time() - self.game_over_time > 2:
                self.reset_game()
                self.game_over_time = None

        # Player turn
        if results.multi_hand_landmarks and not self.game_over and self.current_turn == self.player:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
    
                finger_count = self.count_fingers(hand_landmarks)
                x, y = self.get_hand_position(hand_landmarks, frame.shape)
    
                if finger_count >= 3:
                    self.hand_open = True
                    self.selected_cell = self.map_position_to_cell(x, y, frame.shape)
    
                elif finger_count <= 2 and self.hand_open:
                    current_time = time.time()
                    if current_time - self.last_gesture_time > self.gesture_cooldown:
                        row, col = self.selected_cell
                        if self.board[row][col] == '':
                            self.board[row][col] = self.player
                            self.last_gesture_time = current_time
                            self.hand_open = False
    
                            winner = self.check_winner()
                            if winner:
                                self.game_over = True
                                self.winner = winner
                                self.game_over_time = time.time()
                            else:
                                self.current_turn = self.computer
    
        # Computer turn
        if not self.game_over and self.current_turn == self.computer:
            time.sleep(0.3)
            self.computer_move()
    
            winner = self.check_winner()
            if winner:
                self.game_over = True
                self.winner = winner
            else:
                self.current_turn = self.player
    
        # Status text
        if self.game_over:
            if self.winner == 'Draw':
                text = "Draw"
                color = (255, 255, 0)
            elif self.winner == self.player:
                text = "You Win"
                color = (0, 255, 0)
            else:
                text = "Computer Wins"
                color = (0, 0, 255)
        else:
            text = "Your Turn" if self.current_turn == self.player else "Computer Turn"
            color = (255, 255, 255)
    
        cv2.putText(frame, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
        return frame    

    def start(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            max_num_hands=1
        )
    
        self.reset_game()
        self.running = True
        
    def reset_game(self):
        self.board = [['' for _ in range(3)] for _ in range(3)]
        self.current_turn = self.player
        self.game_over = False
        self.winner = None
        self.selected_cell = [0, 0]
        self.hand_open = False
        self.last_gesture_time = 0
        self.game_over_time = None

    def stop(self):
        self.running = False
    
        if self.hands:
            self.hands.close()
            self.hands = None
    
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None

    

