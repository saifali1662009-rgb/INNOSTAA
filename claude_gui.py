import tkinter as tk
from tkinter import ttk
import threading
import queue
import time
import math
import random
import cv2
from PIL import Image, ImageTk

class BackgroundParticles:
    def __init__(self, canvas, width, height, blob_center, blob_radius):
        self.canvas = canvas
        self.width = width
        self.height = height
        self.cx, self.cy = blob_center
        self.safe_radius = blob_radius + 25     # 80 NO particles on blob

        self.count = 150
        self.particles = []
        self.speed_multiplier = 0.4  

        for _ in range(self.count):
            self.create_particle()

    def create_particle(self):
        while True:
            x = random.randint(0, self.width)
            y = random.randint(0, self.height)
            if math.dist((x, y), (self.cx, self.cy)) > self.safe_radius:
                break

        size = random.randint(2, 4)
        dx = random.uniform(-3.5, 3.5)
        dy = random.uniform(-3.5, 3.5)

        pid = self.canvas.create_oval(
            x, y, x+size, y+size,
            fill="#7FA3C7",
            outline=""
        )

        self.particles.append([pid, x, y, dx, dy])

    def set_state(self, state):
        if state == "idle":
            self.speed_multiplier = 1.5
        elif state == "listening":
            self.speed_multiplier = 3.5
        elif state == "speaking":
            self.speed_multiplier = 6.0

    def update(self):
        for p in self.particles:
            pid, x, y, dx, dy = p

            x += dx * self.speed_multiplier * 3.0
            y += dy * self.speed_multiplier * 3.0

            if x < 0 or x > self.width or y < 0 or y > self.height:
                x = random.randint(0, self.width)
                y = random.randint(0, self.height)

            dist = math.dist((x, y), (self.cx, self.cy))

            if dist < self.safe_radius:
                
                angle = math.atan2(y - self.cy, x - self.cx)
   
                repel_force = (self.safe_radius - dist) / self.safe_radius
            
                dx += math.cos(angle) * repel_force * 0.8
                dy += math.sin(angle) * repel_force * 0.8
            self.canvas.coords(pid, x, y, x+3, y+3)
            p[1], p[2] = x, y

class BlobAnimation:
    """Animated blob that responds to voice activity"""
    def __init__(self, canvas, center_x, center_y, base_radius=120):
        self.canvas = canvas
        self.center_x = center_x
        self.center_y = center_y
        self.base_radius = base_radius
        self.current_radius = base_radius
        self.target_radius = base_radius
        self.points = 12
        self.noise_offsets = [random.uniform(0, 100) for _ in range(self.points)]
        self.time_offset = 0
        self.is_speaking = False
        self.is_listening = False
        
    def get_points(self, dt):
        """Generate blob points"""
        self.time_offset += dt
    
        self.current_radius += (self.target_radius - self.current_radius) * 0.1
        
        points = []
        for i in range(self.points):
            angle = (i / self.points) * 2 * math.pi
            
            noise = math.sin(self.time_offset * 2 + self.noise_offsets[i]) * 15
   
            if self.is_speaking:
                noise += math.sin(self.time_offset * 8) * 25
            elif self.is_listening:
                noise += math.sin(self.time_offset * 4) * 12
            
            radius = self.current_radius + noise
            x = self.center_x + math.cos(angle) * radius
            y = self.center_y + math.sin(angle) * radius
            points.extend([x, y])
        
        return points
    
    def set_state(self, state):
        """Set blob state: 'idle', 'listening', 'speaking'"""
        if state == 'idle':
            self.is_speaking = False
            self.is_listening = False
            self.target_radius = self.base_radius
        elif state == 'listening':
            self.is_speaking = False
            self.is_listening = True
            self.target_radius = self.base_radius + 20
        elif state == 'speaking':
            self.is_speaking = True
            self.is_listening = False
            self.target_radius = self.base_radius + 35


class InnostaaGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Innostaa")
        self.root.geometry("1200x800")
        self.root.configure(bg="#0A1628")  
        
        # Colors from reference
        self.bg_color = "#0A1628"           # Dark blue background 
        self.dark_color = "#E8E8E8"         # Light text 
        self.accent_color = "#A0AEC0"       # Light gray for secondary text 
        self.blob_color = "#1E3A5F"         # Darker blue blob
        self.blob_shadow = "#5A5BB0A1"        # Darker shadow 
        self.blob_core_color = "#264C7A"    # Dark blue core
        
        self.current_state = "idle"
        self.mic_active = False
        self.current_subtitle = ""
        self.active_gesture = None
      
        self.status_queue = queue.Queue()
      
        self.blob = None
        self.blob_shadow = None
        self.blob_shape = None
        self.blob_core = None
        self.particles = None
        
        self.setup_ui()
    
        self.root.after(100, self.initialize_blob)

    def update_mouse_frame(self, frame):
        if not self.root.winfo_exists():
            return
    
        if self.active_gesture is None:
            return   
            
        def _update():
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            img = img.resize((380, 280))
            self.mouse_imgtk = ImageTk.PhotoImage(image=img)
    
            if not hasattr(self, "mouse_video"):
                self.mouse_video = tk.Label(self.mouse_frame, bg="#0D1B2A")
                self.mouse_video.pack(expand=True)
    
            self.mouse_video.config(image=self.mouse_imgtk)
    
        self.root.after(0, _update) 
        
    def setup_ui(self):
        """Setup the user interface"""
        
        header_frame = tk.Frame(self.root, bg=self.bg_color, height=80)
        header_frame.pack(fill=tk.X, padx=40, pady=(20, 0))
        header_frame.pack_propagate(False)
    
        title_label = tk.Label(
            header_frame, 
            text="Innostaa",
            font=("Monotype Corsiva", 40,"bold", "italic"),
            bg=self.bg_color,
            fg=self.dark_color
        )
        title_label.pack(side=tk.LEFT, pady=20)
     
        self.status_label = tk.Label(
            header_frame,
            text="● READY",
            font=("Arial", 11),
            bg=self.bg_color,
            fg=self.accent_color
        )
        self.status_label.pack(side=tk.RIGHT, pady=20)
       
        main_frame = tk.Frame(self.root, bg=self.bg_color)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(
            main_frame,
            bg=self.bg_color,
            highlightthickness=0
        )
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.mouse_frame = tk.Frame(
            main_frame,
            width=400,
            bg="#0D1B2A",
            highlightbackground="#AAAAAA",
            highlightthickness=2
        )
        self.mouse_frame.pack(side=tk.RIGHT, fill=tk.Y)
        self.mouse_frame.pack_propagate(False)
        
        self.mouse_label = tk.Label(
            self.mouse_frame,
            text="NO GESTURE FEATURE ACTIVE",
            font=("Arial", 14, "bold"),
            fg="#FF6B6B",
            bg="#0D1B2A"
        )
        self.mouse_label.pack(expand=True)        
        self.canvas.bind('<Configure>', self.on_window_resize)

        subtitle_frame = tk.Frame(self.root, bg=self.bg_color, height=100)
        subtitle_frame.pack(fill=tk.BOTH, expand=False, padx = 40, pady=(0,10))
        
        self.subtitle_label = tk.Label(
            subtitle_frame,
            text="",
            font=("Bradley Hand ITC", 16, "bold"),
            bg=self.bg_color,
            fg=self.dark_color,
            wraplength=900,
            justify=tk.CENTER
        )
        self.subtitle_label.pack(expand=True)
       
        footer_frame = tk.Frame(self.root, bg=self.bg_color, height=80)
        footer_frame.pack(fill=tk.X, padx=40, pady=(0, 20))
        footer_frame.pack_propagate(False)
    
        mic_container = tk.Frame(footer_frame, bg=self.bg_color)
        mic_container.pack(expand=True)
        
        self.mic_indicator = tk.Canvas(
            mic_container,
            width=20,
            height=20,
            bg=self.bg_color,
            highlightthickness=0
        )
        self.mic_indicator.pack(side=tk.LEFT, padx=(0, 10))
        
        self.mic_circle = self.mic_indicator.create_oval(
            2, 2, 18, 18,
            fill="#CCCCCC",
            outline=""
        )
        
        self.mic_label = tk.Label(
            mic_container,
            text="MICROPHONE: STANDBY",
            font=("Arial", 10),
            bg=self.bg_color,
            fg=self.accent_color
        )
        self.mic_label.pack(side=tk.LEFT)
        
    def on_window_resize(self, event=None):
        """Recenter blob when window is resized"""
        if self.blob is None:
            return
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        new_center_x = canvas_width // 2
        new_center_y = canvas_height // 2
        self.blob.center_x = new_center_x
        self.blob.center_y = new_center_y
        if self.particles:
           self.particles.cx = new_center_x
           self.particles.cy = new_center_y
           self.particles.width = canvas_width
           self.particles.height = canvas_height
        window_width = self.root.winfo_width()
        self.subtitle_label.config(wraplength=max(400, window_width - 200))  
            
    def initialize_blob(self):
        """Initialize blob after canvas is properly sized"""
     
        self.canvas.update()
        
   
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
    
        center_x = canvas_width // 2
        center_y = canvas_height // 2
        
        print(f"Canvas size: {canvas_width}x{canvas_height}")
        print(f"Blob center: ({center_x}, {center_y})")
        
        self.blob = BlobAnimation(self.canvas, center_x, center_y)

        initial_points = self.blob.get_points(0)

        self.particles = BackgroundParticles(
            self.canvas,
            canvas_width,
            canvas_height,
            (center_x, center_y),
            self.blob.base_radius
        )
     
        self.blob_shadow = self.canvas.create_polygon(
            initial_points, 
            fill=self.blob_shadow, 
            outline="", 
            smooth=True
        )
        
        self.blob_shape = self.canvas.create_polygon(
            initial_points, 
            fill=self.blob_color, 
            outline="", 
            smooth=True
        )

        core_radius = 40
        self.blob_core = self.canvas.create_oval(
            center_x - core_radius,
            center_y - core_radius,
            center_x + core_radius,
            center_y + core_radius,
            fill="#B5BCC8",
            outline=""
        )        
      
        self.start_animation()
        
    def start_animation(self):
        """Start the animation loop"""
        self.last_time = time.time()
        self.animate()
        
    def animate(self):
        """Animation loop"""
        if self.particles:
            self.particles.update()

        if self.blob is None:
       
            self.root.after(33, self.animate)
            return
        
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
    
        points = self.blob.get_points(dt)
        
    
        if len(points) >= 6: 
            self.canvas.coords(self.blob_shape, *points)
            
            shadow_points = []
            for i in range(0, len(points), 2):
                shadow_points.extend([points[i] + 8, points[i+1] + 8])
            self.canvas.coords(self.blob_shadow, *shadow_points)
            self.canvas.tag_lower(self.blob_shadow)
        
        core_radius = 40
        self.canvas.coords(
            self.blob_core,
            self.blob.center_x - core_radius,
            self.blob.center_y - core_radius,
            self.blob.center_x + core_radius,
            self.blob.center_y + core_radius
        )
        
        try:
            while True:
                status = self.status_queue.get_nowait()
                self.update_status(status)
        except queue.Empty:
            pass
        
        self.root.after(33, self.animate)  # ~30 FPS

    def set_gesture_state(self, feature_name=None):
        self.active_gesture = feature_name
    
        if feature_name:
            if hasattr(self, "mouse_label"):
                self.mouse_label.pack_forget()
            return
    
        if hasattr(self, "mouse_video"):
            self.mouse_video.destroy()
            del self.mouse_video
    
        self.mouse_label.config(text="NO GESTURE FEATURE ENABLED")
        self.mouse_label.pack(expand=True)  
        
    def update_status(self, status_dict):
        """Update GUI based on status from assistant"""
        state = status_dict.get('state', 'idle')
        subtitle = status_dict.get('subtitle', '')
        mic_active = status_dict.get('mic_active', False)
       
        if state != self.current_state:
            self.current_state = state
            if self.blob:
                self.blob.set_state(state)
          
            if state == 'listening':
                self.status_label.config(text="● LISTENING", fg="#4CAF50")
            elif state == 'speaking':
                self.status_label.config(text="● SPEAKING", fg="#2196F3")
            else:
                self.status_label.config(text="● READY", fg=self.accent_color)
   
        if subtitle != self.current_subtitle:
            self.current_subtitle = subtitle
            self.subtitle_label.config(text=subtitle)
       
        if mic_active != self.mic_active:
            self.mic_active = mic_active
            if mic_active:
                self.mic_indicator.itemconfig(self.mic_circle, fill="#4CAF50")
                self.mic_label.config(text="MICROPHONE: ACTIVE", fg="#4CAF50")
            else:
                self.mic_indicator.itemconfig(self.mic_circle, fill="#CCCCCC")
                self.mic_label.config(text="MICROPHONE: STANDBY", fg=self.accent_color)

        if state != self.current_state:
            self.current_state = state
        
            if self.blob:
                self.blob.set_state(state)
        
            if self.particles:
                self.particles.set_state(state)
                
    def send_status(self, state, subtitle="", mic_active=False):
        """Send status update to GUI"""
        self.status_queue.put({
            'state': state,
            'subtitle': subtitle,
            'mic_active': mic_active
        })

class InnostaaWithGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.gui = InnostaaGUI(self.root)

    def start_gui(self):
        pass 

    def mainloop(self):
        self.root.mainloop()

    def update_status(self, state, subtitle="", mic_active=False):
        if self.gui:
            self.gui.send_status(state, subtitle, mic_active)
            
    def set_gesture_state(self, feature_name=None):
        if self.gui:
            self.gui.set_gesture_state(feature_name)
        
if __name__ == "__main__":
    root = tk.Tk()
    gui = InnostaaGUI(root)
    
    def demo_cycle():
        states = [
            ('idle', '', False),
            ('listening', 'Listening...', True),
            ('idle', 'What can I help you with?', False),
            ('speaking', 'Sure, I can help you with that.', False),
            ('idle', '', False),
        ]
        
        def cycle_states(index=0):
            if index < len(states):
                state, subtitle, mic = states[index]
                gui.send_status(state, subtitle, mic)
                root.after(2000, lambda: cycle_states(index + 1))
        
        root.after(1000, cycle_states)
    
    demo_cycle()
    root.mainloop()
