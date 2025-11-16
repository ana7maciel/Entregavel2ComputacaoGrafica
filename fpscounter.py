# fpscounter.py
import time
import numpy as np
from OpenGL.GL import *
import ctypes


class FPSCounter:
    def __init__(self, average_over=30, stats_interval=5.0, warmup_time=2.0):
        self.frame_times = []
        self.average_over = average_over
        self.last_time = time.time()
        self.current_fps = 0.0
        
        # Text rendering resources
        self.text_vao = None
        self.text_vbo = None
        self.text_shader = None
        self.initialized = False
        
        # Simple 5x7 bitmap font for digits and common characters
        self.font_data = self._create_bitmap_font()
        
        # Statistics tracking
        self.stats_interval = stats_interval  # Interval in seconds
        self.warmup_time = warmup_time  # Time to discard at the beginning
        self.start_time = time.time()  # Track when the counter started
        self.last_stats_time = time.time()
        self.fps_history = []  # Store all FPS values for statistics
        self.enable_stats_print = False  # Flag to enable/disable stats printing
        self.warmup_complete = False  # Flag to track if warmup period is over

    def update(self):
        current_time = time.time()
        frame_time = current_time - self.last_time
        self.last_time = current_time

        # Add current frame time to the list
        self.frame_times.append(frame_time)

        # Keep only the last N frame times
        if len(self.frame_times) > self.average_over:
            self.frame_times.pop(0)

        # Calculate average FPS
        if self.frame_times:
            avg_frame_time = sum(self.frame_times) / len(self.frame_times)
            # Prevent division by zero
            if avg_frame_time > 0.0:
                self.current_fps = 1.0 / avg_frame_time
            else:
                self.current_fps = 0.0
        
        # Track FPS for statistics
        if self.enable_stats_print and self.current_fps > 0:
            # Check if warmup period is complete
            elapsed_since_start = current_time - self.start_time
            
            if not self.warmup_complete:
                if elapsed_since_start >= self.warmup_time:
                    self.warmup_complete = True
                    print(f"[FPS Counter] Warmup period complete ({self.warmup_time:.1f}s). Starting statistics collection.")
                    # Reset the stats timer after warmup
                    self.last_stats_time = current_time
                    self.fps_history = []
            else:
                # Only collect statistics after warmup period
                self.fps_history.append(self.current_fps)
                
                # Check if it's time to print statistics
                if current_time - self.last_stats_time >= self.stats_interval:
                    self._print_stats()
                    self.last_stats_time = current_time
                    self.fps_history = []  # Reset for next interval

        return self.current_fps

    def get_fps(self):
        return self.current_fps
    
    def get_fps_string(self):
        """Returns formatted FPS string"""
        return f"FPS: {self.current_fps:.1f}"
    
    def enable_stats_printing(self, enable=True):
        """Enable or disable periodic FPS statistics printing"""
        self.enable_stats_print = enable
        if enable:
            self.start_time = time.time()
            self.last_stats_time = time.time()
            self.fps_history = []
            self.warmup_complete = False
            print(f"FPS statistics enabled (printing every {self.stats_interval} seconds)")
            print(f"[FPS Counter] Warmup period: {self.warmup_time:.1f}s (initial frames will be discarded)")
        else:
            print("FPS statistics disabled")
    
    def _print_stats(self):
        """Print FPS statistics (min, max, average)"""
        if not self.fps_history:
            return
        
        min_fps = min(self.fps_history)
        max_fps = max(self.fps_history)
        avg_fps = sum(self.fps_history) / len(self.fps_history)
        
        # Calculate elapsed time since start (excluding warmup)
        elapsed_total = time.time() - self.start_time - self.warmup_time
        
        print(f"\n{'='*50}")
        print(f"FPS Statistics (last {self.stats_interval:.1f} seconds):")
        print(f"  Minimum FPS: {min_fps:.2f}")
        print(f"  Maximum FPS: {max_fps:.2f}")
        print(f"  Average FPS: {avg_fps:.2f}")
        print(f"  Samples: {len(self.fps_history)}")
        print(f"  Elapsed time: {elapsed_total:.1f}s (warmup excluded)")
        print(f"{'='*50}\n")
    
    def _create_bitmap_font(self):
        """Create a simple 5x7 bitmap font for digits and letters"""
        # Each character is represented as a 5x7 grid (1 = filled, 0 = empty)
        font = {
            '0': [
                [0,1,1,1,0],
                [1,0,0,0,1],
                [1,0,0,1,1],
                [1,0,1,0,1],
                [1,1,0,0,1],
                [1,0,0,0,1],
                [0,1,1,1,0]
            ],
            '1': [
                [0,0,1,0,0],
                [0,1,1,0,0],
                [0,0,1,0,0],
                [0,0,1,0,0],
                [0,0,1,0,0],
                [0,0,1,0,0],
                [0,1,1,1,0]
            ],
            '2': [
                [0,1,1,1,0],
                [1,0,0,0,1],
                [0,0,0,0,1],
                [0,0,1,1,0],
                [0,1,0,0,0],
                [1,0,0,0,0],
                [1,1,1,1,1]
            ],
            '3': [
                [1,1,1,1,0],
                [0,0,0,0,1],
                [0,0,0,0,1],
                [0,1,1,1,0],
                [0,0,0,0,1],
                [0,0,0,0,1],
                [1,1,1,1,0]
            ],
            '4': [
                [0,0,0,1,0],
                [0,0,1,1,0],
                [0,1,0,1,0],
                [1,0,0,1,0],
                [1,1,1,1,1],
                [0,0,0,1,0],
                [0,0,0,1,0]
            ],
            '5': [
                [1,1,1,1,1],
                [1,0,0,0,0],
                [1,1,1,1,0],
                [0,0,0,0,1],
                [0,0,0,0,1],
                [1,0,0,0,1],
                [0,1,1,1,0]
            ],
            '6': [
                [0,1,1,1,0],
                [1,0,0,0,0],
                [1,0,0,0,0],
                [1,1,1,1,0],
                [1,0,0,0,1],
                [1,0,0,0,1],
                [0,1,1,1,0]
            ],
            '7': [
                [1,1,1,1,1],
                [0,0,0,0,1],
                [0,0,0,1,0],
                [0,0,1,0,0],
                [0,1,0,0,0],
                [0,1,0,0,0],
                [0,1,0,0,0]
            ],
            '8': [
                [0,1,1,1,0],
                [1,0,0,0,1],
                [1,0,0,0,1],
                [0,1,1,1,0],
                [1,0,0,0,1],
                [1,0,0,0,1],
                [0,1,1,1,0]
            ],
            '9': [
                [0,1,1,1,0],
                [1,0,0,0,1],
                [1,0,0,0,1],
                [0,1,1,1,1],
                [0,0,0,0,1],
                [0,0,0,0,1],
                [0,1,1,1,0]
            ],
            'F': [
                [1,1,1,1,1],
                [1,0,0,0,0],
                [1,0,0,0,0],
                [1,1,1,1,0],
                [1,0,0,0,0],
                [1,0,0,0,0],
                [1,0,0,0,0]
            ],
            'P': [
                [1,1,1,1,0],
                [1,0,0,0,1],
                [1,0,0,0,1],
                [1,1,1,1,0],
                [1,0,0,0,0],
                [1,0,0,0,0],
                [1,0,0,0,0]
            ],
            'S': [
                [0,1,1,1,1],
                [1,0,0,0,0],
                [1,0,0,0,0],
                [0,1,1,1,0],
                [0,0,0,0,1],
                [0,0,0,0,1],
                [1,1,1,1,0]
            ],
            ':': [
                [0,0,0,0,0],
                [0,0,1,0,0],
                [0,0,1,0,0],
                [0,0,0,0,0],
                [0,0,1,0,0],
                [0,0,1,0,0],
                [0,0,0,0,0]
            ],
            '.': [
                [0,0,0,0,0],
                [0,0,0,0,0],
                [0,0,0,0,0],
                [0,0,0,0,0],
                [0,0,0,0,0],
                [0,1,1,0,0],
                [0,1,1,0,0]
            ],
            ' ': [
                [0,0,0,0,0],
                [0,0,0,0,0],
                [0,0,0,0,0],
                [0,0,0,0,0],
                [0,0,0,0,0],
                [0,0,0,0,0],
                [0,0,0,0,0]
            ]
        }
        return font
    
    def initialize_text_rendering(self, window_width, window_height):
        """Initialize OpenGL resources for text rendering"""
        self.window_width = window_width
        self.window_height = window_height
        
        # Create simple shader for text rendering
        vertex_shader_src = """
        #version 330 core
        layout (location = 0) in vec2 aPos;
        uniform vec2 uScreenSize;
        
        void main() {
            // Convert from screen coordinates to NDC
            vec2 pos = (aPos / uScreenSize) * 2.0 - 1.0;
            pos.y = -pos.y;  // Flip Y coordinate
            gl_Position = vec4(pos, 0.0, 1.0);
        }
        """
        
        fragment_shader_src = """
        #version 330 core
        uniform vec3 uColor;
        out vec4 FragColor;
        
        void main() {
            FragColor = vec4(uColor, 1.0);
        }
        """
        
        # Compile shaders
        vertex_shader = glCreateShader(GL_VERTEX_SHADER)
        glShaderSource(vertex_shader, vertex_shader_src)
        glCompileShader(vertex_shader)
        
        fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)
        glShaderSource(fragment_shader, fragment_shader_src)
        glCompileShader(fragment_shader)
        
        # Create program
        self.text_shader = glCreateProgram()
        glAttachShader(self.text_shader, vertex_shader)
        glAttachShader(self.text_shader, fragment_shader)
        glLinkProgram(self.text_shader)
        
        glDeleteShader(vertex_shader)
        glDeleteShader(fragment_shader)
        
        # Create VAO/VBO for text quads
        self.text_vao = glGenVertexArrays(1)
        self.text_vbo = glGenBuffers(1)
        
        glBindVertexArray(self.text_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.text_vbo)
        glBufferData(GL_ARRAY_BUFFER, 6 * 2 * 4, None, GL_DYNAMIC_DRAW)  # 6 vertices, 2 floats each
        
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)
        
        self.initialized = True
    
    def render_char(self, char, x, y, size, color=(1.0, 1.0, 1.0)):
        """Render a single character using bitmap font"""
        if not self.initialized:
            return
        
        # Get the bitmap for this character
        if char not in self.font_data:
            return  # Skip unknown characters
        
        bitmap = self.font_data[char]
        pixel_size = size  # Size of each pixel in the bitmap
        
        glUseProgram(self.text_shader)
        glUniform2f(glGetUniformLocation(self.text_shader, "uScreenSize"), 
                   float(self.window_width), float(self.window_height))
        glUniform3f(glGetUniformLocation(self.text_shader, "uColor"), *color)
        
        glBindVertexArray(self.text_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.text_vbo)
        
        # Render each pixel of the bitmap
        for row in range(7):  # 7 rows
            for col in range(5):  # 5 columns
                if bitmap[row][col] == 1:
                    px = x + col * pixel_size
                    py = y + row * pixel_size
                    px2 = px + pixel_size
                    py2 = py + pixel_size
                    
                    vertices = np.array([
                        px, py,
                        px, py2,
                        px2, py2,
                        px, py,
                        px2, py2,
                        px2, py
                    ], dtype=np.float32)
                    
                    glBufferSubData(GL_ARRAY_BUFFER, 0, vertices.nbytes, vertices)
                    glDrawArrays(GL_TRIANGLES, 0, 6)
        
        glBindVertexArray(0)
        glUseProgram(0)
    
    def render_text_simple(self, text, x, y, size=1.0, color=(1.0, 1.0, 1.0)):
        """Render text using bitmap font"""
        if not self.initialized:
            return
        
        char_width = size * 6  # 5 pixels + 1 pixel spacing
        current_x = x
        
        for char in text:
            self.render_char(char, current_x, y, size, color)
            current_x += char_width
    
    def render_fps(self, x=10, y=10, size=1.5, color=(0.0, 1.0, 0.0)):
        """Render FPS counter at specified position"""
        if not self.initialized:
            return
        
        # Save OpenGL state
        old_depth_test = glIsEnabled(GL_DEPTH_TEST)
        old_blend = glIsEnabled(GL_BLEND)
        
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        self.render_text_simple(self.get_fps_string(), x, y, size, color)
        
        # Restore state
        if not old_blend:
            glDisable(GL_BLEND)
        if old_depth_test:
            glEnable(GL_DEPTH_TEST)
    
    def cleanup(self):
        """Clean up OpenGL resources"""
        if self.initialized:
            glDeleteVertexArrays(1, [self.text_vao])
            glDeleteBuffers(1, [self.text_vbo])
            glDeleteProgram(self.text_shader)
            self.initialized = False