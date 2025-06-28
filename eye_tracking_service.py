import logging
import random
import numpy as np
import time
import cv2

class EyeTrackingService:
    """
    Service for eye tracking.
    
    In a full implementation, this would use OpenCV and potentially MediaPipe face mesh
    to track eye positions from a camera feed. This version provides a simulation.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing EyeTrackingService")
        
        # Last known eye position
        self.current_position = {'x': 0.5, 'y': 0.5, 'region': 'center'}
        
        # Generate random eye movements for simulation
        self.start_simulation()
        
        self.logger.info("EyeTrackingService initialized")
    
    def start_simulation(self):
        """Start the eye movement simulation"""
        self.simulation_active = True
        self.last_update = time.time()
        
        # Starting position (center of screen)
        self.target_position = {'x': 0.5, 'y': 0.5}
        self.current_position = {'x': 0.5, 'y': 0.5, 'region': 'center'}
    
    def update_simulation(self):
        """Update the simulated eye position"""
        current_time = time.time()
        
        # Update simulated eye position every few seconds
        if current_time - self.last_update > 2.0:
            # Generate new random target position
            self.target_position = {
                'x': random.uniform(0.1, 0.9),
                'y': random.uniform(0.1, 0.9)
            }
            self.last_update = current_time
        
        # Gradually move toward target position
        delta_x = (self.target_position['x'] - self.current_position['x']) * 0.1
        delta_y = (self.target_position['y'] - self.current_position['y']) * 0.1
        
        self.current_position['x'] += delta_x
        self.current_position['y'] += delta_y
        
        # Determine region based on position
        x, y = self.current_position['x'], self.current_position['y']
        
        if x < 0.33:
            if y < 0.33:
                region = 'top_left'
            elif y > 0.66:
                region = 'bottom_left'
            else:
                region = 'left'
        elif x > 0.66:
            if y < 0.33:
                region = 'top_right'
            elif y > 0.66:
                region = 'bottom_right'
            else:
                region = 'right'
        else:
            if y < 0.33:
                region = 'top'
            elif y > 0.66:
                region = 'bottom'
            else:
                region = 'center'
        
        self.current_position['region'] = region
    
    def get_eye_position(self):
        """Get the current eye position and region"""
        # Update simulation before returning
        self.update_simulation()
        return self.current_position
    
    def process_camera_frame(self, frame):
        """
        Process a camera frame to detect eye position.
        
        In a full implementation, this would use OpenCV and face detection.
        This version simply adds overlay to the frame for demonstration.
        """
        # Update simulated eye position
        self.update_simulation()
        
        # Draw a simple overlay showing gaze direction
        height, width = frame.shape[:2]
        x_pos = int(self.current_position['x'] * width)
        y_pos = int(self.current_position['y'] * height)
        
        # Draw gaze point
        cv2.circle(frame, (x_pos, y_pos), 10, (0, 255, 0), -1)
        
        # Draw regions grid (for reference)
        cv2.line(frame, (width//3, 0), (width//3, height), (100, 100, 100), 1)
        cv2.line(frame, (2*width//3, 0), (2*width//3, height), (100, 100, 100), 1)
        cv2.line(frame, (0, height//3), (width, height//3), (100, 100, 100), 1)
        cv2.line(frame, (0, 2*height//3), (width, 2*height//3), (100, 100, 100), 1)
        
        # Label current region
        cv2.putText(frame, f"Region: {self.current_position['region']}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame
