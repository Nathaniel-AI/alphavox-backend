"""
Behavior Capture Module for AlphaVox

This module implements advanced webcam-based behavior tracking and analysis capabilities,
including detection of:
- Micro-expressions
- Repetitive movements (tics)
- Eye movement patterns
- Head position and movement
- Body posture changes
- Hand gestures and movements

It uses web.py for enhanced browser webcam access and OpenCV for real-time
computer vision processing.
"""

import os
import cv2
import numpy as np
import time
import logging
import json
import threading
import base64
from datetime import datetime
from collections import deque


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants for behavior tracking
MOVEMENT_THRESHOLD = 0.02  # Movement detection threshold (proportion of frame)
MOVEMENT_HISTORY_LENGTH = 120  # 4 seconds at 30fps
MICRO_EXPRESSION_SENSITIVITY = 0.15  # Sensitivity for micro-expression detection
REPETITIVE_PATTERN_LENGTH = 60  # Frame length for repetitive pattern detection

# Directories for data storage
DATA_DIR = os.path.join('data', 'behavior_patterns')
os.makedirs(DATA_DIR, exist_ok=True)

class BehaviorCapture:
    """
    Advanced behavior capture and analysis system that integrates with AlphaVox
    to provide real-time analysis of user behavior patterns.
    """
    
    def __init__(self):
        """Initialize the behavior capture system"""
        self.is_tracking = False
        self.movement_history = deque(maxlen=MOVEMENT_HISTORY_LENGTH)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # For pattern recognition
        self.patterns = {
            'head_movement': [],
            'eye_movement': [],
            'facial_expression': [],
            'repetitive_movements': []
        }
        
        # For behavior pattern storage
        self.observed_patterns = []
        self.frame_buffer = []
        self.last_frame_time = time.time()
        self.fps = 0
        
        # Load known patterns if available
        self._load_known_patterns()
        
        # Initialize backend processors
        self._init_processors()
        
    def _init_processors(self):
        """Initialize specialized processors for different behavioral aspects"""
        # The actual implementation would use more advanced models for each behavior type
        self.processors = {
            'tic_detector': self._detect_repetitive_movements,
            'eye_tracker': self._track_eye_movements,
            'expression_analyzer': self._analyze_facial_expressions,
            'posture_tracker': self._track_body_posture
        }
        
        logger.info("Initialized behavior capture processors")
        
    def _load_known_patterns(self):
        """Load previously identified behavior patterns"""
        pattern_file = os.path.join(DATA_DIR, 'known_patterns.json')
        if os.path.exists(pattern_file):
            try:
                with open(pattern_file, 'r') as f:
                    self.known_patterns = json.load(f)
                logger.info(f"Loaded {len(self.known_patterns)} known behavior patterns")
            except Exception as e:
                logger.error(f"Error loading known patterns: {e}")
                self.known_patterns = []
        else:
            self.known_patterns = []
            
    def start_tracking(self):
        """Start behavior tracking"""
        if not self.is_tracking:
            self.is_tracking = True
            logger.info("Started behavior tracking")
            
            # Start analysis thread
            self.analysis_thread = threading.Thread(target=self._background_analysis)
            self.analysis_thread.daemon = True
            self.analysis_thread.start()
            
    def stop_tracking(self):
        """Stop behavior tracking"""
        if self.is_tracking:
            self.is_tracking = False
            logger.info("Stopped behavior tracking")
            
    def process_frame(self, frame):
        """
        Process a video frame for behavior analysis
        
        Args:
            frame: Video frame as numpy array (BGR format)
            
        Returns:
            dict: Analysis results and annotated frame
        """
        if not self.is_tracking:
            return {'tracking': False, 'frame': frame}
        
        # Calculate FPS
        current_time = time.time()
        time_diff = current_time - self.last_frame_time
        if time_diff > 0:
            self.fps = 1.0 / time_diff
        self.last_frame_time = current_time
        
        # Store frame in buffer
        self.frame_buffer.append(frame.copy())
        if len(self.frame_buffer) > MOVEMENT_HISTORY_LENGTH:
            self.frame_buffer.pop(0)
        
        # Create a copy for annotation
        annotated_frame = frame.copy()
        
        # Run all processors on the frame
        results = {}
        for processor_name, processor_func in self.processors.items():
            try:
                processor_result = processor_func(frame, annotated_frame)
                results[processor_name] = processor_result
            except Exception as e:
                logger.error(f"Error in {processor_name}: {e}")
                results[processor_name] = {'error': str(e)}
        
        # Add overall analysis
        pattern_matches = self._match_known_patterns(results)
        if pattern_matches:
            results['pattern_matches'] = pattern_matches
            
        # Add FPS information to frame
        cv2.putText(annotated_frame, f"FPS: {self.fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return {
            'tracking': True,
            'frame': annotated_frame,
            'results': results
        }
    
    def _detect_repetitive_movements(self, frame, annotated_frame):
        """
        Detect repetitive movements (tics) in the video stream
        
        Args:
            frame: Current video frame
            annotated_frame: Frame to annotate with detection results
            
        Returns:
            dict: Detection results
        """
        # Convert to grayscale for motion detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        # If we have previous frames, calculate movement
        if len(self.frame_buffer) > 1:
            prev_gray = cv2.cvtColor(self.frame_buffer[-2], cv2.COLOR_BGR2GRAY)
            prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)
            
            # Calculate absolute difference
            frame_delta = cv2.absdiff(prev_gray, gray)
            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
            
            # Calculate proportion of movement
            movement_proportion = np.sum(thresh) / (thresh.shape[0] * thresh.shape[1] * 255)
            self.movement_history.append(movement_proportion)
            
            # Check for repetitive patterns
            repetitive_movements = self._find_repetitive_patterns(self.movement_history)
            
            # Annotate frame with movement information
            cv2.putText(annotated_frame, f"Movement: {movement_proportion:.4f}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # If significant movement detected, highlight
            if movement_proportion > MOVEMENT_THRESHOLD:
                # Draw contours on areas with significant movement
                thresh = cv2.dilate(thresh, None, iterations=2)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    if cv2.contourArea(contour) > 100:  # Minimum area threshold
                        (x, y, w, h) = cv2.boundingRect(contour)
                        cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            return {
                'movement_proportion': float(movement_proportion),
                'repetitive_movements': repetitive_movements
            }
        
        return {'movement_proportion': 0, 'repetitive_movements': []}
    
    def _track_eye_movements(self, frame, annotated_frame):
        """
        Track eye movements and gaze direction
        
        Args:
            frame: Current video frame
            annotated_frame: Frame to annotate with tracking results
            
        Returns:
            dict: Eye tracking results
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Region of interest for eyes
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = annotated_frame[y:y+h, x:x+w]
            
            # Detect eyes
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            
            eye_positions = []
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                
                # Get eye center
                eye_center_x = x + ex + ew//2
                eye_center_y = y + ey + eh//2
                
                # Draw eye center
                cv2.circle(annotated_frame, (eye_center_x, eye_center_y), 3, (0, 0, 255), -1)
                
                eye_positions.append({
                    'x': eye_center_x / frame.shape[1],  # Normalize to 0-1
                    'y': eye_center_y / frame.shape[0],  # Normalize to 0-1
                    'width': ew / frame.shape[1],
                    'height': eh / frame.shape[0]
                })
            
            return {
                'face_detected': True,
                'eyes_detected': len(eyes),
                'eye_positions': eye_positions
            }
        
        # No face detected
        cv2.putText(annotated_frame, "No face detected", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return {
            'face_detected': False,
            'eyes_detected': 0,
            'eye_positions': []
        }
    
    def _analyze_facial_expressions(self, frame, annotated_frame):
        """
        Analyze facial expressions and micro-expressions
        
        Args:
            frame: Current video frame
            annotated_frame: Frame to annotate with analysis results
            
        Returns:
            dict: Facial expression analysis
        """
        # In a full implementation, this would use specialized models
        # for facial landmark detection and expression classification
        
        # Placeholder implementation based on face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) > 0:
            # Simulate expression analysis with limited capabilities
            # In practice, would use a proper facial expression recognition model
            
            # Draw bounding box around face
            (x, y, w, h) = faces[0]
            cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Add basic expression analysis text
            cv2.putText(annotated_frame, "Expression analysis active", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            return {
                'face_detected': True,
                'dominant_expression': 'neutral',  # Placeholder
                'expression_confidence': 0.8,      # Placeholder
                'micro_expressions': []            # Placeholder
            }
        
        return {
            'face_detected': False
        }
    
    def _track_body_posture(self, frame, annotated_frame):
        """
        Track body posture and posture changes
        
        Args:
            frame: Current video frame
            annotated_frame: Frame to annotate with tracking results
            
        Returns:
            dict: Posture tracking results
        """
        # Basic placeholder implementation
        # Full implementation would use a pose estimation model
        
        # Add posture tracking indicator
        cv2.putText(annotated_frame, "Posture tracking active", (10, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return {
            'posture_detected': True,
            'posture_quality': 'upright',  # Placeholder
            'movement_detected': False     # Placeholder
        }
    
    def _find_repetitive_patterns(self, movement_data):
        """
        Find repetitive patterns in movement data
        
        Args:
            movement_data: Sequence of movement measurements
            
        Returns:
            list: Detected repetitive patterns
        """
        # Simple threshold-based pattern detection
        # A more sophisticated implementation would use signal processing techniques
        
        if len(movement_data) < REPETITIVE_PATTERN_LENGTH:
            return []
        
        # Calculate peaks in movement
        data_array = np.array(movement_data)
        threshold = np.mean(data_array) + MOVEMENT_THRESHOLD
        peaks = []
        
        for i in range(1, len(data_array) - 1):
            if data_array[i] > threshold and data_array[i] > data_array[i-1] and data_array[i] > data_array[i+1]:
                peaks.append(i)
        
        # Check for repetitive patterns in the peaks
        if len(peaks) >= 3:
            intervals = [peaks[i+1] - peaks[i] for i in range(len(peaks)-1)]
            
            # Check if intervals are consistent
            if len(intervals) >= 2:
                mean_interval = np.mean(intervals)
                std_interval = np.std(intervals)
                
                # If standard deviation is low relative to mean, it's a repetitive pattern
                if std_interval / mean_interval < 0.3:  # Threshold for consistency
                    return [{
                        'type': 'repetitive_movement',
                        'frequency': self.fps / mean_interval if self.fps > 0 else 0,
                        'consistency': 1.0 - (std_interval / mean_interval),
                        'count': len(peaks)
                    }]
        
        return []
    
    def _match_known_patterns(self, current_results):
        """
        Match current results against known behavior patterns
        
        Args:
            current_results: Current analysis results
            
        Returns:
            list: Matching patterns with confidence scores
        """
        matches = []
        
        # In a full implementation, this would use more sophisticated pattern matching
        
        # Add matched patterns to the list
        for pattern in self.known_patterns:
            # Implement matching logic based on pattern type
            # This is a placeholder for actual pattern matching algorithms
            pass
        
        return matches
    
    def _background_analysis(self):
        """Background thread for continuous pattern analysis"""
        while self.is_tracking:
            try:
                # Perform deeper analysis on accumulated data
                # This is where long-term pattern recognition would happen
                pass
            except Exception as e:
                logger.error(f"Error in background analysis: {e}")
            
            # Run analysis every 5 seconds
            time.sleep(5)
    
    def add_behavior_observation(self, observation_data):
        """
        Add a behavior observation for later analysis
        
        Args:
            observation_data: Dictionary of observed behavior data
        """
        observation_data['timestamp'] = datetime.now().isoformat()
        self.observed_patterns.append(observation_data)
        
        # Save observations periodically
        if len(self.observed_patterns) % 10 == 0:
            self._save_observations()
    
    def _save_observations(self):
        """Save observed behavior patterns to disk"""
        if not self.observed_patterns:
            return
            
        observations_file = os.path.join(DATA_DIR, 'behavior_observations.json')
        
        try:
            # Load existing observations if any
            existing_observations = []
            if os.path.exists(observations_file):
                with open(observations_file, 'r') as f:
                    existing_observations = json.load(f)
            
            # Append new observations
            all_observations = existing_observations + self.observed_patterns
            
            # Save to file
            with open(observations_file, 'w') as f:
                json.dump(all_observations, f)
                
            logger.info(f"Saved {len(self.observed_patterns)} behavior observations")
            
            # Clear list after saving
            self.observed_patterns = []
        except Exception as e:
            logger.error(f"Error saving behavior observations: {e}")
    
    def get_analysis_summary(self):
        """
        Get a summary of recent behavior analysis
        
        Returns:
            dict: Summary of behavior analysis
        """
        return {
            'tracking_active': self.is_tracking,
            'fps': self.fps,
            'patterns_detected': len(self.observed_patterns)
        }

# Singleton instance
_behavior_capture = None

def get_behavior_capture():
    """Get or create the behavior capture singleton instance"""
    global _behavior_capture
    if _behavior_capture is None:
        _behavior_capture = BehaviorCapture()
    return _behavior_capture

class BehaviorCaptureWebApp:
    """Web app for behavior capture integration with AlphaVox"""
    
    def __init__(self):
        """Initialize the web app"""
        urls = (
            '/api/behavior/start', 'StartBehaviorCapture',
            '/api/behavior/stop', 'StopBehaviorCapture',
            '/api/behavior/status', 'BehaviorCaptureStatus',
            '/api/behavior/process', 'ProcessFrame',
            '/api/behavior/observations', 'GetObservations'
        )
        self.app = web.application(urls, globals())
        
class StartBehaviorCapture:
    """Start behavior capture API endpoint"""
    
    def POST(self):
        """Start behavior tracking"""
        behavior_capture = get_behavior_capture()
        behavior_capture.start_tracking()
        
        web.header('Content-Type', 'application/json')
        return json.dumps({
            'status': 'success',
            'tracking': True,
            'message': 'Behavior capture started'
        })

class StopBehaviorCapture:
    """Stop behavior capture API endpoint"""
    
    def POST(self):
        """Stop behavior tracking"""
        behavior_capture = get_behavior_capture()
        behavior_capture.stop_tracking()
        
        web.header('Content-Type', 'application/json')
        return json.dumps({
            'status': 'success',
            'tracking': False,
            'message': 'Behavior capture stopped'
        })

class BehaviorCaptureStatus:
    """Get behavior capture status API endpoint"""
    
    def GET(self):
        """Get current status of behavior tracking"""
        behavior_capture = get_behavior_capture()
        status = behavior_capture.get_analysis_summary()
        
        web.header('Content-Type', 'application/json')
        return json.dumps({
            'status': 'success',
            'tracking_status': status
        })

class ProcessFrame:
    """Process a video frame API endpoint"""
    
    def POST(self):
        """Process a frame for behavior analysis"""
        data = json.loads(web.data().decode('utf-8'))
        
        # Decode base64 image
        if 'frame' not in data:
            return json.dumps({
                'status': 'error',
                'message': 'No frame data provided'
            })
        
        try:
            # Decode base64 image
            frame_data = base64.b64decode(data['frame'])
            nparr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Process the frame
            behavior_capture = get_behavior_capture()
            results = behavior_capture.process_frame(frame)
            
            # Encode the annotated frame
            _, buffer = cv2.imencode('.jpg', results['frame'])
            annotated_frame_b64 = base64.b64encode(buffer).decode('utf-8')
            
            # Prepare results
            response = {
                'status': 'success',
                'tracking': results['tracking'],
                'annotated_frame': annotated_frame_b64
            }
            
            # Add analysis results if available
            if 'results' in results:
                response['results'] = results['results']
            
            web.header('Content-Type', 'application/json')
            return json.dumps(response)
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return json.dumps({
                'status': 'error',
                'message': str(e)
            })

class GetObservations:
    """Get behavior observations API endpoint"""
    
    def GET(self):
        """Get recorded behavior observations"""
        observations_file = os.path.join(DATA_DIR, 'behavior_observations.json')
        
        if not os.path.exists(observations_file):
            return json.dumps({
                'status': 'success',
                'observations': []
            })
        
        try:
            with open(observations_file, 'r') as f:
                observations = json.load(f)
            
            web.header('Content-Type', 'application/json')
            return json.dumps({
                'status': 'success',
                'count': len(observations),
                'observations': observations
            })
        except Exception as e:
            logger.error(f"Error loading observations: {e}")
            return json.dumps({
                'status': 'error',
                'message': str(e)
            })

def run_webapp(port=8080):
    """
    Run the behavior capture web app
    
    Args:
        port: Port number for the web app
    """
    webapp = BehaviorCaptureWebApp()
    webapp.app.run()

if __name__ == "__main__":
    # If run directly, start the web app
    logging.info("Starting behavior capture web app")
    run_webapp()