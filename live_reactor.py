"""
AlphaVox - Live Reactor Module
-----------------------------
Simulates or processes live multimodal input through the TemporalNonverbalEngine
and generates enhanced human-quality responses in real-time.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Tuple, Union

from nonverbal_engine import NonverbalEngine

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LiveReactor:
    """
    Live Reactor processes real-time multimodal inputs and provides enhanced responses.
    
    This class:
    - Takes live input streams from gesture, eye tracking, and emotion recognition
    - Processes them through the TemporalNonverbalEngine
    - Generates human-quality responses with intent understanding
    - Provides real-time interpretations and feedback
    """
    
    def __init__(self, use_lstm: bool = True):
        """
        Initialize the Live Reactor.
        
        Args:
            use_lstm: Whether to use advanced LSTM models if available
        """
        # Initialize the  nonverbal engine
        self.engine = engine temporal()
        
        # Track last processed time for throttling
        self.last_processed_time = 0
        self.processing_interval = 0.1  # seconds
        
        # Response history
        self.response_history = []
        self.max_response_history = 20
        
        # Response enhancement options
        self.enhance_responses = True
        self.output_detail_level = "comprehensive"  # comprehensive, moderate, minimal
        
        logger.info("Live Reactor initialized")
    
    def process_live_input(self, gesture=None, eye=None, emotion=None) -> Dict[str, Any]:
        """
        Process a single frame of input from any combination of sources (gesture, eye, emotion)
        
        Args:
            gesture: List of gesture features [wrist_x, wrist_y, elbow_angle, shoulder_angle]
            eye: List of eye movement features [gaze_x, gaze_y, blink_rate]
            emotion: List of emotion features [facial_tension, mouth_curve, eye_openness, eyebrow_position, perspiration]

        Returns:
            Dictionary with processing results and enhanced response
        """
        # Throttle processing to avoid excessive CPU usage
        current_time = time.time()
        if current_time - self.last_processed_time < self.processing_interval:
            # Return the most recent result if throttled
            if self.response_history:
                return self.response_history[-1]
            return {"status": "throttled"}
        
        self.last_processed_time = current_time
        
        # Use default values if features aren't provided
        if gesture is None:
            gesture = [0.5, 0.5, 90.0, 90.0]  # Default neutral position
        
        if eye is None:
            eye = [0.5, 0.5, 3.0]  # Default center gaze
        
        if emotion is None:
            emotion = [0.5, 0.5, 0.5, 0.5, 0.3]  # Default neutral emotion
        
        # Process through the temporal engine
        result = self.engine.process_multimodal_sequence(
            gesture_features=gesture,
            eye_features=eye,
            emotion_features=emotion
        )
        
        # Format the response based on detail level
        formatted_result = self._format_result(result)
        
        # Store in history
        self.response_history.append(formatted_result)
        if len(self.response_history) > self.max_response_history:
            self.response_history.pop(0)
        
        # Print formatted output to console
        self._print_formatted_output(formatted_result)
        
        return formatted_result
    
    def _format_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format the result based on the selected detail level.
        
        Args:
            result: Raw result from the engine
            
        Returns:
            Formatted result dictionary
        """
        if self.output_detail_level == "comprehensive":
            # Include all details
            return result
        
        elif self.output_detail_level == "moderate":
            # Include essential information
            return {
                'timestamp': result['timestamp'],
                'primary_type': result['primary_type'],
                'primary_result': result['primary_result'],
                'enhanced_response': result['enhanced_response']
            }
        
        else:  # minimal
            # Include only the most essential information
            return {
                'primary_type': result['primary_type'],
                'expression': result['primary_result']['expression'],
                'enhanced_response': result['enhanced_response']
            }
    
    def _print_formatted_output(self, result: Dict[str, Any]):
        """
        Print nicely formatted output to the console.
        
        Args:
            result: Formatted result to print
        """
        print("\n[AlphaVox Response]")
        print(f"→ Type: {result['primary_type']}")
        print(f"→ Expression: {result['primary_result']['expression']}")
        print(f"→ Intent: {result['primary_result']['intent']}")
        print(f"→ Confidence: {result['primary_result']['confidence']:.2f}")
        print(f"→ Message: {result['enhanced_response']}")
        print("────────────────────────────\n")
    
    def set_processing_interval(self, interval: float):
        """
        Set the processing interval for throttling.
        
        Args:
            interval: Processing interval in seconds
        """
        if interval > 0:
            self.processing_interval = interval
            logger.info(f"Processing interval set to {interval} seconds")
        else:
            logger.warning("Processing interval must be greater than 0")
    
    def set_output_detail_level(self, level: str):
        """
        Set the output detail level.
        
        Args:
            level: Output detail level ("comprehensive", "moderate", "minimal")
        """
        if level in ["comprehensive", "moderate", "minimal"]:
            self.output_detail_level = level
            logger.info(f"Output detail level set to {level}")
        else:
            logger.warning(f"Invalid output detail level: {level}")
    
    def get_response_history(self, count: int = 5) -> List[Dict[str, Any]]:
        """
        Get recent response history.
        
        Args:
            count: Number of recent responses to retrieve
            
        Returns:
            List of recent responses
        """
        return self.response_history[-count:]


# Create singleton instance
_live_reactor = None

def get_live_reactor():
    """Get the singleton instance of the Live Reactor"""
    global _live_reactor
    if _live_reactor is None:
        _live_reactor = LiveReactor()
    return _live_reactor


# Example usage
def process_live_input(gesture=None, eye=None, emotion=None):
    """
    Process a single frame of input from any combination of sources (gesture, eye, emotion)
    
    Args:
        gesture: List of gesture features [wrist_x, wrist_y, elbow_angle, shoulder_angle]
        eye: List of eye movement features [gaze_x, gaze_y, blink_rate]
        emotion: List of emotion features [facial_tension, mouth_curve, eye_openness, eyebrow_position, perspiration]

    Returns:
        None
    """
    reactor = get_live_reactor()
    result = reactor.process_live_input(
        gesture=gesture,
        eye=eye,
        emotion=emotion
    )
    return result


if __name__ == "__main__":
    # Example inputs (simulate a live signal feed)
    for _ in range(15):
        process_live_input(
            gesture=[0.3, 0.5, 45.0, 70.0],
            eye=[0.4, 0.5, 2.5],
            emotion=[0.2, 0.6, 0.8, 0.3, 0.1]
        )
        time.sleep(0.2)  # Short delay to simulate real-time processing
