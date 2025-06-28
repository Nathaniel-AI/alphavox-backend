"""
AlphaVox - Real Speech Recognition Engine
-----------------------------------------
This module provides real speech recognition capabilities for AlphaVox
using actual microphone input instead of simulated data.
"""

import os
import time
import threading
import logging
import numpy as np
import sounddevice as sd
import queue
from typing import Dict, Any, List, Optional, Callable, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables
AUDIO_SAMPLE_RATE = 16000
MIN_SPEECH_DURATION = 0.5  # seconds
SILENCE_THRESHOLD = 0.1
AUDIO_CACHE_DIR = "audio_cache"

# Create directory for audio cache if it doesn't exist
if not os.path.exists(AUDIO_CACHE_DIR):
    os.makedirs(AUDIO_CACHE_DIR)

class RealSpeechRecognitionEngine:
    """
    Engine for processing speech input from a real microphone
    and converting it to text.
    """
    
    def __init__(self, language: str = "en-US"):
        """
        Initialize the speech recognition engine.
        
        Args:
            language: The default language code to use for recognition
        """
        self.language = language
        self.is_listening = False
        self.callbacks = []
        self.audio_buffer = []
        self.last_speech_time = 0
        
        # Speech detection parameters
        self.silence_threshold = SILENCE_THRESHOLD
        self.min_speech_duration = MIN_SPEECH_DURATION
        
        # Audio processing queue
        self.audio_queue = queue.Queue()
        
        # Available devices
        self.devices = sd.query_devices()
        logger.info(f"Available audio devices: {len(self.devices)}")
        for i, device in enumerate(self.devices):
            logger.info(f"Device {i}: {device['name']}")
        
        # Try to find a suitable input device
        self.input_device = None
        for i, device in enumerate(self.devices):
            if device['max_input_channels'] > 0:
                self.input_device = i
                logger.info(f"Selected input device {i}: {device['name']}")
                break
        
        if self.input_device is None:
            logger.warning("No suitable input device found, using default")
            self.input_device = sd.default.device[0]
        
        logger.info(f"Real Speech Recognition Engine initialized with language: {language}")
    
    def start_listening(self, callback: Optional[Callable] = None) -> bool:
        """
        Start listening for speech input from the microphone.
        
        Args:
            callback: Optional callback function to call when speech is recognized
                     Function signature: callback(text: str, confidence: float, metadata: Dict[str, Any])
        
        Returns:
            True if listening started successfully, False otherwise
        """
        if self.is_listening:
            logger.warning("Speech recognition is already active")
            return False
        
        if callback:
            self.callbacks.append(callback)
        
        self.is_listening = True
        
        # Start the audio processing thread
        self._start_audio_processing_thread()
        
        # Start audio stream from microphone
        try:
            self.stream = sd.InputStream(
                samplerate=AUDIO_SAMPLE_RATE,
                channels=1,
                callback=self._audio_callback,
                device=self.input_device
            )
            self.stream.start()
            logger.info("Started audio stream from microphone")
        except Exception as e:
            logger.error(f"Failed to start audio stream: {e}")
            self.is_listening = False
            return False
        
        logger.info("Real speech recognition started")
        return True
    
    def stop_listening(self) -> bool:
        """
        Stop listening for speech input.
        
        Returns:
            True if listening stopped successfully, False otherwise
        """
        if not self.is_listening:
            logger.warning("Speech recognition is not active")
            return False
        
        self.is_listening = False
        
        # Stop the audio stream
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        
        logger.info("Real speech recognition stopped")
        return True
    
    def _audio_callback(self, indata, frames, time, status):
        """Callback function for the audio stream"""
        if status:
            logger.warning(f"Audio callback status: {status}")
        
        # Put the audio data in the queue
        self.audio_queue.put(indata.copy())
    
    def _start_audio_processing_thread(self):
        """Start a background thread to process audio input"""
        thread = threading.Thread(target=self._audio_processing_loop)
        thread.daemon = True
        thread.start()
    
    def _audio_processing_loop(self):
        """Main audio processing loop that runs in a background thread"""
        try:
            while self.is_listening:
                try:
                    # Get audio data from the queue with a timeout
                    audio_chunk = self.audio_queue.get(timeout=1.0)
                    
                    # Process the audio chunk
                    self._process_audio_chunk(audio_chunk)
                except queue.Empty:
                    # No audio data available, continue
                    continue
        except Exception as e:
            logger.error(f"Error in audio processing loop: {e}")
            self.is_listening = False
    
    def _process_audio_chunk(self, audio_chunk: np.ndarray):
        """
        Process an audio chunk to detect speech and perform recognition.
        
        Args:
            audio_chunk: NumPy array containing audio samples
        """
        # Add the chunk to our buffer
        self.audio_buffer.append(audio_chunk.flatten())
        
        # Limit buffer size to avoid memory issues
        max_buffer_size = int(AUDIO_SAMPLE_RATE * 5)  # 5 seconds of audio
        total_samples = sum(len(chunk) for chunk in self.audio_buffer)
        while total_samples > max_buffer_size and self.audio_buffer:
            removed = self.audio_buffer.pop(0)
            total_samples -= len(removed)
        
        # Check if there is speech in the audio
        if self._detect_speech(audio_chunk):
            self.last_speech_time = time.time()
            
            # If we have enough speech, process it
            if self._check_speech_duration():
                # Convert combined buffer to a single array
                combined_audio = np.concatenate(self.audio_buffer)
                
                # Perform speech recognition
                text, confidence, metadata = self._process_speech(combined_audio)
                
                if text:
                    # Notify all registered callbacks
                    for callback in self.callbacks:
                        callback(text, confidence, metadata)
                    
                    # Clear the buffer after processing
                    self.audio_buffer = []
    
    def _detect_speech(self, audio_chunk: np.ndarray) -> bool:
        """
        Detect if the audio chunk contains speech.
        
        Args:
            audio_chunk: NumPy array containing audio samples
        
        Returns:
            True if speech is detected, False otherwise
        """
        # Energy-based speech detection
        energy = np.mean(np.abs(audio_chunk))
        return energy > self.silence_threshold
    
    def _check_speech_duration(self) -> bool:
        """
        Check if we have captured enough speech to process.
        
        Returns:
            True if we have enough speech, False otherwise
        """
        if not self.audio_buffer:
            return False
        
        # Check if enough time has passed since last speech detection
        time_since_last_speech = time.time() - self.last_speech_time
        return time_since_last_speech > self.min_speech_duration
    
    def _process_speech(self, audio_data: np.ndarray) -> Tuple[str, float, Dict[str, Any]]:
        """
        Process speech data and convert to text using available libraries.
        
        In a real implementation, this would use a speech recognition API
        like OpenAI Whisper, Google Speech-to-Text, etc.
        
        For now, we'll use a basic keyword detection as a simple example.
        
        Args:
            audio_data: NumPy array containing audio samples
        
        Returns:
            Tuple of (recognized text, confidence score, metadata)
        """
        # For a real implementation, you should integrate with a speech recognition service
        # For this example, we'll perform a very simple detection based on audio energy patterns
        
        # Calculate energy patterns in the audio
        frame_size = int(AUDIO_SAMPLE_RATE * 0.02)  # 20ms frames
        frames = [audio_data[i:i+frame_size] for i in range(0, len(audio_data), frame_size)]
        energies = [np.mean(np.abs(frame)) for frame in frames if len(frame) == frame_size]
        
        if not energies:
            return "", 0.0, {"error": "No audio data"}
        
        # Calculate audio features
        avg_energy = np.mean(energies)
        energy_variance = np.var(energies)
        zero_crossings = sum(1 for i in range(1, len(audio_data)) if audio_data[i-1] * audio_data[i] < 0)
        zero_crossing_rate = zero_crossings / len(audio_data)
        
        # Log audio features for debugging
        logger.debug(f"Audio features: energy={avg_energy:.4f}, variance={energy_variance:.4f}, zcr={zero_crossing_rate:.4f}")
        
        # Basic keyword detection based on audio patterns
        # This is a very simplified example and should be replaced with a proper speech recognition API
        
        # If there's significant energy and variation, we assume speech is present
        if avg_energy > self.silence_threshold * 2 and energy_variance > 0.001:
            # In a real implementation, this would be the result from a speech recognition API
            # For now, we return a placeholder message indicating we detected audio
            text = "I detected speech but need a speech recognition API to understand it."
            confidence = avg_energy / (self.silence_threshold * 4)  # Scale confidence based on energy
            confidence = min(max(confidence, 0.1), 0.9)  # Limit to reasonable range
            
            # Log the detection
            logger.info(f"Speech detected: Energy={avg_energy:.4f}, Confidence={confidence:.2f}")
            
            return text, confidence, {
                "language": self.language,
                "duration": len(audio_data) / AUDIO_SAMPLE_RATE,
                "timestamp": time.time(),
                "audio_features": {
                    "energy": float(avg_energy),
                    "variance": float(energy_variance),
                    "zero_crossing_rate": float(zero_crossing_rate)
                }
            }
        else:
            # No speech detected or confidence too low
            return "", 0.0, {"error": "No clear speech detected"}
    
    def set_language(self, language: str) -> bool:
        """
        Set the recognition language.
        
        Args:
            language: Language code (e.g., "en-US", "fr-FR", "es-ES")
        
        Returns:
            True if language was set successfully, False otherwise
        """
        self.language = language
        logger.info(f"Recognition language set to: {language}")
        return True
    
    def adjust_sensitivity(self, silence_threshold: float, min_speech_duration: float) -> bool:
        """
        Adjust the speech detection sensitivity parameters.
        
        Args:
            silence_threshold: Energy threshold for detecting speech
            min_speech_duration: Minimum duration of speech to trigger recognition
        
        Returns:
            True if parameters were adjusted successfully, False otherwise
        """
        if silence_threshold <= 0 or min_speech_duration <= 0:
            logger.error("Invalid sensitivity parameters")
            return False
        
        self.silence_threshold = silence_threshold
        self.min_speech_duration = min_speech_duration
        
        logger.info(f"Sensitivity adjusted: threshold={silence_threshold}, duration={min_speech_duration}")
        return True
    
    def get_audio_devices(self) -> List[Dict[str, Any]]:
        """
        Get a list of available audio input devices.
        
        Returns:
            List of audio device information
        """
        devices = []
        for i, device in enumerate(self.devices):
            if device['max_input_channels'] > 0:
                devices.append({
                    'id': i,
                    'name': device['name'],
                    'channels': device['max_input_channels'],
                    'default': i == sd.default.device[0]
                })
        return devices
    
    def set_input_device(self, device_id: int) -> bool:
        """
        Set the audio input device.
        
        Args:
            device_id: Device ID to use for audio input
            
        Returns:
            True if device was set successfully, False otherwise
        """
        if device_id < 0 or device_id >= len(self.devices):
            logger.error(f"Invalid device ID: {device_id}")
            return False
        
        if self.devices[device_id]['max_input_channels'] <= 0:
            logger.error(f"Device {device_id} has no input channels")
            return False
        
        # If currently listening, stop and restart with new device
        was_listening = self.is_listening
        if was_listening:
            self.stop_listening()
        
        self.input_device = device_id
        logger.info(f"Input device set to {device_id}: {self.devices[device_id]['name']}")
        
        if was_listening:
            self.start_listening()
        
        return True


# Singleton instance
_real_speech_recognition_engine = None

def get_real_speech_recognition_engine() -> RealSpeechRecognitionEngine:
    """Get the singleton instance of the real speech recognition engine"""
    global _real_speech_recognition_engine
    if _real_speech_recognition_engine is None:
        _real_speech_recognition_engine = RealSpeechRecognitionEngine()
    return _real_speech_recognition_engine