"""
Input Analyzer for AlphaVox

This module provides advanced input analysis capabilities for AlphaVox,
including multilayer feature extraction, gesture sequence analysis,
and multimodal input fusion.

It works alongside the nonverbal engine and conversation engine to provide
a comprehensive understanding of user intent across different modalities.
"""

import logging
import json
import random
import time
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import numpy for advanced processing
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    logger.warning("Numpy not available, using fallback analysis methods")

class InputAnalyzer:
    """
    Advanced analyzer for multimodal inputs including gesture sequences,
    eye tracking patterns, and voice modulation.
    
    This class provides sophisticated analysis beyond the basic pattern
    matching in the primary engines.
    """
    
    def __init__(self):
        """Initialize the input analyzer"""
        self.temporal_window_size = 5  # Number of inputs to consider for temporal patterns
        self.recent_gestures = []
        self.recent_eye_positions = []
        self.recent_voice_patterns = []
        
        # Load pattern libraries
        self.gesture_sequences = self._load_gesture_sequences()
        self.eye_patterns = self._load_eye_patterns()
        self.audio_patterns = self._load_audio_patterns()
        
        # Initialize feature extractors
        self.gesture_features = GestureFeatureExtractor()
        self.eye_features = EyeTrackingFeatureExtractor()
        self.audio_features = AudioFeatureExtractor()
        
        # Temporal pattern detector
        self.temporal_detector = TemporalPatternDetector(self.temporal_window_size)
        
        logger.info("Input analyzer initialized")
    
    def _load_gesture_sequences(self) -> Dict[str, Dict[str, Any]]:
        """Load gesture sequence patterns from file or use defaults"""
        try:
            with open('data/gesture_sequences.json', 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # Default sequences
            return {
                'confirmation': {
                    'sequence': ['nod', 'nod'],
                    'meaning': 'Strong confirmation',
                    'confidence': 0.9
                },
                'strong_denial': {
                    'sequence': ['shake', 'shake'],
                    'meaning': 'Strong denial',
                    'confidence': 0.9
                },
                'uncertain': {
                    'sequence': ['tilt_head', 'shake', 'tilt_head'],
                    'meaning': 'Uncertainty',
                    'confidence': 0.8
                },
                'help_urgent': {
                    'sequence': ['point_up', 'wave'],
                    'meaning': 'Urgent help needed',
                    'confidence': 0.9
                },
                'feeling_overwhelmed': {
                    'sequence': ['rapid_blink', 'stimming'],
                    'meaning': 'Feeling overwhelmed',
                    'confidence': 0.85
                }
            }
    
    def _load_eye_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load eye tracking patterns from file or use defaults"""
        try:
            with open('data/eye_patterns.json', 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # Default patterns
            return {
                'scanning': {
                    'pattern': 'horizontal_sweep',
                    'meaning': 'Searching for information',
                    'confidence': 0.7
                },
                'detailed_focus': {
                    'pattern': 'sustained_gaze',
                    'meaning': 'Focused attention',
                    'confidence': 0.8
                },
                'rapid_shifting': {
                    'pattern': 'rapid_point_switches',
                    'meaning': 'Distraction or agitation',
                    'confidence': 0.7
                },
                'avoidance': {
                    'pattern': 'peripheral_focus',
                    'meaning': 'Avoiding direct engagement',
                    'confidence': 0.6
                },
                'selective_attention': {
                    'pattern': 'targeted_switching',
                    'meaning': 'Comparing options',
                    'confidence': 0.7
                }
            }
    
    def _load_audio_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load audio patterns from file or use defaults"""
        try:
            with open('data/audio_patterns.json', 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # Default patterns
            return {
                'rhythmic_pattern': {
                    'pattern': 'repetitive_beats',
                    'meaning': 'Attempting to communicate with rhythm',
                    'confidence': 0.7
                },
                'pitch_rise': {
                    'pattern': 'ascending_tone',
                    'meaning': 'Question or uncertainty',
                    'confidence': 0.8
                },
                'pitch_drop': {
                    'pattern': 'descending_tone',
                    'meaning': 'Conclusion or certainty',
                    'confidence': 0.8
                },
                'volume_increase': {
                    'pattern': 'crescendo',
                    'meaning': 'Increasing urgency or emphasis',
                    'confidence': 0.75
                },
                'variable_tone': {
                    'pattern': 'oscillating_pitch',
                    'meaning': 'Complex emotional state',
                    'confidence': 0.6
                }
            }
    
    def analyze_gesture(self, gesture_name: str, intensity: float = 1.0) -> Dict[str, Any]:
        """
        Analyze a gesture and extract features
        
        Args:
            gesture_name: Name of the gesture
            intensity: Intensity of the gesture (0.0 to 1.0)
            
        Returns:
            dict: Analysis results
        """
        # Add to recent gestures for temporal analysis
        self.recent_gestures.append({
            'name': gesture_name,
            'intensity': intensity,
            'timestamp': datetime.now().isoformat()
        })
        
        # Trim history
        if len(self.recent_gestures) > self.temporal_window_size:
            self.recent_gestures.pop(0)
        
        # Extract features
        features = self.gesture_features.extract_features(gesture_name, intensity)
        
        # Check for temporal patterns
        temporal_analysis = self.temporal_detector.analyze_gestures(self.recent_gestures)
        
        # Combine analyses
        analysis = {
            'basic_features': features,
            'temporal_patterns': temporal_analysis,
            'emotional_indicators': self._extract_emotional_indicators(gesture_name, intensity)
        }
        
        logger.debug(f"Gesture analysis: {analysis}")
        return analysis
    
    def analyze_eye_tracking(self, eye_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze eye tracking data
        
        Args:
            eye_data: Eye tracking data (position, fixation, etc.)
            
        Returns:
            dict: Analysis results
        """
        # Add to recent eye positions for temporal analysis
        self.recent_eye_positions.append({
            'data': eye_data,
            'timestamp': datetime.now().isoformat()
        })
        
        # Trim history
        if len(self.recent_eye_positions) > self.temporal_window_size:
            self.recent_eye_positions.pop(0)
        
        # Extract features
        features = self.eye_features.extract_features(eye_data)
        
        # Check for patterns
        pattern_match = self._match_eye_pattern(self.recent_eye_positions)
        
        # Combine analyses
        analysis = {
            'basic_features': features,
            'pattern_match': pattern_match,
            'attention_indicators': self._extract_attention_indicators(eye_data)
        }
        
        logger.debug(f"Eye tracking analysis: {analysis}")
        return analysis
    
    def analyze_audio(self, audio_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze audio input
        
        Args:
            audio_data: Audio input data (pattern, intensity, etc.)
            
        Returns:
            dict: Analysis results
        """
        # Add to recent voice patterns for temporal analysis
        self.recent_voice_patterns.append({
            'data': audio_data,
            'timestamp': datetime.now().isoformat()
        })
        
        # Trim history
        if len(self.recent_voice_patterns) > self.temporal_window_size:
            self.recent_voice_patterns.pop(0)
        
        # Extract features
        features = self.audio_features.extract_features(audio_data)
        
        # Check for patterns
        pattern_match = self._match_audio_pattern(self.recent_voice_patterns)
        
        # Combine analyses
        analysis = {
            'basic_features': features,
            'pattern_match': pattern_match,
            'vocal_indicators': self._extract_vocal_indicators(audio_data)
        }
        
        logger.debug(f"Audio analysis: {analysis}")
        return analysis
    
    def analyze_multimodal(self, gesture: Optional[str] = None, 
                          eye_data: Optional[Dict[str, Any]] = None,
                          audio_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze multimodal input combining gesture, eye tracking, and audio
        
        Args:
            gesture: Optional gesture name
            eye_data: Optional eye tracking data
            audio_data: Optional audio data
            
        Returns:
            dict: Comprehensive analysis results
        """
        analyses = {}
        
        # Analyze individual modalities if provided
        if gesture:
            analyses['gesture'] = self.analyze_gesture(gesture)
        
        if eye_data:
            analyses['eye_tracking'] = self.analyze_eye_tracking(eye_data)
        
        if audio_data:
            analyses['audio'] = self.analyze_audio(audio_data)
        
        # Perform multimodal fusion
        combined_analysis = self._fuse_modalities(analyses)
        
        return {
            'individual_analyses': analyses,
            'combined_analysis': combined_analysis
        }
    
    def _match_eye_pattern(self, recent_positions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Match eye movement to known patterns
        
        Args:
            recent_positions: Recent eye tracking data
            
        Returns:
            dict: Best matching pattern and confidence
        """
        if len(recent_positions) < 2:
            return {'pattern': 'unknown', 'confidence': 0.0}
        
        # Extract positions
        positions = [p['data'].get('position', {'x': 0.5, 'y': 0.5}) for p in recent_positions]
        
        # Analyze movement pattern
        x_positions = [p.get('x', 0.5) for p in positions]
        y_positions = [p.get('y', 0.5) for p in positions]
        
        # Simple pattern detection
        x_diff = max(x_positions) - min(x_positions)
        y_diff = max(y_positions) - min(y_positions)
        
        # Horizontal vs vertical movement
        if x_diff > 0.3 and y_diff < 0.1:
            pattern = 'horizontal_sweep'
            confidence = min(1.0, x_diff * 2)
        elif y_diff > 0.3 and x_diff < 0.1:
            pattern = 'vertical_sweep'
            confidence = min(1.0, y_diff * 2)
        elif x_diff < 0.1 and y_diff < 0.1:
            pattern = 'sustained_gaze'
            confidence = 0.8
        elif x_diff > 0.4 and y_diff > 0.4:
            pattern = 'rapid_point_switches'
            confidence = 0.7
        else:
            pattern = 'mixed_movement'
            confidence = 0.5
        
        return {
            'pattern': pattern,
            'confidence': confidence,
            'x_range': x_diff,
            'y_range': y_diff
        }
    
    def _match_audio_pattern(self, recent_patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Match audio input to known patterns
        
        Args:
            recent_patterns: Recent audio data
            
        Returns:
            dict: Best matching pattern and confidence
        """
        # Simplified implementation - would normally use real audio processing
        if len(recent_patterns) < 2:
            return {'pattern': 'unknown', 'confidence': 0.0}
        
        # Get the most recent pattern for simplified matching
        latest_pattern = recent_patterns[-1]['data'].get('pattern', 'unknown')
        
        # Map pattern to known categories
        pattern_mapping = {
            'hum': 'oscillating_pitch',
            'click': 'rhythmic_pattern',
            'distress': 'crescendo',
            'soft': 'descending_tone',
            'loud': 'crescendo',
            'short_vowel': 'staccato',
            'repeated_sound': 'repetitive_beats'
        }
        
        detected_pattern = pattern_mapping.get(latest_pattern, 'unknown')
        confidence = 0.7 if detected_pattern != 'unknown' else 0.3
        
        return {
            'pattern': detected_pattern,
            'confidence': confidence,
            'original_pattern': latest_pattern
        }
    
    def _extract_emotional_indicators(self, gesture_name: str, intensity: float) -> Dict[str, float]:
        """
        Extract emotional indicators from a gesture
        
        Args:
            gesture_name: Name of the gesture
            intensity: Intensity of the gesture
            
        Returns:
            dict: Emotional indicators
        """
        # Map gestures to emotional indicators
        emotion_map = {
            'nod': {'agreement': 0.8, 'positivity': 0.6},
            'shake': {'disagreement': 0.8, 'negativity': 0.6},
            'point_up': {'urgency': 0.7, 'attention': 0.8},
            'wave': {'greeting': 0.9, 'positivity': 0.7},
            'thumbs_up': {'approval': 0.9, 'positivity': 0.9},
            'thumbs_down': {'disapproval': 0.9, 'negativity': 0.8},
            'open_palm': {'stopping': 0.8, 'caution': 0.7},
            'stimming': {'distress': 0.7, 'arousal': 0.8},
            'rapid_blink': {'distress': 0.8, 'arousal': 0.9}
        }
        
        # Get default emotions for unknown gestures
        default_emotions = {'neutral': 0.5}
        
        # Get emotions for this gesture, scale by intensity
        emotions = emotion_map.get(gesture_name, default_emotions)
        scaled_emotions = {k: v * intensity for k, v in emotions.items()}
        
        return scaled_emotions
    
    def _extract_attention_indicators(self, eye_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract attention indicators from eye tracking data
        
        Args:
            eye_data: Eye tracking data
            
        Returns:
            dict: Attention indicators
        """
        # Extract relevant features
        position = eye_data.get('position', {'x': 0.5, 'y': 0.5})
        fixation = eye_data.get('fixation', 0.5)  # 0.0 to 1.0
        saccade_velocity = eye_data.get('saccade_velocity', 0.5)  # 0.0 to 1.0
        
        # Calculate attention indicators
        attention = fixation * (1 - saccade_velocity)  # High fixation, low saccade velocity = high attention
        distraction = saccade_velocity
        engagement = fixation * 0.7 + (1 - abs(position.get('x', 0.5) - 0.5)) * 0.3  # Center focus indicates engagement
        
        return {
            'attention': min(1.0, max(0.0, attention)),
            'distraction': min(1.0, max(0.0, distraction)),
            'engagement': min(1.0, max(0.0, engagement))
        }
    
    def _extract_vocal_indicators(self, audio_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract vocal indicators from audio data
        
        Args:
            audio_data: Audio data
            
        Returns:
            dict: Vocal indicators
        """
        # Extract relevant features
        pattern = audio_data.get('pattern', 'unknown')
        volume = audio_data.get('volume', 0.5)  # 0.0 to 1.0
        pitch = audio_data.get('pitch', 0.5)  # 0.0 to 1.0 (low to high)
        
        # Map patterns to indicators
        vocal_map = {
            'hum': {'contemplation': 0.7, 'calmness': 0.6},
            'click': {'precision': 0.8, 'deliberateness': 0.7},
            'distress': {'urgency': 0.9, 'distress': 0.8},
            'soft': {'calmness': 0.8, 'control': 0.7},
            'loud': {'urgency': 0.8, 'emphasis': 0.9},
            'short_vowel': {'acknowledgment': 0.7, 'attention': 0.6},
            'repeated_sound': {'emphasis': 0.8, 'persistence': 0.7}
        }
        
        # Get default indicators for unknown patterns
        default_indicators = {'neutral': 0.5}
        
        # Get indicators for this pattern
        indicators = vocal_map.get(pattern, default_indicators)
        
        # Add volume and pitch indicators
        if volume > 0.7:
            indicators['intensity'] = volume
            indicators['urgency'] = volume * 0.8
        
        if pitch > 0.7:
            indicators['questioning'] = pitch * 0.7
        elif pitch < 0.3:
            indicators['certainty'] = (1 - pitch) * 0.7
        
        return indicators
    
    def _fuse_modalities(self, analyses: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Fuse multiple modalities into a combined analysis
        
        Args:
            analyses: Individual analyses for each modality
            
        Returns:
            dict: Combined analysis
        """
        # Check if we have multiple modalities
        if len(analyses) < 2:
            # Single modality, just return a simplified version
            modality = list(analyses.keys())[0] if analyses else 'none'
            analysis = analyses.get(modality, {})
            return {
                'primary_modality': modality,
                'confidence': analysis.get('confidence', 0.5),
                'pattern': analysis.get('pattern', 'unknown'),
                'intent_indicators': {}
            }
        
        # Collect intent indicators from each modality
        intent_indicators = {}
        confidence_sum = 0.0
        count = 0
        
        # Process gesture analysis
        if 'gesture' in analyses:
            gesture_analysis = analyses['gesture']
            emotional_indicators = gesture_analysis.get('emotional_indicators', {})
            for emotion, value in emotional_indicators.items():
                intent_indicators[emotion] = intent_indicators.get(emotion, 0.0) + value
            confidence_sum += gesture_analysis.get('basic_features', {}).get('confidence', 0.5)
            count += 1
        
        # Process eye tracking analysis
        if 'eye_tracking' in analyses:
            eye_analysis = analyses['eye_tracking']
            attention_indicators = eye_analysis.get('attention_indicators', {})
            for attention, value in attention_indicators.items():
                intent_indicators[attention] = intent_indicators.get(attention, 0.0) + value
            confidence_sum += eye_analysis.get('pattern_match', {}).get('confidence', 0.5)
            count += 1
        
        # Process audio analysis
        if 'audio' in analyses:
            audio_analysis = analyses['audio']
            vocal_indicators = audio_analysis.get('vocal_indicators', {})
            for vocal, value in vocal_indicators.items():
                intent_indicators[vocal] = intent_indicators.get(vocal, 0.0) + value
            confidence_sum += audio_analysis.get('pattern_match', {}).get('confidence', 0.5)
            count += 1
        
        # Normalize intent indicators
        for intent, value in intent_indicators.items():
            intent_indicators[intent] = min(1.0, max(0.0, value / count))
        
        # Determine primary modality by confidence
        primary_modality = 'none'
        best_confidence = 0.0
        
        for modality, analysis in analyses.items():
            if modality == 'gesture':
                confidence = analysis.get('basic_features', {}).get('confidence', 0.0)
            else:
                confidence = analysis.get('pattern_match', {}).get('confidence', 0.0)
            
            if confidence > best_confidence:
                best_confidence = confidence
                primary_modality = modality
        
        # Calculate overall confidence
        overall_confidence = confidence_sum / count if count > 0 else 0.5
        
        # Determine most likely intent and emotional state
        sorted_indicators = sorted(intent_indicators.items(), key=lambda x: x[1], reverse=True)
        primary_indicators = sorted_indicators[:3] if sorted_indicators else []
        
        return {
            'primary_modality': primary_modality,
            'confidence': overall_confidence,
            'primary_indicators': primary_indicators,
            'intent_indicators': intent_indicators
        }


class GestureFeatureExtractor:
    """Extracts features from gestures"""
    
    def extract_features(self, gesture_name: str, intensity: float = 1.0) -> Dict[str, Any]:
        """
        Extract features from a gesture
        
        Args:
            gesture_name: Name of the gesture
            intensity: Intensity of the gesture
            
        Returns:
            dict: Extracted features
        """
        # Simple implementation - would be expanded with real gesture recognition
        gesture_features = {
            'name': gesture_name,
            'intensity': intensity,
            'confidence': 0.8 * intensity,  # Confidence drops with lower intensity
            'motion_type': self._classify_motion_type(gesture_name),
            'speed': random.uniform(0.7, 1.0) * intensity,  # Simulated speed
            'direction': self._get_gesture_direction(gesture_name)
        }
        
        return gesture_features
    
    def _classify_motion_type(self, gesture_name: str) -> str:
        """Classify the motion type of a gesture"""
        motion_types = {
            'nod': 'vertical_oscillation',
            'shake': 'horizontal_oscillation',
            'point_up': 'directional_static',
            'wave': 'horizontal_sweep',
            'thumbs_up': 'static_emblematic',
            'thumbs_down': 'static_emblematic',
            'open_palm': 'static_emblematic',
            'stimming': 'repetitive_self_directed',
            'rapid_blink': 'facial_repetitive'
        }
        
        return motion_types.get(gesture_name, 'unknown')
    
    def _get_gesture_direction(self, gesture_name: str) -> str:
        """Get the direction of a gesture"""
        directions = {
            'nod': 'vertical',
            'shake': 'horizontal',
            'point_up': 'upward',
            'wave': 'lateral',
            'thumbs_up': 'upward',
            'thumbs_down': 'downward',
            'open_palm': 'outward',
            'stimming': 'circular',
            'rapid_blink': 'forward'
        }
        
        return directions.get(gesture_name, 'neutral')


class EyeTrackingFeatureExtractor:
    """Extracts features from eye tracking data"""
    
    def extract_features(self, eye_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract features from eye tracking data
        
        Args:
            eye_data: Eye tracking data
            
        Returns:
            dict: Extracted features
        """
        # Extract position
        position = eye_data.get('position', {'x': 0.5, 'y': 0.5})
        region = eye_data.get('region', 'center')
        
        # Calculate distance from center
        center_x, center_y = 0.5, 0.5
        x = position.get('x', center_x)
        y = position.get('y', center_y)
        distance_from_center = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
        
        # Extract other features if available
        fixation = eye_data.get('fixation', 0.5)
        saccade_velocity = eye_data.get('saccade_velocity', 0.5)
        blink_rate = eye_data.get('blink_rate', 0.3)
        
        # Calculate derived features
        attention_score = fixation * (1 - saccade_velocity)
        agitation_score = saccade_velocity * blink_rate
        
        return {
            'position': position,
            'region': region,
            'distance_from_center': distance_from_center,
            'fixation': fixation,
            'saccade_velocity': saccade_velocity,
            'blink_rate': blink_rate,
            'attention_score': attention_score,
            'agitation_score': agitation_score
        }


class AudioFeatureExtractor:
    """Extracts features from audio data"""
    
    def extract_features(self, audio_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract features from audio data
        
        Args:
            audio_data: Audio data
            
        Returns:
            dict: Extracted features
        """
        # Extract basic features
        pattern = audio_data.get('pattern', 'unknown')
        volume = audio_data.get('volume', 0.5)
        pitch = audio_data.get('pitch', 0.5)
        duration = audio_data.get('duration', 1.0)
        
        # Calculate derived features
        intensity = volume * duration
        expressiveness = (volume * 0.5 + pitch * 0.5) * duration
        
        # Classify pattern
        pattern_category = self._classify_audio_pattern(pattern)
        
        return {
            'pattern': pattern,
            'pattern_category': pattern_category,
            'volume': volume,
            'pitch': pitch,
            'duration': duration,
            'intensity': intensity,
            'expressiveness': expressiveness
        }
    
    def _classify_audio_pattern(self, pattern: str) -> str:
        """Classify the audio pattern"""
        pattern_categories = {
            'hum': 'tonal',
            'click': 'percussive',
            'distress': 'vocal_emotional',
            'soft': 'vocal_controlled',
            'loud': 'vocal_projected',
            'short_vowel': 'vocal_brief',
            'repeated_sound': 'rhythmic'
        }
        
        return pattern_categories.get(pattern, 'unknown')


class TemporalPatternDetector:
    """Detects temporal patterns in sequences of inputs"""
    
    def __init__(self, window_size: int = 5):
        """
        Initialize the detector
        
        Args:
            window_size: Number of inputs to consider in temporal window
        """
        self.window_size = window_size
    
    def analyze_gestures(self, recent_gestures: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze a sequence of gestures for temporal patterns
        
        Args:
            recent_gestures: Recent gesture inputs
            
        Returns:
            dict: Detected patterns and confidence
        """
        if len(recent_gestures) < 2:
            return {'sequence': None, 'pattern': 'insufficient_data', 'confidence': 0.0}
        
        # Extract sequence of gesture names
        gesture_names = [g['name'] for g in recent_gestures]
        
        # Look for repetition patterns
        if len(gesture_names) >= 2 and all(g == gesture_names[0] for g in gesture_names):
            return {
                'sequence': gesture_names,
                'pattern': 'repetition',
                'confidence': 0.9,
                'intensity': sum(g.get('intensity', 1.0) for g in recent_gestures) / len(recent_gestures),
                'count': len(gesture_names)
            }
        
        # Look for alternating patterns
        if len(gesture_names) >= 4 and len(set(gesture_names)) == 2:
            # Check if it's a perfect alternation (A, B, A, B, ...)
            is_alternating = True
            for i in range(2, len(gesture_names)):
                if gesture_names[i] != gesture_names[i % 2]:
                    is_alternating = False
                    break
            
            if is_alternating:
                return {
                    'sequence': gesture_names,
                    'pattern': 'alternation',
                    'confidence': 0.8,
                    'items': list(set(gesture_names))
                }
        
        # Look for known sequences (example: nod followed by point_up)
        known_sequences = {
            ('nod', 'point_up'): {'pattern': 'affirmative_request', 'confidence': 0.8},
            ('shake', 'open_palm'): {'pattern': 'strong_rejection', 'confidence': 0.85},
            ('wave', 'thumbs_up'): {'pattern': 'friendly_approval', 'confidence': 0.8},
            ('nod', 'nod', 'thumbs_up'): {'pattern': 'strong_agreement', 'confidence': 0.9}
        }
        
        # Check all subsequences of appropriate length
        for seq_len in range(2, min(5, len(gesture_names) + 1)):
            for start in range(len(gesture_names) - seq_len + 1):
                subsequence = tuple(gesture_names[start:start+seq_len])
                if subsequence in known_sequences:
                    result = known_sequences[subsequence].copy()
                    result['sequence'] = subsequence
                    return result
        
        # No specific pattern detected
        return {
            'sequence': gesture_names,
            'pattern': 'unclassified',
            'confidence': 0.3
        }


# Singleton instance
_input_analyzer = None

def get_input_analyzer():
    """Get or create the input analyzer singleton"""
    global _input_analyzer
    if _input_analyzer is None:
        _input_analyzer = InputAnalyzer()
    return _input_analyzer