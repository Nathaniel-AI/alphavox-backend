import logging
import os
import pickle
import threading
import numpy as np
import spacy
from typing import Dict, List, Any, Optional
from collections import deque
from datetime import datetime
from neural_learning_core import NeuralLearningCore
from research_module import AlphaVoxResearchModule

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Data directory
DATA_DIR = 'data'
CONTEXT_FILE = os.path.join(DATA_DIR, 'input_context.pkl')
os.makedirs(DATA_DIR, exist_ok=True)

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    logger.error(f"Failed to load spaCy model: {str(e)}")
    logger.info("Install spaCy and download model: pip install spacy && python -m spacy download en_core_web_sm")
    # Create a basic fallback model that just tokenizes text
    nlp = spacy.blank("en")

class AlphaVoxInputProcessor:
    """Processes multi-modal inputs (gestures, symbols, text, sounds) for AlphaVox with NLU."""
    
    def __init__(self, model_dir: str = 'models', max_memory: int = 1000):
        """Initialize the processor with context tracking and research integration."""
        self.model_dir = model_dir
        self.memory = deque(maxlen=max_memory)
        self.context_window = {}
        self.lock = threading.Lock()
        self.nlc = NeuralLearningCore()
        self.research_module = AlphaVoxResearchModule()
        self.symbol_map = {
            'question': {'intent': 'ask_question', 'message': 'I have a question.', 'emotion': 'inquisitive'},
            'food': {'intent': 'request_food', 'message': "I'm hungry. I want food.", 'emotion': 'urgent'},
            'drink': {'intent': 'request_drink', 'message': "I'm thirsty. I want a drink.", 'emotion': 'urgent'},
            'happy': {'intent': 'express_joy', 'message': "I'm feeling happy!", 'emotion': 'positive'},
            'sad': {'intent': 'express_sadness', 'message': "I'm feeling sad.", 'emotion': 'negative'},
            'pain': {'intent': 'report_pain', 'message': "I'm in pain.", 'emotion': 'negative'},
        }
        self.gesture_map = {
            'Hand Up': {'intent': 'request_attention', 'message': 'I need attention.', 'emotion': 'urgent'},
            'Wave Left': {'intent': 'greeting', 'message': 'Hello!', 'emotion': 'positive'},
            'Wave Right': {'intent': 'greeting', 'message': 'Hi there!', 'emotion': 'positive'},
            'Head Jerk': {'intent': 'express_distress', 'message': "I'm feeling overwhelmed.", 'emotion': 'negative'}
        }
        self.load_context()
        self.update_from_research()
        logger.info("AlphaVoxInputProcessor initialized")

    def load_context(self):
        """Load context window from disk."""
        if os.path.exists(CONTEXT_FILE):
            try:
                with open(CONTEXT_FILE, 'rb') as f:
                    self.context_window = pickle.load(f)
                logger.info(f"Loaded context window with {len(self.context_window)} users")
            except Exception as e:
                logger.error(f"Error loading context: {str(e)}")
                self.context_window = {}

    def save_context(self):
        """Save context window to disk."""
        with self.lock:
            try:
                with open(CONTEXT_FILE, 'wb') as f:
                    pickle.dump(self.context_window, f)
                logger.info(f"Saved context window for {len(self.context_window)} users")
            except Exception as e:
                logger.error(f"Error saving context: {str(e)}")

    def update_from_research(self):
        """Update mappings from research insights."""
        try:
            research_updates = self.research_module.update_knowledge_base()
            for strategy in research_updates.get('updates_applied', {}).get('new_strategies', []):
                if strategy['type'] == 'communication' and 'pecs' in strategy['description'].lower():
                    self.symbol_map['request'] = {
                        'intent': 'request_item',
                        'message': 'I want something.',
                        'emotion': 'urgent'
                    }
            for intent in research_updates.get('updates_applied', {}).get('updated_intents', []):
                if intent['intent'] not in self.nlc.intent_weights:
                    self.nlc.intent_weights[intent['intent']] = intent['weight']
            logger.info("Updated input processor with research insights")
            self.save_context()
        except Exception as e:
            logger.error(f"Error updating from research: {str(e)}")

    def process_interaction(self, interaction: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """Process an interaction and integrate with NLC."""
        try:
            interaction = self._validate_interaction(interaction)
            interaction['context'] = self._add_context(interaction, user_id)
            
            if interaction['type'] == 'gesture':
                interaction = self._process_gesture(interaction)
            elif interaction['type'] == 'symbol':
                interaction = self._process_symbol(interaction)
            elif interaction['type'] == 'text':
                interaction = self._process_text(interaction)
            elif interaction['type'] == 'sound':
                interaction = self._process_sound(interaction)
            
            result = self.nlc.process_interaction(interaction, user_id)
            self.memory.append({
                'user_id': user_id,
                'interaction': interaction,
                'result': result,
                'timestamp': datetime.now()
            })
            self._update_context(user_id, interaction, result)
            self.save_context()
            
            logger.info(f"Processed interaction for user {user_id}: {result}")
            return result
        except Exception as e:
            logger.error(f"Error processing interaction for user {user_id}: {str(e)}", exc_info=True)
            return {
                'root_cause': 'error',
                'confidence': 0.0,
                'error': str(e)
            }

    def _validate_interaction(self, interaction: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize interaction data."""
        required_fields = ['type']
        for field in required_fields:
            if field not in interaction:
                logger.error(f"Missing required field: {field}")
                raise ValueError(f"Missing required field: {field}")
        
        valid_types = ['gesture', 'symbol', 'text', 'sound', 'eye_tracking', 'unknown']
        if interaction['type'] not in valid_types:
            logger.warning(f"Invalid interaction type: {interaction['type']}")
            interaction['type'] = 'unknown'
        
        interaction.setdefault('intent', 'unknown')
        interaction.setdefault('emotion', 'neutral')
        interaction.setdefault('confidence', 0.5)
        interaction.setdefault('input', '')
        
        return interaction

    def _process_gesture(self, interaction: Dict[str, Any]) -> Dict[str, Any]:
        """Process gesture input using pre-trained model."""
        try:
            features = interaction.get('input', [])
            if not features or not isinstance(features, list):
                logger.warning("Invalid gesture features")
                return interaction
            
            model_path = os.path.join(self.model_dir, 'gesture_model.pkl')
            if not os.path.exists(model_path):
                logger.error("Gesture model not found")
                return interaction
            
            with open(model_path, 'rb') as f:
                gesture_model = pickle.load(f)
            
            X = np.array([features])
            prediction = gesture_model.predict(X)[0]
            confidence = float(np.max(gesture_model.predict_proba(X)))
            
            gesture_info = self.gesture_map.get(prediction, {
                'intent': 'unknown',
                'message': f'Gesture {prediction} not recognized.',
                'emotion': 'neutral'
            })
            
            interaction['intent'] = gesture_info['intent']
            interaction['message'] = gesture_info['message']
            interaction['emotion'] = gesture_info['emotion']
            interaction['confidence'] = confidence
            
            logger.info(f"Processed gesture: {prediction} -> {gesture_info['intent']} (confidence: {confidence:.2f})")
            return interaction
        except Exception as e:
            logger.error(f"Error processing gesture: {str(e)}")
            return interaction

    def _process_symbol(self, interaction: Dict[str, Any]) -> Dict[str, Any]:
        """Process symbol input."""
        try:
            symbol = interaction.get('input', '')
            symbol_info = self.symbol_map.get(symbol, {
                'intent': 'unknown',
                'message': f'Symbol {symbol} not recognized.',
                'emotion': 'neutral',
                'confidence': 0.9
            })
            
            interaction['intent'] = symbol_info['intent']
            interaction['message'] = symbol_info['message']
            interaction['emotion'] = symbol_info['emotion']
            interaction['confidence'] = symbol_info.get('confidence', 0.9)
            
            logger.info(f"Processed symbol: {symbol} -> {symbol_info['intent']}")
            return interaction
        except Exception as e:
            logger.error(f"Error processing symbol: {str(e)}")
            return interaction

    def _process_text(self, interaction: Dict[str, Any]) -> Dict[str, Any]:
        """Process text input with NLU."""
        try:
            text = interaction.get('input', '')
            doc = nlp(text.lower())
            intent = 'communicate'
            emotion = 'neutral'
            confidence = 0.9
            
            if any(token.text in ['help', 'need', 'please', 'urgent'] for token in doc):
                intent = 'request_help'
                emotion = 'urgent'
            elif any(token.text in ['happy', 'glad', 'thank', 'good'] for token in doc):
                intent = 'express_joy'
                emotion = 'positive'
            elif any(token.text in ['sad', 'upset', 'sorry', 'bad'] for token in doc):
                intent = 'express_sadness'
                emotion = 'negative'
            elif any(token.text in ['question', 'ask', 'why', 'how', 'what', 'when', 'where'] for token in doc):
                intent = 'ask_question'
                emotion = 'inquisitive'
            
            interaction['intent'] = intent
            interaction['message'] = text if intent == 'communicate' else f"I want to say: {text}"
            interaction['emotion'] = emotion
            interaction['confidence'] = confidence
            
            logger.info(f"Processed text: {text} -> {intent} (confidence: {confidence:.2f})")
            return interaction
        except Exception as e:
            logger.error(f"Error processing text: {str(e)}")
            return interaction

    def _process_sound(self, interaction: Dict[str, Any]) -> Dict[str, Any]:
        """Process sound input (placeholder for sound recognition)."""
        try:
            sound_pattern = interaction.get('input', '')
            # Placeholder: Map sound patterns to intents
            sound_map = {
                'grunt': {'intent': 'express_discomfort', 'message': "I'm uncomfortable.", 'emotion': 'negative'},
                'whine': {'intent': 'request_attention', 'message': 'I need attention.', 'emotion': 'urgent'}
            }
            sound_info = sound_map.get(sound_pattern, {
                'intent': 'unknown',
                'message': f'Sound {sound_pattern} not recognized.',
                'emotion': 'neutral',
                'confidence': 0.7
            })
            
            interaction['intent'] = sound_info['intent']
            interaction['message'] = sound_info['message']
            interaction['emotion'] = sound_info['emotion']
            interaction['confidence'] = sound_info.get('confidence', 0.7)
            
            logger.info(f"Processed sound: {sound_pattern} -> {sound_info['intent']}")
            return interaction
        except Exception as e:
            logger.error(f"Error processing sound: {str(e)}")
            return interaction

    def _add_context(self, interaction: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """Add contextual information to the interaction."""
        context = {}
        now = datetime.now()
        context['time_of_day'] = now.strftime('%H:%M')
        context['day_of_week'] = now.strftime('%A')
        
        with self.lock:
            if user_id in self.context_window:
                prev_context = self.context_window[user_id].get('context', {})
                if 'location' in prev_context:
                    context['location'] = prev_context['location']
                if 'activity' in prev_context:
                    context['previous_activity'] = prev_context['activity']
                interactions = self.context_window[user_id].get('interactions', [])
                recent_count = sum(1 for ts in interactions if (now - ts).total_seconds() < 3600)
                context['interaction_frequency'] = recent_count
                context['recent_root_causes'] = [
                    rc['root_cause'] for rc in self.context_window[user_id].get('root_causes', [])[-5:]
                ]
        
        context.update(self._get_research_context())
        return context

    def _get_research_context(self) -> Dict[str, Any]:
        """Add research-informed context."""
        try:
            research_updates = self.research_module.cache.get('updates_applied', {})
            therapy_recommendations = research_updates.get('therapy_recommendations', [])
            if therapy_recommendations:
                return {
                    'recommended_therapy': therapy_recommendations[0]['therapy'],
                    'therapy_description': therapy_recommendations[0]['description']
                }
            return {}
        except Exception as e:
            logger.error(f"Error getting research context: {str(e)}")
            return {}

    def _update_context(self, user_id: str, interaction: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Update the context window with new interaction data."""
        with self.lock:
            if user_id not in self.context_window:
                self.context_window[user_id] = {
                    'interactions': deque(maxlen=100),
                    'context': {},
                    'root_causes': deque(maxlen=50)
                }
            
            self.context_window[user_id]['interactions'].append(datetime.now())
            self.context_window[user_id]['context'] = interaction.get('context', {})
            self.context_window[user_id]['root_causes'].append({
                'root_cause': result.get('root_cause', 'unknown'),
                'confidence': result.get('confidence', 0.0),
                'timestamp': datetime.now()
            })

    def get_memory(self) -> List[Dict]:
        """Retrieve recent interactions from memory."""
        return list(self.memory)

# Singleton instance
_input_processor = None
_lock = threading.Lock()

def get_input_processor():
    """Get or create the AlphaVoxInputProcessor singleton."""
    global _input_processor
    with _lock:
        if _input_processor is None:
            _input_processor = AlphaVoxInputProcessor()
        return _input_processor