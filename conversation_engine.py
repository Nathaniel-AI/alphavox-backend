"""
Conversation Engine for AlphaVox

This module implements the conversational intelligence for AlphaVox, including
natural language understanding, response generation, and context management.

It integrates with the nonverbal engine to provide a unified communication system
that can interpret both verbal and nonverbal inputs.
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

# Try to import advanced NLP libraries, fall back to simpler methods if not available
try:
    import numpy as np
    ADVANCED_MODE = True
    logger.info("Using advanced NLP mode")
except ImportError:
    ADVANCED_MODE = False
    logger.info("Using basic NLP mode (numpy not available)")

# Try to import anthropic for advanced AI responses
try:
    import anthropic
    ANTHROPIC_CLIENT = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    HAS_ANTHROPIC = True
    logger.info("Anthropic API available for advanced conversational capabilities")
except (ImportError, Exception) as e:
    HAS_ANTHROPIC = False
    logger.warning(f"Anthropic API not available: {str(e)}")

class ConversationEngine:
    """
    Main conversation engine that processes text input, manages context,
    and generates appropriate responses.
    
    Features:
    - Intent recognition
    - Context management
    - Natural language generation
    - Emotional state tracking
    - Memory of past interactions
    """
    
    def __init__(self, nonverbal_engine=None):
        """
        Initialize the conversation engine
        
        Args:
            nonverbal_engine: NonverbalEngine instance for multimodal communication
        """
        self.nonverbal_engine = nonverbal_engine
        self.conversation_history = []
        self.max_history_length = 20
        self.emotional_state = {
            'valence': 0.0,  # -1.0 to 1.0, negative to positive
            'arousal': 0.0,  # 0.0 to 1.0, calm to excited
            'dominance': 0.5  # 0.0 to 1.0, submissive to dominant
        }
        
        # Load language resources
        self.intents = self._load_intents()
        self.responses = self._load_responses()
        self.language_map = self._load_language_map()
        
        # Advanced conversation state
        self.current_topic = None
        self.pending_questions = []
        
        # Adaptation metrics
        self.adaptation_stats = {
            'intent_recognition': {
                'successes': 0,
                'failures': 0
            },
            'response_generation': {
                'successes': 0,
                'failures': 0
            }
        }
        
        logger.info("Conversation engine initialized")
    
    def _load_intents(self) -> Dict[str, Dict[str, Any]]:
        """Load intent definitions from file or use defaults"""
        try:
            with open('data/intents.json', 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # Default intents
            return {
                'greeting': {
                    'patterns': ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening'],
                    'responses': ['Hello!', 'Hi there!', 'Greetings!'],
                    'context_required': False
                },
                'farewell': {
                    'patterns': ['goodbye', 'bye', 'see you', 'later', 'good night'],
                    'responses': ['Goodbye!', 'See you later!', 'Until next time!'],
                    'context_required': False
                },
                'help': {
                    'patterns': ['help', 'assist', 'support', 'how do I', 'what can you do'],
                    'responses': ['I can help you communicate. Try using gestures or symbols!',
                                'I\'m here to assist with communication needs.',
                                'I can interpret gestures, eye movements, and speech to help you express yourself.'],
                    'context_required': False
                },
                'request_info': {
                    'patterns': ['what is', 'how does', 'can you explain', 'tell me about'],
                    'responses': ['I\'ll try to explain that for you.',
                                'Let me find information about that.',
                                'Here\'s what I know about that topic:'],
                    'context_required': True
                },
                'express_needs': {
                    'patterns': ['I need', 'I want', 'I would like', 'can I have'],
                    'responses': ['I understand you need something.',
                                'Let me help you with that request.',
                                'I\'ll assist you with that need.'],
                    'context_required': True
                }
            }
    
    def _load_responses(self) -> Dict[str, List[str]]:
        """Load response templates from file or use defaults"""
        try:
            with open('data/responses.json', 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # Default responses by category
            return {
                'fallback': [
                    "I'm not sure I understood that. Could you try again?",
                    "I'm still learning. Could you phrase that differently?",
                    "I didn't quite catch that. Could you explain it another way?"
                ],
                'clarification': [
                    "Could you provide more details about that?",
                    "I'd like to understand better. Can you tell me more?",
                    "Could you elaborate on that point?"
                ],
                'acknowledgment': [
                    "I understand.",
                    "Got it.",
                    "I see what you mean."
                ],
                'positive': [
                    "That's great!",
                    "Wonderful!",
                    "Excellent!"
                ],
                'negative': [
                    "I'm sorry to hear that.",
                    "That's unfortunate.",
                    "I understand this is difficult."
                ],
                'encouragement': [
                    "You're doing great!",
                    "Keep going, you're making progress!",
                    "That's the right approach!"
                ]
            }
    
    def _load_language_map(self) -> Dict[str, Dict[str, Any]]:
        """Load language mapping for multilingual support"""
        try:
            with open('language_map.json', 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # Default minimal language map
            return {
                'en': {
                    'name': 'English',
                    'greetings': ['Hello', 'Hi', 'Welcome'],
                    'farewells': ['Goodbye', 'Bye', 'See you later'],
                    'yes': ['Yes', 'Yeah', 'Correct'],
                    'no': ['No', 'Nope', 'Incorrect']
                }
            }
    
    def process_text(self, text: str, user_id: Optional[str] = None, 
                    context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process text input and generate a response
        
        Args:
            text: Input text from the user
            user_id: Optional user identifier for personalization
            context: Optional context information (location, time, etc.)
            
        Returns:
            dict: Response with intent, confidence, message, etc.
        """
        logger.info(f"Processing text: {text}")
        
        # Clean and normalize input
        cleaned_text = text.strip().lower()
        
        # Add to conversation history
        self.conversation_history.append({
            'role': 'user',
            'text': cleaned_text,
            'timestamp': datetime.now().isoformat()
        })
        self._trim_history()
        
        # Try to use Anthropic for advanced conversations
        if HAS_ANTHROPIC and len(cleaned_text) > 10:
            try:
                return self._generate_advanced_response(cleaned_text, context)
            except Exception as e:
                logger.error(f"Error using Anthropic: {str(e)}")
                # Fall back to basic response
        
        # Identify intent
        intent, confidence, entities = self._identify_intent(cleaned_text)
        
        # Generate appropriate response
        response_text, emotion, emotion_tier = self._generate_response(
            intent, cleaned_text, confidence, entities, context
        )
        
        # Calculate emotional impact
        self._update_emotional_state(intent, confidence)
        
        # Record the response in conversation history
        self.conversation_history.append({
            'role': 'assistant',
            'text': response_text,
            'intent': intent,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat()
        })
        
        # Return formatted response
        return {
            'status': 'success',
            'message': response_text,
            'intent': intent,
            'confidence': confidence,
            'expression': emotion,
            'emotion_tier': emotion_tier
        }
    
    def _generate_advanced_response(self, text: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a response using Anthropic's Claude API
        
        Args:
            text: User input text
            context: Context information
            
        Returns:
            dict: Response with message, intent, etc.
        """
        if not HAS_ANTHROPIC:
            raise Exception("Anthropic API not available")
        
        # Build conversation history for context
        messages = []
        for entry in self.conversation_history[-5:]:  # Last 5 interactions
            if entry['role'] == 'user':
                messages.append({"role": "user", "content": entry['text']})
            else:
                messages.append({"role": "assistant", "content": entry['text']})
        
        # Add current message
        messages.append({"role": "user", "content": text})
        
        # Create system prompt with persona information
        system_prompt = (
            "You are AlphaVox, an AI assistant designed to help with communication. "
            "You are helpful, compassionate, and focused on understanding the user's needs. "
            "Keep your responses clear, concise, and conversational."
        )
        
        # Add context information if available
        if context:
            context_str = ". ".join([f"{k}: {v}" for k, v in context.items()])
            system_prompt += f" Context: {context_str}"
        
        # Send request to Anthropic
        response = ANTHROPIC_CLIENT.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1024,
            system=system_prompt,
            messages=messages
        )
        
        # Extract response
        message = response.content[0].text
        
        # Analyze emotion in response
        emotion_words = {
            'positive': ['happy', 'great', 'good', 'excellent', 'wonderful'],
            'negative': ['sorry', 'unfortunate', 'sad', 'difficult', 'problem'],
            'urgent': ['important', 'critical', 'urgent', 'immediately', 'crucial'],
            'inquisitive': ['curious', 'wonder', 'interesting', 'question', 'perhaps']
        }
        
        # Determine emotion
        emotion_counts = {emotion: 0 for emotion in emotion_words}
        for emotion, words in emotion_words.items():
            for word in words:
                emotion_counts[emotion] += message.lower().count(word)
        
        dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0]
        if all(count == 0 for count in emotion_counts.values()):
            dominant_emotion = 'neutral'
        
        # Determine emotion tier based on sentence structure and language
        if '!' in message or 'URGENT' in message.upper():
            emotion_tier = 'strong'
        elif any(word in message.lower() for word in ['very', 'really', 'extremely']):
            emotion_tier = 'moderate'
        else:
            emotion_tier = 'mild'
        
        return {
            'status': 'success',
            'message': message,
            'intent': 'respond',  # Generic intent for AI responses
            'confidence': 0.95,  # High confidence for AI-generated responses
            'expression': dominant_emotion,
            'emotion_tier': emotion_tier
        }
    
    def _identify_intent(self, text: str) -> Tuple[str, float, Dict[str, Any]]:
        """
        Identify the intent of the input text
        
        Args:
            text: Input text
            
        Returns:
            tuple: (intent, confidence, entities)
        """
        best_intent = 'unknown'
        best_confidence = 0.0
        entities = {}
        
        # Simple pattern matching for intents
        for intent_name, intent_data in self.intents.items():
            for pattern in intent_data['patterns']:
                if pattern in text:
                    confidence = 0.7 + (len(pattern) / len(text)) * 0.3
                    if confidence > best_confidence:
                        best_intent = intent_name
                        best_confidence = confidence
        
        # Extract entities (simple implementation)
        # TODO: Implement more sophisticated entity extraction
        common_entities = {
            'location': ['home', 'school', 'hospital', 'outside', 'inside'],
            'time': ['morning', 'afternoon', 'evening', 'night', 'now', 'later'],
            'person': ['doctor', 'nurse', 'teacher', 'mom', 'dad', 'caregiver']
        }
        
        for entity_type, entity_values in common_entities.items():
            for value in entity_values:
                if value in text:
                    entities[entity_type] = value
        
        # Add randomness to simulate real-world uncertainty
        confidence_variation = random.uniform(-0.1, 0.1)
        best_confidence = min(0.99, max(0.2, best_confidence + confidence_variation))
        
        return best_intent, best_confidence, entities
    
    def _generate_response(self, intent: str, text: str, confidence: float, 
                        entities: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Tuple[str, str, str]:
        """
        Generate a response based on intent and context
        
        Args:
            intent: Identified intent
            text: Original input text
            confidence: Confidence score
            entities: Extracted entities
            context: Context information
            
        Returns:
            tuple: (response_text, emotion, emotion_tier)
        """
        # Get intent-specific responses if available
        if intent in self.intents and 'responses' in self.intents[intent]:
            response_options = self.intents[intent]['responses']
            response = random.choice(response_options)
        elif confidence < 0.4:
            # Low confidence, use clarification response
            response = random.choice(self.responses['clarification'])
        else:
            # Fallback response
            response = random.choice(self.responses['fallback'])
        
        # Determine emotional expression based on intent
        if intent in ['greeting', 'help']:
            emotion = 'positive'
            emotion_tier = 'mild'
        elif intent in ['farewell']:
            emotion = 'neutral'
            emotion_tier = 'mild'
        elif intent in ['express_needs']:
            emotion = 'positive' if confidence > 0.7 else 'inquisitive'
            emotion_tier = 'moderate'
        else:
            emotion = 'neutral'
            emotion_tier = 'mild'
        
        # Add entities to response if available
        if entities and confidence > 0.6:
            entity_phrases = []
            for entity_type, entity_value in entities.items():
                if entity_type == 'location':
                    entity_phrases.append(f"at {entity_value}")
                elif entity_type == 'time':
                    entity_phrases.append(f"during the {entity_value}")
                elif entity_type == 'person':
                    entity_phrases.append(f"with the {entity_value}")
            
            if entity_phrases:
                entity_text = " " + " ".join(entity_phrases)
                response = response.rstrip('.') + entity_text + "."
        
        return response, emotion, emotion_tier
    
    def _update_emotional_state(self, intent: str, confidence: float):
        """
        Update the emotional state based on the interaction
        
        Args:
            intent: The identified intent
            confidence: Confidence score
        """
        # Map intents to emotional impact
        intent_valence = {
            'greeting': 0.2,
            'farewell': 0.1,
            'help': 0.2,
            'request_info': 0.1,
            'express_needs': 0.0,
            'unknown': -0.1
        }
        
        # Update emotional state components
        valence_impact = intent_valence.get(intent, 0.0) * confidence
        self.emotional_state['valence'] = max(-1.0, min(1.0, 
                                            self.emotional_state['valence'] + valence_impact))
        
        # Arousal increases with interaction, decays over time
        self.emotional_state['arousal'] = max(0.0, min(1.0,
                                            self.emotional_state['arousal'] + 0.1 * confidence))
        
        # Dominance depends on the type of interaction
        if intent in ['express_needs', 'request_info']:
            # User is directing the conversation
            dominance_impact = -0.05 * confidence
        else:
            # Neutral impact
            dominance_impact = 0.0
            
        self.emotional_state['dominance'] = max(0.0, min(1.0,
                                            self.emotional_state['dominance'] + dominance_impact))
    
    def _trim_history(self):
        """Trim conversation history to the maximum length"""
        if len(self.conversation_history) > self.max_history_length:
            # Keep the most recent conversations
            self.conversation_history = self.conversation_history[-self.max_history_length:]
    
    def get_emotional_state(self) -> Dict[str, float]:
        """
        Get the current emotional state
        
        Returns:
            dict: Emotional state components
        """
        return self.emotional_state
    
    def save_models(self):
        """Save learned models and conversation patterns"""
        # Placeholder for actual model saving
        logger.info("Saving conversation models")
    
    def register_feedback(self, response_id: str, success: bool, feedback: Optional[str] = None):
        """
        Register feedback about a response
        
        Args:
            response_id: Identifier for the response
            success: Whether the response was successful
            feedback: Optional feedback text
        """
        # Update adaptation stats
        if success:
            self.adaptation_stats['response_generation']['successes'] += 1
        else:
            self.adaptation_stats['response_generation']['failures'] += 1
            
        logger.info(f"Registered feedback for response {response_id}: {'success' if success else 'failure'}")
        
        # TODO: Implement learning from feedback

# Singleton instance
_conversation_engine = None

def get_conversation_engine(nonverbal_engine=None):
    """Get or create the conversation engine singleton"""
    global _conversation_engine
    if _conversation_engine is None:
        _conversation_engine = ConversationEngine(nonverbal_engine)
    return _conversation_engine