
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from collections import deque
from datetime import datetime
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import spacy
from scipy.stats import entropy

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Data directory for models and memory
DATA_DIR = 'data'
MODEL_DIR = 'models'
MEMORY_FILE = os.path.join(DATA_DIR, 'nlc_memory.pkl')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Load spaCy for contextual analysis
try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    logger.error(f"Failed to load spaCy model: {str(e)}")
    logger.info("Creating a basic spaCy model as fallback")
    # Create a basic fallback model that just tokenizes text
    nlp = spacy.blank("en")

class NeuralLearningCore:
    """Neural Learning Core for AlphaVox to learn root causes of user behaviors."""
    
    def __init__(self, max_memory: int = 1000, learning_rate: float = 0.01):
        """Initialize the NLC with memory and learning components."""
        self.max_memory = max_memory
        self.learning_rate = learning_rate
        self.memory = deque(maxlen=max_memory)  # Store interactions
        self.root_cause_model = None
        self.scaler = StandardScaler()
        self.emotion_map = {
            'positive': 1.0, 'neutral': 0.0, 'negative': -1.0,
            'urgent': 0.5, 'inquisitive': 0.2, 'confused': -0.5
        }
        self.intent_weights = {}  # Track intent importance
        self.load_memory()
        self.initialize_model()
        logger.info("Neural Learning Core initialized")

    def initialize_model(self):
        """Initialize or load the root cause model."""
        model_path = os.path.join(MODEL_DIR, 'root_cause_model.pkl')
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    self.root_cause_model = pickle.load(f)
                logger.info(f"Loaded root cause model from {model_path}")
            except Exception as e:
                logger.error(f"Error loading root cause model: {str(e)}")
                self.root_cause_model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            self.root_cause_model = RandomForestClassifier(n_estimators=100, random_state=42)
        logger.info("Root cause model initialized")

    def load_memory(self):
        """Load interaction memory from disk."""
        if os.path.exists(MEMORY_FILE):
            try:
                with open(MEMORY_FILE, 'rb') as f:
                    saved_memory = pickle.load(f)
                    self.memory = deque(saved_memory, maxlen=self.max_memory)
                logger.info(f"Loaded {len(self.memory)} interactions from memory")
            except Exception as e:
                logger.error(f"Error loading memory: {str(e)}")
                self.memory = deque(maxlen=self.max_memory)
        else:
            self.memory = deque(maxlen=self.max_memory)

    def save_memory(self):
        """Save interaction memory to disk."""
        try:
            with open(MEMORY_FILE, 'wb') as f:
                pickle.dump(list(self.memory), f)
            logger.info(f"Saved {len(self.memory)} interactions to memory")
        except Exception as e:
            logger.error(f"Error saving memory: {str(e)}")

    def process_interaction(self, interaction: Dict, user_id: str) -> Dict:
        """Process a user interaction and infer root causes."""
        try:
            # Extract features from interaction
            features = self._extract_features(interaction)
            context = interaction.get('context', {})
            timestamp = datetime.now()

            # Infer root cause
            root_cause, confidence = self._infer_root_cause(features)

            # Update memory
            memory_entry = {
                'user_id': user_id,
                'interaction': interaction,
                'features': features,
                'root_cause': root_cause,
                'confidence': confidence,
                'timestamp': timestamp
            }
            self.memory.append(memory_entry)

            # Update model with feedback
            self._update_model(features, root_cause, interaction.get('feedback', None))

            # Update intent weights
            intent = interaction.get('intent', 'unknown')
            self.intent_weights[intent] = self.intent_weights.get(intent, 0.0) + self.learning_rate * confidence

            self.save_memory()
            logger.info(f"Processed interaction for user {user_id}: Root cause = {root_cause} (confidence: {confidence:.2f})")
            return {'root_cause': root_cause, 'confidence': confidence, 'features': features}
        except Exception as e:
            logger.error(f"Error processing interaction: {str(e)}")
            return {'root_cause': 'unknown', 'confidence': 0.0, 'features': []}

    def _extract_features(self, interaction: Dict) -> List[float]:
        """Extract features from an interaction for root cause analysis."""
        features = []
        
        # Input type (gesture, symbol, text, sound)
        input_type = interaction.get('type', 'unknown')
        input_type_encoding = {
            'gesture': 1.0, 'symbol': 2.0, 'text': 3.0, 'sound': 4.0, 'unknown': 0.0
        }
        features.append(input_type_encoding.get(input_type, 0.0))

        # Emotion score
        emotion = interaction.get('emotion', 'neutral')
        features.append(self.emotion_map.get(emotion, 0.0))

        # Confidence score
        confidence = interaction.get('confidence', 0.5)
        features.append(float(confidence))

        # Context features (time of day, interaction frequency)
        context = interaction.get('context', {})
        time_of_day = datetime.strptime(context.get('time_of_day', '12:00'), '%H:%M').hour / 24.0
        features.append(time_of_day)
        
        # Interaction frequency for user
        user_id = interaction.get('user_id', 'unknown')
        recent_interactions = sum(1 for m in self.memory if m['user_id'] == user_id and (datetime.now() - m['timestamp']).total_seconds() < 3600)
        features.append(recent_interactions / 10.0)  # Normalize

        # Text complexity (if text input)
        if input_type == 'text':
            doc = nlp(interaction.get('input', ''))
            complexity = len([token for token in doc if not token.is_stop and not token.is_punct]) / max(len(doc), 1)
            features.append(complexity)
        else:
            features.append(0.0)

        return features

    def _infer_root_cause(self, features: List[float]) -> Tuple[str, float]:
        """Infer the root cause of an interaction using the model."""
        try:
            # Scale features
            X = np.array([features])
            X_scaled = self.scaler.fit_transform(X) if not self.scaler.mean_ else self.scaler.transform(X)
            
            # Predict root cause
            root_causes = [
                'emotional_state', 'sensory_trigger', 'communication_intent',
                'social_context', 'cognitive_load', 'unknown'
            ]
            if self.root_cause_model:
                prediction = self.root_cause_model.predict(X_scaled)[0]
                probabilities = self.root_cause_model.predict_proba(X_scaled)[0]
                confidence = float(np.max(probabilities))
                root_cause = root_causes[prediction]
            else:
                root_cause = 'unknown'
                confidence = 0.5
            
            # Adjust confidence based on entropy
            if self.root_cause_model:
                entropy_val = entropy(probabilities)
                confidence *= (1 - entropy_val / np.log(len(root_causes)))
            
            return root_cause, confidence
        except Exception as e:
            logger.error(f"Error inferring root cause: {str(e)}")
            return 'unknown', 0.0

    def _update_model(self, features: List[float], root_cause: str, feedback: Optional[Dict]):
        """Update the root cause model with new data."""
        try:
            root_causes = [
                'emotional_state', 'sensory_trigger', 'communication_intent',
                'social_context', 'cognitive_load', 'unknown'
            ]
            if feedback and 'correct_root_cause' in feedback:
                true_label = feedback['correct_root_cause']
                if true_label in root_causes:
                    X = np.array([features])
                    y = np.array([root_causes.index(true_label)])
                    X_scaled = self.scaler.fit_transform(X) if not self.scaler.mean_ else self.scaler.transform(X)
                    self.root_cause_model.fit(X_scaled, y)
                    logger.info(f"Updated model with feedback: {true_label}")
            # Periodically retrain with memory
            if len(self.memory) > 100:
                self._retrain_model()
        except Exception as e:
            logger.error(f"Error updating model: {str(e)}")

    def _retrain_model(self):
        """Retrain the root cause model using memory data."""
        try:
            X = []
            y = []
            for entry in self.memory:
                if entry['confidence'] > 0.7:  # Use high-confidence interactions
                    X.append(entry['features'])
                    root_causes = [
                        'emotional_state', 'sensory_trigger', 'communication_intent',
                        'social_context', 'cognitive_load', 'unknown'
                    ]
                    y.append(root_causes.index(entry['root_cause']))
            
            if len(X) > 10:  # Minimum data threshold
                X = np.array(X)
                y = np.array(y)
                X_scaled = self.scaler.fit_transform(X)
                self.root_cause_model.fit(X_scaled, y)
                model_path = os.path.join(MODEL_DIR, 'root_cause_model.pkl')
                with open(model_path, 'wb') as f:
                    pickle.dump(self.root_cause_model, f)
                logger.info(f"Retrained and saved root cause model with {len(X)} samples")
        except Exception as e:
            logger.error(f"Error retraining model: {str(e)}")

    def get_user_insights(self, user_id: str) -> Dict:
        """Generate insights about a user's root causes."""
        try:
            user_interactions = [m for m in self.memory if m['user_id'] == user_id]
            if not user_interactions:
                return {'insights': [], 'summary': 'No interactions found'}
            
            # Analyze root cause distribution
            root_counts = {}
            for entry in user_interactions:
                rc = entry['root_cause']
                root_counts[rc] = root_counts.get(rc, 0) + 1
            
            # Generate insights
            insights = []
            total = len(user_interactions)
            for rc, count in root_counts.items():
                if count / total > 0.3:  # Significant pattern
                    insights.append({
                        'root_cause': rc,
                        'frequency': count / total,
                        'description': f"User frequently exhibits {rc} (e.g., {count} instances). Consider tailored responses."
                    })
            
            # Summary statistics
            summary = {
                'total_interactions': total,
                'dominant_root_cause': max(root_counts, key=root_counts.get, default='unknown'),
                'intent_diversity': len(set(m['interaction']['intent'] for m in user_interactions))
            }
            
            logger.info(f"Generated insights for user {user_id}: {len(insights)} insights")
            return {'insights': insights, 'summary': summary}
        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}")
            return {'insights': [], 'summary': f"Error: {str(e)}"}

# Singleton instance
_neural_learning_core = None

def get_neural_learning_core():
    """Get or create the NeuralLearningCore singleton."""
    global _neural_learning_core
    if _neural_learning_core is None:
        _neural_learning_core = NeuralLearningCore()
    return _neural_learning_core
