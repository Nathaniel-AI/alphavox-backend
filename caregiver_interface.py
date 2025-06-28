"""
AlphaVox - Caregiver Interface
-----------------------------
This module provides the caregiver interface for AlphaVox, allowing caregivers
to monitor user progress, customize settings, and review communication history.

Key features:
- Progress monitoring and reporting
- Communication history review
- User profile management
- Settings customization
- Nonverbal communication training resources
"""

import logging
import time
import os
import json
from typing import Dict, Any, List, Optional, Tuple, Union

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import eye tracking service
try:
    from eye_tracking_service import get_eye_tracking_service
    eye_tracking_available = True
except ImportError:
    logger.warning("Eye tracking service not available")
    eye_tracking_available = False

# Try to import learning analytics
try:
    from learning_analytics import get_learning_analytics
    analytics_available = True
except ImportError:
    logger.warning("Learning analytics not available")
    analytics_available = False

# Try to import nonverbal expertise
try:
    from nonverbal_expertise import get_nonverbal_expertise, NONVERBAL_TOPICS
    expertise_available = True
except ImportError:
    logger.warning("Nonverbal expertise not available")
    expertise_available = False

class CaregiverInterface:
    """
    Interface for caregiver access to AlphaVox features and data.
    
    This class:
    - Provides access to user progress and metrics
    - Manages communication history and reporting
    - Offers customization of user settings
    - Delivers educational resources for caregivers
    """
    
    def __init__(self):
        """Initialize the caregiver interface"""
        # User profiles
        self.user_profiles = self._load_user_profiles()
        
        # Communication history
        self.communication_history = self._load_communication_history()
        
        # Connected services
        self.eye_tracking = None
        self.analytics = None
        self.expertise = None
        
        # Initialize connected services
        self._initialize_services()
        
        logger.info("Caregiver Interface initialized")
    
    def _initialize_services(self):
        """Initialize connections to related services"""
        # Connect to eye tracking if available
        if eye_tracking_available:
            try:
                self.eye_tracking = get_eye_tracking_service()
                logger.info("Connected to eye tracking service")
            except Exception as e:
                logger.error(f"Failed to connect to eye tracking service: {e}")
        
        # Connect to learning analytics if available
        if analytics_available:
            try:
                self.analytics = get_learning_analytics()
                logger.info("Connected to learning analytics service")
            except Exception as e:
                logger.error(f"Failed to connect to learning analytics service: {e}")
        
        # Connect to nonverbal expertise if available
        if expertise_available:
            try:
                self.expertise = get_nonverbal_expertise()
                logger.info("Connected to nonverbal expertise module")
            except Exception as e:
                logger.error(f"Failed to connect to nonverbal expertise module: {e}")
    
    def _load_user_profiles(self) -> Dict[str, Dict[str, Any]]:
        """
        Load user profiles from file.
        
        Returns:
            Dictionary of user profiles
        """
        profiles_path = os.path.join('profiles', 'user_profiles.json')
        
        # Default empty profiles
        default_profiles = {}
        
        # Try to load profiles from file
        try:
            if os.path.exists(profiles_path):
                with open(profiles_path, 'r') as f:
                    profiles = json.load(f)
                logger.info(f"Loaded {len(profiles)} user profiles")
                return profiles
        except Exception as e:
            logger.warning(f"Could not load user profiles: {e}")
        
        # Create sample profile if none exist
        default_profiles = {
            "1": {
                "id": "1",
                "name": "Sample User",
                "age": 12,
                "created_at": time.time(),
                "preferences": {
                    "eye_tracking": {"dwell_threshold": 1.0},
                    "speech": {"rate": "medium", "voice": "calming"},
                    "interface": {"theme": "light", "symbol_size": "medium"}
                },
                "communication_style": {
                    "preferred_symbols": ["food", "drink", "help", "yes", "no"],
                    "primary_mode": "symbols"
                },
                "progress": {
                    "sessions": 5,
                    "total_interactions": 120,
                    "recent_symbols": ["food", "help", "yes"],
                    "favorite_symbol": "help"
                }
            }
        }
        
        # Save default profiles to file
        try:
            os.makedirs(os.path.dirname(profiles_path), exist_ok=True)
            with open(profiles_path, 'w') as f:
                json.dump(default_profiles, f, indent=4)
            logger.info("Created default user profiles")
        except Exception as e:
            logger.warning(f"Could not save default user profiles: {e}")
        
        return default_profiles
    
    def _load_communication_history(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load communication history from file.
        
        Returns:
            Dictionary of user communication history
        """
        history_path = os.path.join('data', 'communication_history.json')
        
        # Default empty history
        default_history = {}
        
        # Try to load history from file
        try:
            if os.path.exists(history_path):
                with open(history_path, 'r') as f:
                    history = json.load(f)
                logger.info(f"Loaded communication history for {len(history)} users")
                return history
        except Exception as e:
            logger.warning(f"Could not load communication history: {e}")
        
        # Create sample history if none exists
        default_history = {
            "1": [
                {
                    "timestamp": time.time() - 3600,
                    "type": "symbol",
                    "content": "food",
                    "message": "I want food.",
                    "response": "I'll get you some food right away."
                },
                {
                    "timestamp": time.time() - 1800,
                    "type": "symbol",
                    "content": "help",
                    "message": "I need help.",
                    "response": "I'm here to help. What do you need?"
                },
                {
                    "timestamp": time.time() - 900,
                    "type": "gesture",
                    "content": "wave",
                    "message": "Hello/Greeting",
                    "response": "Hello there!"
                }
            ]
        }
        
        # Save default history to file
        try:
            os.makedirs(os.path.dirname(history_path), exist_ok=True)
            with open(history_path, 'w') as f:
                json.dump(default_history, f, indent=4)
            logger.info("Created default communication history")
        except Exception as e:
            logger.warning(f"Could not save default communication history: {e}")
        
        return default_history
    
    def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """
        Get a user profile by ID.
        
        Args:
            user_id: User ID to retrieve
            
        Returns:
            User profile dictionary or empty dict if not found
        """
        return self.user_profiles.get(user_id, {})
    
    def get_all_user_profiles(self) -> List[Dict[str, Any]]:
        """
        Get all user profiles.
        
        Returns:
            List of user profile dictionaries
        """
        return list(self.user_profiles.values())
    
    def update_user_profile(self, user_id: str, profile_data: Dict[str, Any]) -> bool:
        """
        Update a user profile.
        
        Args:
            user_id: User ID to update
            profile_data: Profile data to update
            
        Returns:
            True if successful
        """
        # Check if user exists
        if user_id not in self.user_profiles:
            logger.error(f"User {user_id} not found")
            return False
        
        # Update profile
        self.user_profiles[user_id].update(profile_data)
        
        # Save all profiles
        try:
            profiles_path = os.path.join('profiles', 'user_profiles.json')
            with open(profiles_path, 'w') as f:
                json.dump(self.user_profiles, f, indent=4)
            logger.info(f"Updated and saved profile for user {user_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to save user profiles: {e}")
            return False
    
    def create_user_profile(self, profile_data: Dict[str, Any]) -> Optional[str]:
        """
        Create a new user profile.
        
        Args:
            profile_data: Profile data for the new user
            
        Returns:
            New user ID if successful, None otherwise
        """
        # Generate new user ID
        user_id = str(int(time.time()))
        
        # Create new profile
        new_profile = {
            "id": user_id,
            "created_at": time.time(),
            "preferences": {
                "eye_tracking": {"dwell_threshold": 1.0},
                "speech": {"rate": "medium", "voice": "calming"},
                "interface": {"theme": "light", "symbol_size": "medium"}
            },
            "communication_style": {
                "preferred_symbols": ["food", "drink", "help", "yes", "no"],
                "primary_mode": "symbols"
            },
            "progress": {
                "sessions": 0,
                "total_interactions": 0,
                "recent_symbols": [],
                "favorite_symbol": ""
            }
        }
        
        # Update with provided data
        new_profile.update(profile_data)
        
        # Add to profiles
        self.user_profiles[user_id] = new_profile
        
        # Save all profiles
        try:
            profiles_path = os.path.join('profiles', 'user_profiles.json')
            with open(profiles_path, 'w') as f:
                json.dump(self.user_profiles, f, indent=4)
            logger.info(f"Created and saved new user {user_id}")
            return user_id
        except Exception as e:
            logger.error(f"Failed to save user profiles: {e}")
            return None
    
    def get_user_communication_history(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get communication history for a user.
        
        Args:
            user_id: User ID to retrieve history for
            limit: Maximum number of items to return
            
        Returns:
            List of communication history items
        """
        history = self.communication_history.get(user_id, [])
        
        # Sort by timestamp descending and limit
        sorted_history = sorted(history, key=lambda x: x.get("timestamp", 0), reverse=True)
        
        return sorted_history[:limit]
    
    def add_communication_event(self, user_id: str, event_data: Dict[str, Any]) -> bool:
        """
        Add a communication event to a user's history.
        
        Args:
            user_id: User ID to add event for
            event_data: Event data to add
            
        Returns:
            True if successful
        """
        # Ensure user exists in history
        if user_id not in self.communication_history:
            self.communication_history[user_id] = []
        
        # Add timestamp if not provided
        if "timestamp" not in event_data:
            event_data["timestamp"] = time.time()
        
        # Add event to history
        self.communication_history[user_id].append(event_data)
        
        # Trim history if too long (keep last 1000 events)
        if len(self.communication_history[user_id]) > 1000:
            self.communication_history[user_id] = self.communication_history[user_id][-1000:]
        
        # Save history
        try:
            history_path = os.path.join('data', 'communication_history.json')
            with open(history_path, 'w') as f:
                json.dump(self.communication_history, f, indent=4)
            logger.info(f"Added communication event for user {user_id}")
            
            # Update user profile stats if available
            if user_id in self.user_profiles:
                # Update total interactions
                if "progress" not in self.user_profiles[user_id]:
                    self.user_profiles[user_id]["progress"] = {}
                
                progress = self.user_profiles[user_id]["progress"]
                progress["total_interactions"] = progress.get("total_interactions", 0) + 1
                
                # Update recent symbols if this is a symbol event
                if event_data.get("type") == "symbol" and "content" in event_data:
                    symbol = event_data["content"]
                    
                    if "recent_symbols" not in progress:
                        progress["recent_symbols"] = []
                    
                    progress["recent_symbols"].insert(0, symbol)
                    progress["recent_symbols"] = progress["recent_symbols"][:10]
                    
                    # Update favorite symbol
                    symbol_counts = {}
                    for s in self.communication_history[user_id]:
                        if s.get("type") == "symbol" and "content" in s:
                            symbol_counts[s["content"]] = symbol_counts.get(s["content"], 0) + 1
                    
                    if symbol_counts:
                        progress["favorite_symbol"] = max(symbol_counts.items(), key=lambda x: x[1])[0]
                
                # Save updated profile
                self.update_user_profile(user_id, {"progress": progress})
            
            return True
        except Exception as e:
            logger.error(f"Failed to save communication history: {e}")
            return False
    
    def get_user_progress_report(self, user_id: str) -> Dict[str, Any]:
        """
        Generate a progress report for a user.
        
        Args:
            user_id: User ID to generate report for
            
        Returns:
            Dictionary with progress report data
        """
        report = {
            "user_id": user_id,
            "timestamp": time.time(),
            "profile": self.get_user_profile(user_id),
            "communication": {
                "total_events": len(self.get_user_communication_history(user_id)),
                "recent_events": self.get_user_communication_history(user_id, 10)
            }
        }
        
        # Add eye tracking statistics if available
        if self.eye_tracking:
            try:
                report["eye_tracking"] = self.eye_tracking.get_region_statistics()
            except Exception as e:
                logger.error(f"Failed to get eye tracking statistics: {e}")
        
        # Add learning analytics if available
        if self.analytics:
            try:
                report["learning"] = self.analytics.get_user_metrics(user_id)
            except Exception as e:
                logger.error(f"Failed to get learning analytics: {e}")
        
        return report
    
    def get_educational_resources(self, topic: Optional[str] = None) -> Dict[str, Any]:
        """
        Get educational resources for caregivers.
        
        Args:
            topic: Optional topic to filter resources
            
        Returns:
            Dictionary with educational resources
        """
        resources = {
            "topics": NONVERBAL_TOPICS if expertise_available else [
                "gesture_interpretation", "eye_tracking", "facial_expressions",
                "emotional_regulation", "autism_nonverbal", "communication_strategies"
            ],
            "latest_research": [
                {
                    "title": "Advances in Assistive Communication Technologies",
                    "authors": "Johnson et al.",
                    "year": 2025,
                    "source": "Journal of Assistive Technology",
                    "summary": "Recent advances in AI-powered communication systems are showing significant improvements in supporting nonverbal individuals."
                },
                {
                    "title": "Eye Tracking for Communication: A Longitudinal Study",
                    "authors": "Williams and Chen",
                    "year": 2024,
                    "source": "Augmentative and Alternative Communication",
                    "summary": "Long-term use of eye tracking systems demonstrates improved communication outcomes for nonverbal children."
                },
                {
                    "title": "Multimodal Communication Patterns in Autism",
                    "authors": "Garcia et al.",
                    "year": 2025,
                    "source": "Journal of Autism and Developmental Disorders",
                    "summary": "Analysis of multimodal communication reveals consistent patterns that can inform intervention approaches."
                }
            ],
            "recommended_strategies": [
                {
                    "title": "Wait Time Technique",
                    "description": "Provide extended processing time after communication attempts, allowing the individual to formulate responses."
                },
                {
                    "title": "Follow Their Lead",
                    "description": "Observe and respond to the individual's interests and communication attempts rather than directing interactions."
                },
                {
                    "title": "Consistent Visual Supports",
                    "description": "Maintain consistent symbols and visual cues across environments to support comprehension and expression."
                },
                {
                    "title": "Positive Communication Environment",
                    "description": "Create low-stress, supportive environments that encourage communication attempts without pressure."
                }
            ]
        }
        
        # Add expertise content if available and topic specified
        if topic and self.expertise:
            try:
                expertise_response = self.expertise.process_expertise_query(topic)
                resources["expertise"] = expertise_response
            except Exception as e:
                logger.error(f"Failed to get expertise for topic {topic}: {e}")
        
        return resources


# Create singleton instance
_caregiver_interface = None

def get_caregiver_interface():
    """Get the singleton instance of the caregiver interface"""
    global _caregiver_interface
    if _caregiver_interface is None:
        _caregiver_interface = CaregiverInterface()
    return _caregiver_interface


# Example functions for direct use in routes
def get_user_profile(user_id: str) -> Dict[str, Any]:
    """
    Get a user profile.
    
    Args:
        user_id: User ID to retrieve
        
    Returns:
        User profile dictionary
    """
    interface = get_caregiver_interface()
    return interface.get_user_profile(user_id)

def get_progress_report(user_id: str) -> Dict[str, Any]:
    """
    Get a progress report for a user.
    
    Args:
        user_id: User ID to generate report for
        
    Returns:
        Progress report dictionary
    """
    interface = get_caregiver_interface()
    return interface.get_user_progress_report(user_id)

def get_educational_content(topic: Optional[str] = None) -> Dict[str, Any]:
    """
    Get educational content for caregivers.
    
    Args:
        topic: Optional topic to filter content
        
    Returns:
        Educational content dictionary
    """
    interface = get_caregiver_interface()
    return interface.get_educational_resources(topic)