"""
AlphaVox Learning Journey Module

This module implements enhanced learning capabilities for AlphaVox, tracking
user progress, providing adaptive learning paths, and offering insights 
on learning milestones and achievements.
"""

import json
import os
import logging
import datetime
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
LEARNING_DATA_PATH = "attached_assets/learning_log.json"
KNOWLEDGE_BASE_PATH = "attached_assets/knowledge_base.json"
TOPICS_PATH = "attached_assets/topics.json"
FACTS_PATH = "attached_assets/facts.json"
KNOWLEDGE_GRAPH_PATH = "attached_assets/knowledge_graph.json"

class LearningJourney:
    """
    Enhanced learning journey manager for AlphaVox that tracks user progress,
    recommends learning paths, and provides insights on achievements.
    """
    
    def __init__(self):
        """Initialize the learning journey manager."""
        self.learning_log = self._load_json(LEARNING_DATA_PATH, default=[])
        self.knowledge_base = self._load_json(KNOWLEDGE_BASE_PATH, default={})
        self.topics = self._load_json(TOPICS_PATH, default=[])
        self.facts = self._load_json(FACTS_PATH, default={})
        self.knowledge_graph = self._load_json(KNOWLEDGE_GRAPH_PATH, default={"nodes": [], "edges": []})
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(LEARNING_DATA_PATH), exist_ok=True)
    
    def _load_json(self, path: str, default: Any = None) -> Any:
        """Load data from a JSON file or return default if file doesn't exist."""
        try:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    return json.load(f)
            return default
        except Exception as e:
            logger.error(f"Error loading {path}: {e}")
            return default
    
    def _save_json(self, data: Any, path: str) -> bool:
        """Save data to a JSON file."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving {path}: {e}")
            return False
    
    def log_learning_event(self, 
                         event_type: str,
                         user_id: str,
                         details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Record a learning event in the user's learning journey.
        
        Args:
            event_type: Type of learning event (e.g., 'topic_mastered', 'question_answered')
            user_id: Unique identifier for the user
            details: Additional details about the learning event
            
        Returns:
            Dict containing the recorded event
        """
        event = {
            "id": len(self.learning_log) + 1,
            "timestamp": datetime.datetime.now().isoformat(),
            "event_type": event_type,
            "user_id": user_id,
            "details": details
        }
        
        self.learning_log.append(event)
        self._save_json(self.learning_log, LEARNING_DATA_PATH)
        
        # Update knowledge graph if applicable
        if event_type in ['topic_explored', 'fact_learned', 'concept_connected']:
            self._update_knowledge_graph(event)
        
        return event
    
    def get_learning_summary(self, user_id: str) -> Dict[str, Any]:
        """
        Get a summary of the user's learning journey.
        
        Args:
            user_id: Unique identifier for the user
            
        Returns:
            Dict containing a summary of the user's learning journey
        """
        # Filter events for the user
        user_events = [e for e in self.learning_log if e["user_id"] == user_id]
        
        # Count events by type
        event_counts = defaultdict(int)
        for event in user_events:
            event_counts[event["event_type"]] += 1
        
        # Get topics explored
        topics_explored = set()
        for event in user_events:
            if event["event_type"] == "topic_explored" and "topic" in event["details"]:
                topics_explored.add(event["details"]["topic"])
        
        # Get facts learned
        facts_learned = set()
        for event in user_events:
            if event["event_type"] == "fact_learned" and "fact_id" in event["details"]:
                facts_learned.add(event["details"]["fact_id"])
        
        # Calculate learning metrics
        learning_velocity = len(user_events) / max(1, (datetime.datetime.now() - 
                                                  datetime.datetime.fromisoformat(user_events[0]["timestamp"] 
                                                                              if user_events else datetime.datetime.now().isoformat())).days)
        
        # Generate summary
        return {
            "user_id": user_id,
            "total_events": len(user_events),
            "event_counts": dict(event_counts),
            "topics_explored": list(topics_explored),
            "facts_learned": list(facts_learned),
            "learning_velocity": learning_velocity,
            "first_activity": user_events[0]["timestamp"] if user_events else None,
            "last_activity": user_events[-1]["timestamp"] if user_events else None
        }
    
    def get_recommended_topics(self, user_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get recommended topics for the user based on their learning journey.
        
        Args:
            user_id: Unique identifier for the user
            limit: Maximum number of topics to recommend
            
        Returns:
            List of recommended topics
        """
        # Get topics the user has already explored
        user_events = [e for e in self.learning_log if e["user_id"] == user_id]
        explored_topics = set()
        for event in user_events:
            if event["event_type"] == "topic_explored" and "topic" in event["details"]:
                explored_topics.add(event["details"]["topic"])
        
        # Filter topics that haven't been explored yet
        unexplored_topics = [t for t in self.topics if t["name"] not in explored_topics]
        
        # Sort by relevance (for now, just return the first few unexplored topics)
        return unexplored_topics[:limit]
    
    def add_topic(self, name: str, description: str, difficulty: str, prerequisites: List[str] = []) -> Dict[str, Any]:
        """
        Add a new topic to the learning system.
        
        Args:
            name: Name of the topic
            description: Description of the topic
            difficulty: Difficulty level (e.g., 'beginner', 'intermediate', 'advanced')
            prerequisites: List of prerequisite topics
            
        Returns:
            The newly added topic
        """
        topic = {
            "id": len(self.topics) + 1,
            "name": name,
            "description": description,
            "difficulty": difficulty,
            "prerequisites": prerequisites
        }
        
        self.topics.append(topic)
        self._save_json(self.topics, TOPICS_PATH)
        
        # Update knowledge graph
        self._add_topic_to_knowledge_graph(topic)
        
        return topic
    
    def add_fact(self, topic: str, content: str, source: str = None) -> Dict[str, Any]:
        """
        Add a new fact to the knowledge base.
        
        Args:
            topic: Topic the fact relates to
            content: Content of the fact
            source: Source of the fact
            
        Returns:
            The newly added fact
        """
        fact_id = f"fact_{len(self.facts) + 1}"
        fact = {
            "id": fact_id,
            "topic": topic,
            "content": content,
            "source": source,
            "created_at": datetime.datetime.now().isoformat()
        }
        
        self.facts[fact_id] = fact
        self._save_json(self.facts, FACTS_PATH)
        
        # Update knowledge graph
        self._add_fact_to_knowledge_graph(fact)
        
        return fact
    
    def _update_knowledge_graph(self, event: Dict[str, Any]) -> None:
        """Update the knowledge graph based on a learning event."""
        if event["event_type"] == "topic_explored" and "topic" in event["details"]:
            # Check if node already exists
            topic = event["details"]["topic"]
            if not any(n["id"] == topic for n in self.knowledge_graph["nodes"]):
                self.knowledge_graph["nodes"].append({
                    "id": topic,
                    "type": "topic",
                    "label": topic
                })
        
        elif event["event_type"] == "fact_learned" and "fact_id" in event["details"]:
            fact_id = event["details"]["fact_id"]
            if fact_id in self.facts:
                fact = self.facts[fact_id]
                topic = fact["topic"]
                
                # Add nodes if they don't exist
                if not any(n["id"] == fact_id for n in self.knowledge_graph["nodes"]):
                    self.knowledge_graph["nodes"].append({
                        "id": fact_id,
                        "type": "fact",
                        "label": fact["content"][:30] + "..." if len(fact["content"]) > 30 else fact["content"]
                    })
                
                if not any(n["id"] == topic for n in self.knowledge_graph["nodes"]):
                    self.knowledge_graph["nodes"].append({
                        "id": topic,
                        "type": "topic",
                        "label": topic
                    })
                
                # Add edge between fact and topic
                edge_id = f"{fact_id}_{topic}"
                if not any(e["id"] == edge_id for e in self.knowledge_graph["edges"]):
                    self.knowledge_graph["edges"].append({
                        "id": edge_id,
                        "source": fact_id,
                        "target": topic,
                        "type": "belongs_to"
                    })
        
        elif event["event_type"] == "concept_connected" and "concepts" in event["details"]:
            concepts = event["details"]["concepts"]
            if len(concepts) >= 2:
                for i in range(len(concepts)):
                    for j in range(i+1, len(concepts)):
                        # Add edge between concepts
                        edge_id = f"{concepts[i]}_{concepts[j]}"
                        if not any(e["id"] == edge_id for e in self.knowledge_graph["edges"]):
                            self.knowledge_graph["edges"].append({
                                "id": edge_id,
                                "source": concepts[i],
                                "target": concepts[j],
                                "type": "connected"
                            })
        
        # Save knowledge graph
        self._save_json(self.knowledge_graph, KNOWLEDGE_GRAPH_PATH)
    
    def _add_topic_to_knowledge_graph(self, topic: Dict[str, Any]) -> None:
        """Add a topic to the knowledge graph."""
        # Add node for the topic
        if not any(n["id"] == topic["name"] for n in self.knowledge_graph["nodes"]):
            self.knowledge_graph["nodes"].append({
                "id": topic["name"],
                "type": "topic",
                "label": topic["name"]
            })
        
        # Add edges for prerequisites
        for prereq in topic["prerequisites"]:
            # Add node for prerequisite if it doesn't exist
            if not any(n["id"] == prereq for n in self.knowledge_graph["nodes"]):
                self.knowledge_graph["nodes"].append({
                    "id": prereq,
                    "type": "topic",
                    "label": prereq
                })
            
            # Add edge
            edge_id = f"{prereq}_{topic['name']}"
            if not any(e["id"] == edge_id for e in self.knowledge_graph["edges"]):
                self.knowledge_graph["edges"].append({
                    "id": edge_id,
                    "source": prereq,
                    "target": topic["name"],
                    "type": "prerequisite"
                })
        
        # Save knowledge graph
        self._save_json(self.knowledge_graph, KNOWLEDGE_GRAPH_PATH)
    
    def _add_fact_to_knowledge_graph(self, fact: Dict[str, Any]) -> None:
        """Add a fact to the knowledge graph."""
        # Add node for the fact
        if not any(n["id"] == fact["id"] for n in self.knowledge_graph["nodes"]):
            self.knowledge_graph["nodes"].append({
                "id": fact["id"],
                "type": "fact",
                "label": fact["content"][:30] + "..." if len(fact["content"]) > 30 else fact["content"]
            })
        
        # Add node for the topic if it doesn't exist
        topic = fact["topic"]
        if not any(n["id"] == topic for n in self.knowledge_graph["nodes"]):
            self.knowledge_graph["nodes"].append({
                "id": topic,
                "type": "topic",
                "label": topic
            })
        
        # Add edge between fact and topic
        edge_id = f"{fact['id']}_{topic}"
        if not any(e["id"] == edge_id for e in self.knowledge_graph["edges"]):
            self.knowledge_graph["edges"].append({
                "id": edge_id,
                "source": fact["id"],
                "target": topic,
                "type": "belongs_to"
            })
        
        # Save knowledge graph
        self._save_json(self.knowledge_graph, KNOWLEDGE_GRAPH_PATH)
    
    def get_knowledge_graph(self) -> Dict[str, Any]:
        """Get the knowledge graph."""
        return self.knowledge_graph
    
    def get_learning_path(self, user_id: str, goal_topic: str) -> List[Dict[str, Any]]:
        """
        Generate a personalized learning path to reach a goal topic.
        
        Args:
            user_id: Unique identifier for the user
            goal_topic: The topic the user wants to learn
            
        Returns:
            List of steps in the learning path
        """
        # Get topics the user has already explored
        user_events = [e for e in self.learning_log if e["user_id"] == user_id]
        explored_topics = set()
        for event in user_events:
            if event["event_type"] == "topic_explored" and "topic" in event["details"]:
                explored_topics.add(event["details"]["topic"])
        
        # Find the goal topic
        goal = next((t for t in self.topics if t["name"] == goal_topic), None)
        if not goal:
            return []
        
        # Build a path of prerequisites not yet explored
        path = []
        queue = [goal]
        visited = set()
        
        while queue:
            current = queue.pop(0)
            if current["name"] in visited:
                continue
                
            visited.add(current["name"])
            
            # Add to path if not already explored
            if current["name"] not in explored_topics:
                path.append(current)
            
            # Add prerequisites to queue
            for prereq_name in current["prerequisites"]:
                prereq = next((t for t in self.topics if t["name"] == prereq_name), None)
                if prereq and prereq_name not in visited:
                    queue.append(prereq)
        
        # Reverse path to get prerequisites first
        path.reverse()
        
        return path
    
    def get_learning_statistics(self, user_id: str) -> Dict[str, Any]:
        """
        Get detailed learning statistics for the user.
        
        Args:
            user_id: Unique identifier for the user
            
        Returns:
            Dict containing detailed learning statistics
        """
        # Filter events for the user
        user_events = [e for e in self.learning_log if e["user_id"] == user_id]
        
        # Group events by day
        events_by_day = defaultdict(list)
        for event in user_events:
            date = datetime.datetime.fromisoformat(event["timestamp"]).date().isoformat()
            events_by_day[date].append(event)
        
        # Calculate daily activity
        daily_activity = {date: len(events) for date, events in events_by_day.items()}
        
        # Calculate topic progress
        topic_progress = {}
        for topic in self.topics:
            topic_name = topic["name"]
            events = [e for e in user_events if 
                      e["event_type"] == "topic_explored" and 
                      e["details"].get("topic") == topic_name]
            
            if events:
                # Calculate progress based on number of interactions with the topic
                progress = min(1.0, len(events) / 5)  # Assume 5 interactions = 100% progress
                topic_progress[topic_name] = progress
            else:
                topic_progress[topic_name] = 0.0
        
        # Calculate fact mastery
        fact_mastery = {}
        for fact_id, fact in self.facts.items():
            events = [e for e in user_events if 
                     e["event_type"] == "fact_learned" and 
                     e["details"].get("fact_id") == fact_id]
            
            if events:
                fact_mastery[fact_id] = 1.0
            else:
                fact_mastery[fact_id] = 0.0
        
        return {
            "user_id": user_id,
            "total_events": len(user_events),
            "daily_activity": daily_activity,
            "topic_progress": topic_progress,
            "fact_mastery": fact_mastery,
            "topics_mastered": sum(1 for p in topic_progress.values() if p >= 0.8),
            "facts_learned": sum(1 for m in fact_mastery.values() if m > 0),
            "learning_days": len(daily_activity)
        }

# Singleton instance
_instance = None

def get_learning_journey():
    """Get or create the learning journey singleton instance."""
    global _instance
    if _instance is None:
        _instance = LearningJourney()
    return _instance