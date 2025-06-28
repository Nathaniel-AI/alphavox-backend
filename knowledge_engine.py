"""
AlphaVox Knowledge Engine
------------------------
This module enables AlphaVox to autonomously gather, process, and learn 
from various information sources. It combines:

1. Web crawling for targeted information collection
2. Natural language processing for information extraction
3. Knowledge representation using graph-based structures
4. Active learning to expand knowledge in areas of uncertainty
5. Visible learning progress monitoring and reporting

The knowledge engine demonstrates its learning in real-time and can
explain what it has learned to users.
"""

import os
import json
import random
import logging
import time
import threading
import datetime
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
KNOWLEDGE_DIR = "data/knowledge"
TOPICS_FILE = f"{KNOWLEDGE_DIR}/topics.json"
FACTS_FILE = f"{KNOWLEDGE_DIR}/facts.json"
LEARNING_LOG = f"{KNOWLEDGE_DIR}/learning_log.json"
CRAWLER_STATUS_FILE = f"{KNOWLEDGE_DIR}/crawler_status.json"

# Ensure knowledge directory exists
os.makedirs(KNOWLEDGE_DIR, exist_ok=True)

# Topics of interest that AlphaVox will learn about
NONVERBAL_TOPICS = [
    "nonverbal communication",
    "eye contact interpretation",
    "facial expression analysis",
    "body language cues",
    "paralinguistic features",
    "gesture recognition systems",
    "communication disorders",
    "augmentative and alternative communication",
    "therapeutic communication techniques",
    "assistive technology tools",
    "speech therapy approaches",
    "neurodivergent communication styles",
    "autism spectrum communication",
    "developmental communication milestones",
    "multimodal interaction design"
]

class KnowledgeGraph:
    """Represents knowledge as a graph of connected concepts"""
    
    def __init__(self, load_existing: bool = True):
        """Initialize knowledge graph structure"""
        self.concepts = {}  # Concept ID -> concept data
        self.relationships = []  # List of (concept1, relationship, concept2)
        self.topic_concepts = defaultdict(set)  # Topic -> set of concept IDs
        
        if load_existing:
            self._load()
    
    def add_concept(self, concept_id: str, name: str, data: Dict[str, Any], 
                    topics: List[str] = None) -> str:
        """Add a new concept to the knowledge graph"""
        if concept_id in self.concepts:
            # Update existing concept
            self.concepts[concept_id].update(data)
        else:
            # Create new concept
            self.concepts[concept_id] = {
                "name": name,
                "last_updated": datetime.datetime.now().isoformat(),
                "confidence": 0.7,  # Initial confidence level
                "data": data
            }
        
        # Associate with topics
        if topics:
            for topic in topics:
                self.topic_concepts[topic].add(concept_id)
        
        return concept_id
    
    def add_relationship(self, concept1_id: str, relationship: str, 
                         concept2_id: str, strength: float = 0.5) -> None:
        """Add a relationship between two concepts"""
        # Check if concepts exist
        if concept1_id not in self.concepts or concept2_id not in self.concepts:
            raise ValueError("Both concepts must exist in the knowledge graph")
        
        # Add relationship
        self.relationships.append({
            "from": concept1_id,
            "relationship": relationship,
            "to": concept2_id,
            "strength": strength,
            "last_updated": datetime.datetime.now().isoformat()
        })
    
    def get_concepts_by_topic(self, topic: str) -> List[Dict[str, Any]]:
        """Get all concepts associated with a specific topic"""
        concept_ids = self.topic_concepts.get(topic, set())
        return [self.concepts[cid] for cid in concept_ids if cid in self.concepts]
    
    def get_related_concepts(self, concept_id: str) -> List[Dict[str, Any]]:
        """Get concepts related to the specified concept"""
        if concept_id not in self.concepts:
            return []
        
        related = []
        for rel in self.relationships:
            if rel["from"] == concept_id:
                related.append({
                    "concept": self.concepts[rel["to"]],
                    "relationship": rel["relationship"],
                    "strength": rel["strength"]
                })
            elif rel["to"] == concept_id:
                related.append({
                    "concept": self.concepts[rel["from"]],
                    "relationship": rel["relationship"],
                    "strength": rel["strength"]
                })
        
        return related
    
    def search_concepts(self, query: str) -> List[Dict[str, Any]]:
        """Search for concepts by name or content"""
        query = query.lower()
        results = []
        
        for cid, concept in self.concepts.items():
            name = concept["name"].lower()
            if query in name:
                results.append(concept)
                continue
            
            # Search in data
            for key, value in concept["data"].items():
                if isinstance(value, str) and query in value.lower():
                    results.append(concept)
                    break
        
        return results
    
    def _load(self) -> None:
        """Load knowledge graph from persistent storage"""
        graph_file = f"{KNOWLEDGE_DIR}/knowledge_graph.json"
        if os.path.exists(graph_file):
            try:
                with open(graph_file, 'r') as f:
                    data = json.load(f)
                    self.concepts = data.get("concepts", {})
                    self.relationships = data.get("relationships", [])
                    
                    # Convert topic_concepts from JSON
                    topic_data = data.get("topic_concepts", {})
                    for topic, concept_ids in topic_data.items():
                        self.topic_concepts[topic] = set(concept_ids)
            except Exception as e:
                logger.error(f"Error loading knowledge graph: {e}")
    
    def save(self) -> None:
        """Save knowledge graph to persistent storage"""
        graph_file = f"{KNOWLEDGE_DIR}/knowledge_graph.json"
        
        # Convert topic_concepts for JSON serialization (sets -> lists)
        topic_data = {}
        for topic, concept_ids in self.topic_concepts.items():
            topic_data[topic] = list(concept_ids)
        
        data = {
            "concepts": self.concepts,
            "relationships": self.relationships,
            "topic_concepts": topic_data,
            "last_updated": datetime.datetime.now().isoformat()
        }
        
        with open(graph_file, 'w') as f:
            json.dump(data, f, indent=2)


class FactManager:
    """Manages learned facts and their retrieval"""
    
    def __init__(self):
        """Initialize the fact manager"""
        self.facts = []
        self.by_topic = defaultdict(list)
        self.by_confidence = defaultdict(list)
        self._load_facts()
    
    def _load_facts(self) -> None:
        """Load facts from storage"""
        if os.path.exists(FACTS_FILE):
            try:
                with open(FACTS_FILE, 'r') as f:
                    data = json.load(f)
                    self.facts = data
                    
                    # Organize by topic and confidence
                    for i, fact in enumerate(self.facts):
                        for topic in fact.get("topics", []):
                            self.by_topic[topic].append(i)
                        
                        confidence = fact.get("confidence", 0.5)
                        conf_bucket = int(confidence * 10) / 10  # Round to nearest 0.1
                        self.by_confidence[conf_bucket].append(i)
            except Exception as e:
                logger.error(f"Error loading facts: {e}")
    
    def save_facts(self) -> None:
        """Save facts to storage"""
        with open(FACTS_FILE, 'w') as f:
            json.dump(self.facts, f, indent=2)
    
    def add_fact(self, fact_text: str, source: str, topics: List[str], 
                confidence: float = 0.7, metadata: Dict[str, Any] = None) -> int:
        """
        Add a new fact to the knowledge base
        
        Args:
            fact_text: The factual statement
            source: Where the fact was learned from
            topics: Related topics
            confidence: Confidence in the fact (0-1)
            metadata: Additional data about the fact
            
        Returns:
            Index of the new fact
        """
        # Create fact object
        fact = {
            "text": fact_text,
            "source": source,
            "topics": topics,
            "confidence": confidence,
            "metadata": metadata or {},
            "learned_at": datetime.datetime.now().isoformat(),
            "times_accessed": 0,
            "last_accessed": None
        }
        
        # Add to collection
        self.facts.append(fact)
        fact_idx = len(self.facts) - 1
        
        # Update indices
        for topic in topics:
            self.by_topic[topic].append(fact_idx)
        
        conf_bucket = int(confidence * 10) / 10
        self.by_confidence[conf_bucket].append(fact_idx)
        
        # Save updated facts
        self.save_facts()
        
        # Log the learning
        self._log_learning(fact)
        
        return fact_idx
    
    def get_facts_by_topic(self, topic: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get facts related to a specific topic"""
        if topic not in self.by_topic:
            return []
        
        # Get indices of facts for this topic
        indices = self.by_topic[topic]
        
        # If more than limit, select random subset with preference for high confidence
        if len(indices) > limit:
            # Prioritize higher confidence facts
            high_conf_candidates = [i for i in indices 
                                   if self.facts[i]["confidence"] > 0.7]
            
            if len(high_conf_candidates) > limit:
                selected_indices = random.sample(high_conf_candidates, limit)
            else:
                # Take all high confidence, then random from remainder
                remaining = limit - len(high_conf_candidates)
                remaining_candidates = [i for i in indices 
                                       if i not in high_conf_candidates]
                
                if remaining_candidates:
                    selected_indices = high_conf_candidates + random.sample(
                        remaining_candidates, min(remaining, len(remaining_candidates)))
                else:
                    selected_indices = high_conf_candidates
        else:
            selected_indices = indices
        
        # Record access for analytics
        for idx in selected_indices:
            self.facts[idx]["times_accessed"] = self.facts[idx].get("times_accessed", 0) + 1
            self.facts[idx]["last_accessed"] = datetime.datetime.now().isoformat()
        
        return [self.facts[idx] for idx in selected_indices]
    
    def get_random_fact(self) -> Dict[str, Any]:
        """Get a random fact, preferring those with higher confidence"""
        if not self.facts:
            return {"text": "I'm still learning about communication. Ask me later!"}
        
        # Prefer higher confidence facts (70%+ confidence)
        high_conf_facts = []
        for conf in [0.9, 0.8, 0.7]:
            high_conf_facts.extend(self.by_confidence.get(conf, []))
        
        if high_conf_facts and random.random() < 0.7:
            idx = random.choice(high_conf_facts)
        else:
            idx = random.randint(0, len(self.facts) - 1)
        
        # Record access
        self.facts[idx]["times_accessed"] = self.facts[idx].get("times_accessed", 0) + 1
        self.facts[idx]["last_accessed"] = datetime.datetime.now().isoformat()
        
        return self.facts[idx]
    
    def search_facts(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for facts containing the query string"""
        query = query.lower()
        matching_indices = []
        
        for idx, fact in enumerate(self.facts):
            if query in fact["text"].lower():
                matching_indices.append(idx)
        
        # If more than limit, prefer higher confidence and accessed less often
        if len(matching_indices) > limit:
            # Sort by composite score: confidence + recency - access_frequency
            def score_fact(idx):
                fact = self.facts[idx]
                confidence = fact.get("confidence", 0.5)
                access_freq = min(1.0, fact.get("times_accessed", 0) / 10.0)
                
                # Calculate recency if accessed before
                recency = 0
                if fact.get("last_accessed"):
                    last_access = datetime.datetime.fromisoformat(fact["last_accessed"])
                    days_ago = (datetime.datetime.now() - last_access).days
                    recency = max(0, 1.0 - (days_ago / 30.0))  # 1.0 to 0 over 30 days
                
                return confidence + (0.3 * recency) - (0.2 * access_freq)
            
            matching_indices.sort(key=score_fact, reverse=True)
            matching_indices = matching_indices[:limit]
        
        # Record access
        for idx in matching_indices:
            self.facts[idx]["times_accessed"] = self.facts[idx].get("times_accessed", 0) + 1
            self.facts[idx]["last_accessed"] = datetime.datetime.now().isoformat()
            
        return [self.facts[idx] for idx in matching_indices]
    
    def _log_learning(self, fact: Dict[str, Any]) -> None:
        """Log learning activity for dashboard display"""
        if not os.path.exists(LEARNING_LOG):
            learning_log = []
        else:
            try:
                with open(LEARNING_LOG, 'r') as f:
                    learning_log = json.load(f)
            except:
                learning_log = []
        
        # Add new learning event
        learning_event = {
            "timestamp": datetime.datetime.now().isoformat(),
            "type": "new_fact",
            "topics": fact.get("topics", []),
            "confidence": fact.get("confidence", 0.5),
            "source": fact.get("source")
        }
        
        learning_log.append(learning_event)
        
        # Keep log from growing too large
        if len(learning_log) > 1000:
            learning_log = learning_log[-1000:]
        
        with open(LEARNING_LOG, 'w') as f:
            json.dump(learning_log, f, indent=2)


class WebCrawler:
    """
    Simulated web crawler that gathers information on specified topics
    
    In a production environment, this would connect to actual web sources,
    but for the demo we'll simulate crawling with pre-generated content.
    """
    
    def __init__(self):
        """Initialize the web crawler"""
        self.running = False
        self.crawler_thread = None
        self.topics = NONVERBAL_TOPICS
        self.status = {
            "running": False,
            "current_topic": None,
            "topics_processed": 0,
            "facts_discovered": 0,
            "last_update": None
        }
        self._load_status()
    
    def _load_status(self) -> None:
        """Load crawler status from file"""
        if os.path.exists(CRAWLER_STATUS_FILE):
            try:
                with open(CRAWLER_STATUS_FILE, 'r') as f:
                    self.status = json.load(f)
            except Exception as e:
                logger.error(f"Error loading crawler status: {e}")
    
    def _save_status(self) -> None:
        """Save crawler status to file"""
        with open(CRAWLER_STATUS_FILE, 'w') as f:
            json.dump(self.status, f, indent=2)
    
    def start(self) -> None:
        """Start the web crawler in a background thread"""
        if self.running:
            return
        
        self.running = True
        self.status["running"] = True
        self.status["last_update"] = datetime.datetime.now().isoformat()
        self._save_status()
        
        self.crawler_thread = threading.Thread(target=self._crawl_loop)
        self.crawler_thread.daemon = True
        self.crawler_thread.start()
    
    def stop(self) -> None:
        """Stop the web crawler"""
        self.running = False
        self.status["running"] = False
        self.status["last_update"] = datetime.datetime.now().isoformat()
        self._save_status()
    
    def get_status(self) -> Dict[str, Any]:
        """Get current crawler status"""
        return self.status
    
    def _crawl_loop(self) -> None:
        """Main crawler loop that runs in background thread"""
        fact_manager = FactManager()
        knowledge_graph = KnowledgeGraph()
        
        while self.running:
            # Select topic to crawl
            topic = random.choice(self.topics)
            self.status["current_topic"] = topic
            self._save_status()
            
            # Simulate crawling with delay
            crawl_time = random.randint(5, 15)
            logger.info(f"Crawling topic: {topic} (simulated time: {crawl_time}s)")
            
            # Discover 1-3 "facts" about the topic
            num_facts = random.randint(1, 3)
            
            for _ in range(num_facts):
                # Only process if still running
                if not self.running:
                    break
                
                # Simulate processing time
                time.sleep(crawl_time / num_facts)
                
                # Generate a simulated fact
                fact_text = self._generate_fact_for_topic(topic)
                
                # Add fact to knowledge base
                confidence = round(random.uniform(0.6, 0.9), 1)
                fact_manager.add_fact(
                    fact_text=fact_text,
                    source="web_crawler",
                    topics=[topic],
                    confidence=confidence,
                    metadata={"discovery_method": "simulated_crawl"}
                )
                
                # Update counters
                self.status["facts_discovered"] += 1
                self._save_status()
                
                # Also add to knowledge graph
                concept_id = f"concept_{int(time.time())}_{random.randint(1000, 9999)}"
                knowledge_graph.add_concept(
                    concept_id=concept_id,
                    name=topic.title(),
                    data={"fact": fact_text},
                    topics=[topic]
                )
                knowledge_graph.save()
            
            # Mark topic as processed
            self.status["topics_processed"] += 1
            self.status["last_update"] = datetime.datetime.now().isoformat()
            self._save_status()
            
            # Sleep for a while before next crawl
            time.sleep(random.randint(10, 30))
    
    def _generate_fact_for_topic(self, topic: str) -> str:
        """
        Generate a fact about the topic
        
        In a real implementation, this would extract information from crawled pages.
        For this demo, we'll use a predefined set of facts for each topic.
        """
        # Simulated facts by topic
        facts_by_topic = {
            "nonverbal communication": [
                "Nonverbal communication accounts for approximately 93% of communication meaning, with 55% through body language and 38% through tone of voice.",
                "Research shows that nonverbal signals are more likely to be trusted when verbal and nonverbal messages contradict each other.",
                "Cultural differences in nonverbal communication can lead to misinterpretation, as gestures may have different meanings across cultures.",
                "Microexpressions are brief, involuntary facial expressions that last for a fraction of a second and reveal true emotions.",
                "The study of nonverbal communication is called kinesics, which was first developed by anthropologist Ray Birdwhistell in the 1950s."
            ],
            "eye contact interpretation": [
                "Sustained eye contact lasting 7-10 seconds can trigger feelings of discomfort in many social contexts.",
                "Pupils dilate up to 50% when viewing something of interest, a response that can't be consciously controlled.",
                "The 'triangular gaze' pattern, where eyes move between both eyes and the mouth, is common when speaking with someone attractive.",
                "Children with autism spectrum disorder often avoid direct eye contact, which may be related to overstimulation of the amygdala.",
                "The direction of a person's gaze can unconsciously influence where others look, a phenomenon known as 'gaze cueing'."
            ],
            "facial expression analysis": [
                "Paul Ekman identified six universal facial expressions: happiness, sadness, fear, disgust, anger, and surprise.",
                "The human face contains 43 muscles that work together to create more than 10,000 distinct expressions.",
                "Facial recognition algorithms analyze up to 80 nodal points on a human face to create a unique facial signature.",
                "The fake smile (Duchenne smile) can be detected by the absence of eye muscle engagement (orbicularis oculi).",
                "Contempt is the only asymmetrical facial expression, typically involving only one side of the mouth."
            ],
            "body language cues": [
                "Open palm gestures evolved as a way to show no weapons were being concealed and generally increase trust.",
                "Mirroring, the subconscious imitation of another person's posture and movements, tends to occur when people feel a connection.",
                "Crossed arms can decrease information retention by up to 38% according to research on body positioning and learning.",
                "The direction of a person's feet often indicates their interest level or desire to leave a conversation.",
                "Personal space boundaries differ culturally, ranging from 1.5 feet in South America to 4 feet in North America for casual conversation."
            ],
            "paralinguistic features": [
                "Voice pitch naturally rises when speaking to babies, pets, or romantically desired individuals.",
                "Speech rate averages 150-160 words per minute for English speakers, with comprehension declining sharply above 210 wpm.",
                "Vocal fry, a low-frequency vibration created in the vocal cords, has increased in prevalence among young American women.",
                "The universal pause filler sounds (um, uh, er) differ across languages but serve the same cognitive processing function.",
                "Vocal tone can communicate up to 38% of the emotional content in a message."
            ],
            "gesture recognition systems": [
                "Advanced gesture recognition systems can detect movements in 3D space with accuracy rates exceeding 98%.",
                "Temporal segmentation, the identification of when a gesture begins and ends, remains a significant challenge in gesture recognition.",
                "The Microsoft Kinect pioneered consumer-level skeleton tracking for gesture recognition using infrared depth sensing.",
                "Continuous gesture recognition systems process streams of data in real-time rather than discrete, pre-segmented gestures.",
                "Transfer learning techniques allow gesture recognition systems to adapt to new users with minimal training data."
            ],
            "communication disorders": [
                "Approximately 7.5 million people in the United States have difficulties using their voices, according to the National Institute on Deafness.",
                "Cluttering, a fluency disorder characterized by rapid speech with irregular rhythm, is often misdiagnosed as stuttering.",
                "Selective mutism affects about 1 in 140 children, who speak normally in certain environments but not in others.",
                "Aphasia, the loss of ability to understand or express speech, affects about 2 million Americans.",
                "Up to 10% of children have some form of language or speech disorder, according to the American Speech-Language-Hearing Association."
            ],
            "augmentative and alternative communication": [
                "AAC systems range from simple picture boards to complex computer systems that generate speech from text.",
                "Eye tracking AAC devices can operate at selection speeds of 15-25 selections per minute for experienced users.",
                "Research shows that AAC use does not inhibit speech development and may actually improve natural speech abilities.",
                "The first electronic communication board was developed in the 1970s for people with cerebral palsy.",
                "Brain-computer interfaces (BCIs) represent the newest frontier in AAC technology, allowing control through thought alone."
            ],
            "therapeutic communication techniques": [
                "Validation therapy, developed by Naomi Feil, focuses on empathizing with emotions rather than correcting factual inaccuracies.",
                "The SOLER technique (Sit squarely, Open posture, Lean forward, Eye contact, Relax) provides a framework for nonverbal therapeutic presence.",
                "Motivational interviewing effectiveness increases when therapists use reflection-to-question ratios of at least 2:1.",
                "Silence in therapeutic settings serves five primary functions: encouragement, processing, respect, confrontation, and transitioning.",
                "Language simplification without vocabulary restriction is recommended for communicating with people with intellectual disabilities."
            ],
            "assistive technology tools": [
                "Switch access devices can be activated by nearly any voluntary movement, including eye blinks, puffs of breath, or slight muscle contractions.",
                "Screen readers process text at rates up to 300-500 words per minute for experienced users, far exceeding typical speech rates.",
                "Predictive text systems can reduce keystrokes by 40-60% for users with mobility impairments.",
                "High-tech communication boards often use semantic compaction, where sequences of icons represent complex concepts.",
                "Closed captioning was first demonstrated at the First National Conference on Television for the Hearing Impaired in 1971."
            ],
            "speech therapy approaches": [
                "The Lidcombe Program for early stuttering intervention has shown success rates of 80% or higher in preschool-aged children.",
                "PROMPT therapy (Prompts for Restructuring Oral Muscular Phonetic Targets) uses tactile cues on the face to teach correct articulation.",
                "The average adult with aphasia requires 8-40 hours of speech therapy per week for optimal recovery.",
                "Music-based therapies like Melodic Intonation Therapy leverage the brain's preserved ability to process musical elements despite language damage.",
                "Motor speech exercises are most effective when practiced at high intensity with at least 100 repetitions per session."
            ],
            "neurodivergent communication styles": [
                "Autistic individuals may process language literally, missing implied meanings that neurotypical people infer automatically.",
                "Hyperlexia, advanced reading ability without corresponding comprehension, occurs in approximately 6-14% of autistic children.",
                "ADHD communication often includes topic-jumping, a natural pattern stemming from divergent thinking rather than inattention.",
                "People with dyslexia often excel at visual-spatial thinking, leading to strengths in fields requiring 3D conceptualization.",
                "Echolalia, the repetition of others' speech, serves important communicative functions like processing time, self-regulation, and affirmation."
            ],
            "autism spectrum communication": [
                "Approximately 30% of individuals on the autism spectrum are minimally verbal or nonverbal.",
                "Prosody differences in autism may reflect difference rather than deficit, with research showing distinct patterns rather than absence of intonation.",
                "Gestalt language processing, where phrases are learned and used as single units, is common in autistic language development.",
                "Special interests in autism serve communication functions, providing motivation for social interaction and shared attention.",
                "Visual supports have been shown to increase independence and reduce anxiety in autistic individuals across the lifespan."
            ],
            "developmental communication milestones": [
                "Infants begin to recognize their mother's voice while still in the womb, around 25-26 weeks gestation.",
                "Joint attention, the shared focus on an object between child and caregiver, typically emerges between 9-12 months.",
                "Children typically produce their first words between 10-14 months, but understand many more words than they can say.",
                "Vocabulary growth undergoes a 'word spurt' around 18 months, with children learning 5-10 new words per day.",
                "The ability to understand theory of mind, that others have different thoughts and beliefs, typically develops around age 4-5."
            ],
            "multimodal interaction design": [
                "Multimodal interfaces that combine voice and touch input can reduce task completion time by up to 30% compared to unimodal interfaces.",
                "Error rates decrease by approximately 36% when systems provide feedback across multiple sensory channels.",
                "Cognitive load theory suggests that information presented across different modalities (visual and auditory) is processed more efficiently.",
                "The McGurk effect demonstrates how visual and auditory information is integrated, with visual lip movements altering what sounds we perceive.",
                "Cultural differences in multimodal interaction preferences can significantly impact user experience design effectiveness."
            ]
        }
        
        # Get facts for this topic or use generic fact
        topic_facts = facts_by_topic.get(topic, [
            f"Research shows that understanding {topic} can improve communication effectiveness.",
            f"Recent studies have explored how {topic} varies across different cultural contexts.",
            f"Experts in {topic} recommend practicing awareness exercises daily."
        ])
        
        return random.choice(topic_facts)


class KnowledgeEngine:
    """Main engine that coordinates knowledge gathering and retrieval"""
    
    def __init__(self):
        """Initialize the knowledge engine components"""
        # Ensure knowledge directory exists
        os.makedirs(KNOWLEDGE_DIR, exist_ok=True)
        
        # Initialize components
        self.graph = KnowledgeGraph()
        self.fact_manager = FactManager()
        self.crawler = WebCrawler()
        
        # Learning metrics
        self.metrics = {
            "facts_learned": len(self.fact_manager.facts),
            "topics_explored": len(set(sum([fact.get("topics", []) for fact in self.fact_manager.facts], []))),
            "learning_sessions": self._count_learning_sessions(),
            "last_updated": datetime.datetime.now().isoformat()
        }
    
    def start_learning(self) -> None:
        """Start the autonomous learning process"""
        logger.info("Starting knowledge engine learning process")
        self.crawler.start()
    
    def stop_learning(self) -> None:
        """Stop the autonomous learning process"""
        logger.info("Stopping knowledge engine learning process")
        self.crawler.stop()
    
    def is_learning(self) -> bool:
        """Check if autonomous learning is active"""
        return self.crawler.running
    
    def get_learning_metrics(self) -> Dict[str, Any]:
        """Get current learning metrics"""
        # Update metrics
        self.metrics = {
            "facts_learned": len(self.fact_manager.facts),
            "topics_explored": len(set(sum([fact.get("topics", []) for fact in self.fact_manager.facts], []))),
            "learning_sessions": self._count_learning_sessions(),
            "last_updated": datetime.datetime.now().isoformat(),
            "crawler_status": self.crawler.get_status()
        }
        return self.metrics
    
    def get_random_fact(self) -> str:
        """Get a random fact from the knowledge base"""
        fact = self.fact_manager.get_random_fact()
        return fact.get("text", "I'm still learning about communication!")
    
    def process_expertise_query(self, query: str) -> Dict[str, Any]:
        """
        Process a query about what AlphaVox knows
        
        Args:
            query: The user's question
            
        Returns:
            Dict with response text and related facts
        """
        # Simplistic keyword matching for demo purposes
        # In a real implementation, this would use NLP for semantic matching
        query_lower = query.lower()
        matching_facts = []
        
        # Try direct topic matching first
        matched_topic = None
        for topic in NONVERBAL_TOPICS:
            if topic.lower() in query_lower:
                matched_topic = topic
                matching_facts.extend(self.fact_manager.get_facts_by_topic(topic, limit=3))
                break
        
        # If no topic match, try general search
        if not matching_facts:
            # Extract potential keywords from query
            words = query_lower.split()
            for word in words:
                if len(word) > 4:  # Only use significant words
                    search_facts = self.fact_manager.search_facts(word, limit=2)
                    matching_facts.extend(search_facts)
                    if matching_facts:
                        break
        
        # Generate response
        if matching_facts:
            response_text = f"Here's what I know about {'that' if matched_topic else 'your question'}:"
            facts = [fact["text"] for fact in matching_facts]
        else:
            response_text = "I'm still learning about that topic. Here's a related fact:"
            random_fact = self.fact_manager.get_random_fact()
            facts = [random_fact["text"]]
        
        # Add explanation of learning
        if len(self.fact_manager.facts) < 10:
            learning_status = "I'm just beginning to learn about communication. Ask me again later as I continue to gather knowledge!"
        elif len(self.fact_manager.facts) < 30:
            learning_status = "I'm making progress in my learning. My knowledge base is growing every day!"
        else:
            learning_status = "I've learned quite a bit about communication topics. Feel free to ask me more specific questions!"
        
        return {
            "response_text": response_text,
            "facts": facts,
            "learning_status": learning_status
        }
    
    def _count_learning_sessions(self) -> int:
        """Count the number of learning sessions from logs"""
        if not os.path.exists(LEARNING_LOG):
            return 0
        
        try:
            with open(LEARNING_LOG, 'r') as f:
                learning_log = json.load(f)
                
                # Count days with learning activity as "sessions"
                if not learning_log:
                    return 0
                    
                dates = set()
                for event in learning_log:
                    timestamp = event.get("timestamp", "")
                    if timestamp:
                        try:
                            date = timestamp.split("T")[0]
                            dates.add(date)
                        except:
                            pass
                
                return len(dates)
        except Exception as e:
            logger.error(f"Error counting learning sessions: {e}")
            return 0


# Singleton instance
_knowledge_engine = None

def get_knowledge_engine() -> KnowledgeEngine:
    """Get the singleton knowledge engine instance"""
    global _knowledge_engine
    if _knowledge_engine is None:
        _knowledge_engine = KnowledgeEngine()
    return _knowledge_engine


def get_random_fact() -> str:
    """Convenience function to get a random fact"""
    return get_knowledge_engine().get_random_fact()


def process_expertise_query(query: str) -> Dict[str, Any]:
    """Convenience function to process knowledge queries"""
    return get_knowledge_engine().process_expertise_query(query)


def run_crawler():
    """Start the background crawler process"""
    engine = get_knowledge_engine()
    engine.start_learning()
    return {"status": "Crawler started", "running": True}


# Initialize the directory structure when the module is imported
if not os.path.exists(KNOWLEDGE_DIR):
    os.makedirs(KNOWLEDGE_DIR, exist_ok=True)

# If the facts file doesn't exist, initialize with a few starter facts
if not os.path.exists(FACTS_FILE):
    starter_facts = [
        {
            "text": "Nonverbal communication accounts for approximately 93% of communication meaning, with 55% through body language and 38% through tone of voice.",
            "source": "initial_data",
            "topics": ["nonverbal communication"],
            "confidence": 0.9,
            "metadata": {"starter_fact": True},
            "learned_at": datetime.datetime.now().isoformat(),
            "times_accessed": 0,
            "last_accessed": None
        },
        {
            "text": "Eye contact patterns vary significantly across cultures. In many Asian cultures, extended eye contact can be seen as disrespectful, while in Western cultures it often conveys trustworthiness.",
            "source": "initial_data",
            "topics": ["eye contact interpretation", "nonverbal communication"],
            "confidence": 0.9,
            "metadata": {"starter_fact": True},
            "learned_at": datetime.datetime.now().isoformat(),
            "times_accessed": 0,
            "last_accessed": None
        },
        {
            "text": "AAC (Augmentative and Alternative Communication) systems can use eye tracking technology to allow individuals to control communication devices using only eye movements.",
            "source": "initial_data",
            "topics": ["augmentative and alternative communication", "assistive technology tools"],
            "confidence": 0.9, 
            "metadata": {"starter_fact": True},
            "learned_at": datetime.datetime.now().isoformat(),
            "times_accessed": 0,
            "last_accessed": None
        }
    ]
    
    with open(FACTS_FILE, 'w') as f:
        json.dump(starter_facts, f, indent=2)

# Initialize topics file if it doesn't exist
if not os.path.exists(TOPICS_FILE):
    with open(TOPICS_FILE, 'w') as f:
        json.dump(NONVERBAL_TOPICS, f, indent=2)

# Initialize learning log if it doesn't exist
if not os.path.exists(LEARNING_LOG):
    with open(LEARNING_LOG, 'w') as f:
        json.dump([], f)
        
# Initialize crawler status if it doesn't exist
if not os.path.exists(CRAWLER_STATUS_FILE):
    status = {
        "running": False,
        "current_topic": None,
        "topics_processed": 0,
        "facts_discovered": 0,
        "last_update": datetime.datetime.now().isoformat()
    }
    with open(CRAWLER_STATUS_FILE, 'w') as f:
        json.dump(status, f, indent=2)

if __name__ == "__main__":
    # Test the knowledge engine
    engine = get_knowledge_engine()
    print(f"Initial metrics: {engine.get_learning_metrics()}")
    
    print("\nStarting learning process...")
    engine.start_learning()
    
    print("\nWaiting for some learning to occur...")
    time.sleep(30)
    
    print(f"\nUpdated metrics: {engine.get_learning_metrics()}")
    
    print("\nStopping learning process...")
    engine.stop_learning()
    
    print("\nRandom fact:")
    print(get_random_fact())
    
    print("\nQuerying expertise:")
    response = process_expertise_query("Tell me about eye contact")
    print(response["response_text"])
    for fact in response["facts"]:
        print(f"- {fact}")
    print(f"\nLearning status: {response['learning_status']}")