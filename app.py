import os
import logging
import tempfile
import json
import io
import csv
import traceback
import time  # Added for sleep function
import math  # Added for math functions
import pandas as pd
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, redirect, url_for, Response, session, flash, send_file
import threading
import pygame
import cv2
import numpy as np
from gtts import gTTS
from flask_cors import CORS
from dotenv import load_dotenv
import os

load_dotenv()  # this reads the .env file

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173"])  # Or whatever Vite dev port you use

# Import app_init (centralized app and db setup)
from app_init import app, db

# Import custom modules
from nonverbal_engine import NonverbalEngine
from eye_tracking_service import EyeTrackingService
from sound_recognition_service import SoundRecognitionService
from learning_analytics import LearningAnalytics
from behavior_capture import get_behavior_capture

# Import our new color scheme module instead of the original one
from color_scheme_module import color_scheme_bp
app.register_blueprint(color_scheme_bp)

# Advanced AI modules (import with error handling)
try:
    from interpreter import get_interpreter
    from conversation_engine import get_conversation_engine
    from input_analyzer import get_input_analyzer
    from behavioral_interpreter import get_behavioral_interpreter
    ADVANCED_AI_AVAILABLE = True
    logging.info("Advanced AI modules loaded successfully")
except ImportError as e:
    ADVANCED_AI_AVAILABLE = False
    logging.warning(f"Advanced AI modules not available: {str(e)}")

# Setup logging
logging.basicConfig(level=logging.DEBUG)

# Initialize pygame for audio
try:
    # Try to initialize the audio mixer
    pygame.mixer.init()
    logging.info("Pygame mixer initialized for audio")
except pygame.error:
    # Handle the absence of an audio device
    logging.warning("No audio device available. Audio not initialized.")

# Global variables
ai_running = False
ai_thread = None
nonverbal_engine = None
eye_tracking_service = None
sound_recognition_service = None
interpreter = None
behavior_capture = None

# Data directory
os.makedirs('data', exist_ok=True)
INTERACTIONS_FILE = 'data/user_interactions.json'

# Make functions available to templates
# Define or import the get_current_scheme function
def get_current_scheme(user_id=None):
    """
    Get the current color scheme for a user.
    
    Args:
        user_id: Optional user ID to get personalized color scheme
        
    Returns:
        dict: Color scheme configuration
    """
    try:
        if user_id:
            # Try to get user-specific scheme from database
            from models import UserPreference
            
            # Get user preference for color scheme if it exists
            preference = UserPreference.query.filter_by(
                user_id=user_id, 
                preference_type='color_scheme'
            ).first()
            
            if preference and preference.value:
                # Return user's preferred scheme
                if isinstance(preference.value, str):
                    try:
                        import json
                        return json.loads(preference.value)
                    except:
                        pass
                elif isinstance(preference.value, dict):
                    return preference.value
        
        # If no user_id or no preference found, get default scheme from new module
        from color_scheme_module import get_default_scheme
        return get_default_scheme()
    
    except Exception as e:
        logging.error(f"Error getting color scheme: {str(e)}")
        # Return a simple default scheme if all else fails
        return {
            "primary": "#00ccff",
            "secondary": "#0066cc", 
            "background": "#222222",
            "text": "#ffffff",
            "accent": "#ff9900"
        }

app.jinja_env.globals.update(get_current_scheme=get_current_scheme)

# Initialize services
def init_services():
    global nonverbal_engine, eye_tracking_service, sound_recognition_service, interpreter
    global knowledge_integration, speech_integration, caregiver_dashboard, temporal_engine, behavior_capture
    
    nonverbal_engine = NonverbalEngine()
    
    # Initialize real eye tracking service if available, fall back to simulated service
    try:
        from real_eye_tracking import get_real_eye_tracking_service
        eye_tracking_service = get_real_eye_tracking_service()
        logging.info("Using real eye tracking service")
    except Exception as e:
        logging.warning(f"Real eye tracking not available: {str(e)}, using simulated service")
        eye_tracking_service = EyeTrackingService()
    
    # Initialize real speech recognition if available, fall back to simulated service
    try:
        from real_speech_recognition import get_real_speech_recognition_engine
        speech_engine = get_real_speech_recognition_engine()
        sound_recognition_service = SoundRecognitionService(speech_engine=speech_engine)
        logging.info("Using real speech recognition engine")
    except Exception as e:
        logging.warning(f"Real speech recognition not available: {str(e)}, using simulated service")
        sound_recognition_service = SoundRecognitionService()
        
    # Initialize behavior capture system
    try:
        behavior_capture = get_behavior_capture()
        logging.info("Behavior capture system initialized")
    except Exception as e:
        logging.warning(f"Behavior capture system initialization failed: {str(e)}")
        behavior_capture = None
    
    # Initialize the self-learning and self-modifying processes
    try:
        # Initialize self-learning (but don't start it automatically)
        from ai_learning_engine import get_self_improvement_engine
        learning_engine = get_self_improvement_engine()
        
        # Initialize self-modifying code capability
        from self_modifying_code import get_self_modifying_code_engine
        code_engine = get_self_modifying_code_engine()
        
        # Note: Learning processes will be started via the UI controls,
        # not automatically at initialization
        
        logging.info("Initialized AI self-learning and self-modification capabilities")
    except Exception as e:
        logging.error(f"Error initializing self-learning: {str(e)}")
        logging.error(traceback.format_exc())
    
    # Initialize advanced AI components if available
    if ADVANCED_AI_AVAILABLE:
        try:
            # Initialize the integrated interpreter
            interpreter = get_interpreter()
            logging.info("Advanced AI interpreter initialized")
            
            # Initialize other advanced components
            get_conversation_engine(nonverbal_engine)
            get_input_analyzer()
            get_behavioral_interpreter()
            
            logging.info("All advanced AI components initialized")
        except Exception as e:
            logging.error(f"Error initializing advanced AI components: {str(e)}")
            logging.error(traceback.format_exc())
    
    # Initialize integration modules
    try:
        # Initialize knowledge integration module
        try:
            from modules.knowledge_integration import get_knowledge_integration
            knowledge_integration = get_knowledge_integration()
            logging.info("Knowledge integration initialized")
        except ImportError as e:
            logging.warning(f"Knowledge integration not available: {str(e)}")
            knowledge_integration = None
        
        # Initialize speech integration module
        try:
            from modules.speech_integration import get_speech_integration
            speech_integration = get_speech_integration()
            logging.info("Speech integration initialized")
        except ImportError as e:
            logging.warning(f"Speech integration not available: {str(e)}")
            speech_integration = None
        
        # Initialize caregiver dashboard
        try:
            from modules.caregiver_dashboard import get_caregiver_dashboard
            caregiver_dashboard = get_caregiver_dashboard()
            logging.info("Caregiver dashboard initialized")
        except ImportError as e:
            logging.warning(f"Caregiver dashboard not available: {str(e)}")
            caregiver_dashboard = None
        
        # Initialize audio processor
        try:
            from modules.audio_processor import get_audio_processor
            audio_processor = get_audio_processor()
            logging.info("Audio processor initialized")
        except ImportError as e:
            logging.warning(f"Audio processor not available: {str(e)}")
            audio_processor = None
        
        # Try to initialize temporal engine
        try:
            from attached_assets.engine_temporal import TemporalNonverbalEngine
            temporal_engine = TemporalNonverbalEngine()
            logging.info("Temporal nonverbal engine initialized")
        except ImportError as e:
            logging.warning(f"Temporal nonverbal engine not available: {str(e)}")
            temporal_engine = None
        
        # Try to initialize the conversation integration
        try:
            from attached_assets.conversation_integration import ConversationIntegration
            conversation_integration = ConversationIntegration()
            logging.info("Conversation integration initialized")
        except ImportError as e:
            logging.warning(f"Conversation integration not available: {str(e)}")
        
        # Register routes for integration modules
        
        # Register caregiver routes if dashboard is available
        if caregiver_dashboard:
            try:
                from modules.caregiver_dashboard import register_caregiver_routes
                register_caregiver_routes(app)
                logging.info("Caregiver routes registered")
            except (ImportError, AttributeError) as e:
                logging.warning(f"Could not register caregiver routes: {e}")
        
        # Register audio routes if processor is available
        if audio_processor:
            try:
                from modules.audio_processor import register_audio_routes
                register_audio_routes(app)
                logging.info("Audio routes registered")
            except (ImportError, AttributeError) as e:
                logging.warning(f"Could not register audio routes: {e}")
                
        # Register learning journey routes
        try:
            logging.info("Attempting to register learning routes...")
            try:
                # Import from the routes package which should now expose register_learning_routes
                from routes import register_learning_routes
                logging.info("Import of register_learning_routes successful")
                register_learning_routes(app)
                logging.info("Learning journey routes registered successfully")
            except Exception as route_error:
                logging.error(f"Detailed error registering learning routes: {route_error}")
                logging.error(traceback.format_exc())
                
                # Try to manually import to see where the error is
                try:
                    import inspect
                    from routes import learning_routes
                    logging.info(f"Source of learning_routes: {inspect.getfile(learning_routes)}")
                except Exception as import_error:
                    logging.error(f"Error importing learning_routes directly: {import_error}")
                    logging.error(traceback.format_exc())
            
            # Print registered routes for debugging
            logging.info("Registered routes:")
            for rule in app.url_map.iter_rules():
                logging.info(f"Route: {rule.endpoint} -> {rule}")
                
        except (ImportError, AttributeError) as e:
            logging.error(f"Could not register learning routes: {e}")
            logging.error(traceback.format_exc())
            
            # Fallback route for learning hub to avoid errors
            @app.route('/learning')
            def learning_hub_fallback():
                """Fallback route for Learning Hub."""
                try:
                    from learning_journey import get_learning_journey
                    learning_journey = get_learning_journey()
                    
                    # Get user data and stats
                    user_id = session.get('user_id', 'default_user')
                    topics = learning_journey.topics
                    base_stats = learning_journey.get_learning_statistics(user_id)
                    
                    # Enhance the stats with more details needed by the template
                    stats = {
                        "topics_explored": len(base_stats.get("topic_progress", {}).keys()),
                        "total_topics": len(topics),
                        "facts_learned": base_stats.get("facts_learned", 0),
                        "total_facts": len(learning_journey.facts),
                        "learning_days": base_stats.get("learning_days", 0),
                        "learning_streak": 1,  # Simple default
                        "achievements_earned": 0,
                        "total_achievements": 10,
                        "topic_mastery": {k: int(v * 100) for k, v in base_stats.get("topic_progress", {}).items()},
                        "recent_activities": []
                    }
                    
                    # Add some recent activities based on learning log
                    for event in learning_journey.learning_log[-5:]:
                        if event["user_id"] == user_id:
                            stats["recent_activities"].append({
                                "event_type": event["event_type"],
                                "timestamp": event["timestamp"],
                                "details": event["details"]
                            })
                    
                    # Reverse to get newest first
                    stats["recent_activities"].reverse()
                    
                    return render_template(
                        'learning/hub.html',
                        topics=topics,
                        stats=stats
                    )
                except Exception as e:
                    logging.error(f"Error in learning hub fallback: {e}")
                    return render_template('error.html', 
                                         message="The Learning Hub is currently undergoing maintenance. Please check back later.")
            
        # Register adaptive conversation routes
        try:
            from routes.adaptive_conversation_routes import register_adaptive_routes
            register_adaptive_routes(app)
            logging.info("Adaptive conversation routes registered")
        except (ImportError, AttributeError) as e:
            logging.warning(f"Could not register adaptive conversation routes: {e}")
        
    except Exception as e:
        logging.error(f"Error initializing integration modules: {str(e)}")
        logging.error(traceback.format_exc())

# Save user interaction to JSON file and database
def save_interaction(text, intent, confidence):
    interaction = {
        'text': text,
        'intent': intent,
        'confidence': confidence,
        'timestamp': str(datetime.now())
    }
    
    # Create file if it doesn't exist
    if not os.path.exists(INTERACTIONS_FILE):
        with open(INTERACTIONS_FILE, 'w') as f:
            json.dump([], f)
    
    # Read existing interactions
    with open(INTERACTIONS_FILE, 'r') as f:
        try:
            interactions = json.load(f)
        except json.JSONDecodeError:
            interactions = []
    
    # Add new interaction and save
    interactions.append(interaction)
    with open(INTERACTIONS_FILE, 'w') as f:
        json.dump(interactions, f)
    
    # Also save to database
    try:
        from models import UserInteraction
        user_id = session.get('user_id')
        db_interaction = UserInteraction(
            user_id=user_id,
            text=text,
            intent=intent,
            confidence=confidence
        )
        db.session.add(db_interaction)
        db.session.commit()
        logging.debug(f"Saved interaction to database: {text}")
    except Exception as e:
        logging.error(f"Error saving interaction to database: {str(e)}")

# Global cache for speech files
speech_files = {}
speech_file_max_age = 300  # 5 minutes in seconds
latest_speech_file = None
audio_dir = os.path.join(os.getcwd(), 'static', 'audio')

# Create audio directory if it doesn't exist
os.makedirs(audio_dir, exist_ok=True)

# Text-to-speech function with emotion processing
def text_to_speech(text, emotion=None, emotion_tier=None, voice_id="us_male"):
    """
    Generate and play text-to-speech with emotional context.
    
    Args:
        text (str): Text to speak
        emotion (str, optional): Emotional expression (e.g., positive, negative)
        emotion_tier (str, optional): Intensity of emotion (mild, moderate, strong, urgent)
        voice_id (str, optional): The voice profile to use (default: us_male)
    """
    global latest_speech_file
    
    # Try to use the advanced TTS service first
    try:
        from attached_assets.advanced_tts_service import text_to_speech_with_emotion, get_voice_description
        
        # Get the user's voice preference if available, otherwise use default male voice
        user_id = session.get('user_id')
        if not voice_id or voice_id == "default":
            try:
                if user_id:
                    from models import UserProfile
                    profile = UserProfile.query.filter_by(user_id=user_id).first()
                    if profile and profile.voice_profile:
                        voice_id = profile.voice_profile
                
                # Use male voice as default
                if not voice_id or voice_id == "default":
                    voice_id = "us_male"
            except Exception as e:
                logging.error(f"Error getting voice preference: {str(e)}")
                voice_id = "us_male"  # Fall back to male voice
        
        # Log voice information
        voice_info = get_voice_description(voice_id)
        logging.info(f"Using voice: {voice_info.get('label')} ({voice_id})")
        
        # Use advanced TTS with emotion
        filepath = text_to_speech_with_emotion(
            text=text,
            emotion=emotion,
            emotion_tier=emotion_tier,
            voice_id=voice_id
        )
        
        # Extract filename for client access
        filename = os.path.basename(filepath)
        latest_speech_file = filename
        
        logging.info(f"Generated speech with advanced TTS: {filename}")
        return f"/static/audio/{filename}"
        
    except (ImportError, Exception) as e:
        # Log error and fall back to basic TTS
        logging.warning(f"Advanced TTS failed, falling back to basic TTS: {str(e)}")
        
        # Default speech rate for basic TTS
        rate = 1.0
        volume = 1.0
        
        # Apply emotional context to speech parameters
        if emotion and emotion_tier:
            # Adjust rate based on emotion type
            if emotion == "positive":
                rate = 1.1
            elif emotion == "negative":
                rate = 0.9
            elif emotion == "urgent":
                rate = 1.2
            
            # Further adjust based on emotion tier
            if emotion_tier == "mild":
                rate = rate * 0.9
            elif emotion_tier == "strong":
                rate = rate * 1.1
            elif emotion_tier == "urgent":
                rate = rate * 1.2
                volume = 1.2  # Increase volume for urgent messages
        elif emotion:
            # Fallback if only emotion is provided
            if emotion == "excited":
                rate = 1.2
            elif emotion == "sad":
                rate = 0.8
            elif emotion == "urgent":
                rate = 1.3
        
        # Log speech details
        logging.debug(f"Speaking with emotion: {emotion}, tier: {emotion_tier}, rate: {rate}")
        
        # Create a unique filename based on text and emotion
        import hashlib
        text_hash = hashlib.md5(f"{text}_{emotion}_{emotion_tier}_{rate}".encode()).hexdigest()
        filename = f"{text_hash}.mp3"
        filepath = os.path.join(audio_dir, filename)
        
        # Store the filename for client to access
        latest_speech_file = filename
        
        # Only generate if file doesn't exist
        if not os.path.exists(filepath):
            try:
                # Adjust speech parameters based on emotion
                tts = gTTS(text=text, lang='en', slow=(rate < 0.9))
                tts.save(filepath)
                logging.info(f"Generated speech file: {filename}")
            except Exception as e:
                logging.error(f"Error generating speech: {str(e)}")
                return
    
    # Return the URL for the client to play
    return f"/static/audio/{filename}"

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/public/hardware-test')
def public_hardware_test():
    """Public hardware testing page for microphone and camera that doesn't require login"""
    return render_template('hardware_test_public.html')
    
@app.route('/voice_test')
def voice_test():
    """Direct access to voice testing interface"""
    return render_template('voice_test.html')
    
@app.route('/simple_voice_test')
def simple_voice_test():
    """Simplified voice testing page that's more reliable"""
    return render_template('simple_voice_test.html')
    
@app.route('/demo_male_voice')
def demo_male_voice():
    """Direct demonstration of male voice without requiring JavaScript"""
    try:
        # Generate a voice sample
        text = "This is a demonstration of the male voice for AlphaVox. The US male voice is now the default throughout the system."
        
        # Create the audio file
        speech_file = text_to_speech(
            text=text,
            emotion="neutral",
            emotion_tier="moderate",
            voice_id="us_male"
        )
        
        # Extract filename from the URL
        filename = speech_file.split('/')[-1]
        filepath = os.path.join('static/audio', filename)
        
        # Return a page with an embedded audio element
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Male Voice Demo</title>
            <style>
                body {{ font-family: Arial; background: #222; color: white; padding: 20px; text-align: center; }}
                h1 {{ color: #00ccff; }}
                audio {{ margin: 20px 0; }}
                .container {{ max-width: 600px; margin: 0 auto; background: #333; padding: 20px; border-radius: 10px; }}
                a {{ color: #00ccff; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>AlphaVox Male Voice Demo</h1>
                <p>{text}</p>
                <audio controls autoplay src="/static/audio/{filename}"></audio>
                <p><a href="/">Return to Home</a> | <a href="/simple_voice_test">More Voice Tests</a></p>
            </div>
        </body>
        </html>
        """
    except Exception as e:
        return f"Error generating speech: {str(e)}", 500
    
@app.route('/api/generate_speech', methods=['POST'])
def generate_speech_api():
    """API endpoint to generate text-to-speech with specified parameters"""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    # Extract parameters from request
    text = data.get('text', 'Hello, this is a test of the text to speech system.')
    voice_id = data.get('voice_id', 'us_male')
    emotion = data.get('emotion', 'neutral')
    emotion_tier = data.get('emotion_tier', 'moderate')
    
    # Log the request
    logging.info(f"Generating speech: voice={voice_id}, emotion={emotion}, tier={emotion_tier}")
    
    try:
        # Generate speech using our TTS function
        speech_url = text_to_speech(
            text=text,
            emotion=emotion,
            emotion_tier=emotion_tier,
            voice_id=voice_id
        )
        
        # Return the URL to the client
        return jsonify({
            "status": "success",
            "speech_url": speech_url
        })
    except Exception as e:
        logging.error(f"Error generating speech: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500
    
# API routes for hardware testing
@app.route('/api/audio/devices', methods=['GET'])
def get_audio_devices():
    """API endpoint to get available audio devices"""
    # Return simulated audio devices for now
    devices = [
        {
            'id': 'sim-1',
            'name': 'Default Microphone',
            'channels': 1,
            'default': True
        }
    ]
    return jsonify(devices)

@app.route('/speak')
def speak_text():
    """Simple GET endpoint for text-to-speech that returns the audio file directly"""
    text = request.args.get('text', 'Hello')
    emotion = request.args.get('emotion', 'neutral')
    emotion_tier = request.args.get('emotion_tier', 'moderate')
    voice_id = request.args.get('voice_id', 'us_male')
    
    try:
        # Generate speech and get the filename
        speech_url = text_to_speech(
            text=text,
            emotion=emotion,
            emotion_tier=emotion_tier,
            voice_id=voice_id
        )
        
        # Extract the filename from the URL 
        # speech_url will be something like /static/audio/filename.mp3
        filename = os.path.basename(speech_url)
        file_path = os.path.join('static', 'audio', filename)
        
        # Return the audio file directly 
        return send_file(file_path, mimetype='audio/mpeg')
    except Exception as e:
        app.logger.error(f"Error in speak_text: {str(e)}")
        return "Error generating speech", 500

@app.route('/api/audio/process', methods=['POST'])
def process_audio_api():
    """API endpoint to process audio data for speech recognition"""
    try:
        data = request.get_json()
        if not data or 'audio_data' not in data:
            return jsonify({'error': 'No audio data provided'}), 400
        
        # For testing, just return a simulated response
        return jsonify({
            'text': 'This is a simulated speech recognition response.',
            'confidence': 0.9
        })
    except Exception as e:
        logging.error(f"Error processing audio: {str(e)}")
        return jsonify({'error': str(e)}), 500
        
def generate_simulated_frames():
    """Generate simulated video frames with eye tracking overlay"""
    # Generate a simple test pattern
    while True:
        # Create a test pattern image
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add some visual elements
        # Draw a grid pattern
        for x in range(0, 640, 40):
            cv2.line(frame, (x, 0), (x, 480), (50, 50, 50), 1)
        for y in range(0, 480, 40):
            cv2.line(frame, (0, y), (640, y), (50, 50, 50), 1)
            
        # Draw circle simulating eye tracking
        t = time.time()
        x = int(320 + 200 * math.sin(t))
        y = int(240 + 150 * math.cos(t * 1.3))
        cv2.circle(frame, (x, y), 20, (0, 255, 0), -1)
        
        # Add text
        cv2.putText(frame, "Eye Tracking Simulation", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add crosshair in the center
        cv2.line(frame, (310, 240), (330, 240), (0, 0, 255), 2)
        cv2.line(frame, (320, 230), (320, 250), (0, 0, 255), 2)
        
        # Convert to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        # Yield the frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # Simulate 15 FPS
        time.sleep(1/15)

@app.route('/start', methods=['POST'])
def start():
    name = request.form.get('name', 'User')
    session['name'] = name
    
    # Store user in database
    try:
        from models import User
        
        # Use app context
        with app.app_context():
            # Check if user exists
            user = User.query.filter_by(name=name).first()
            if not user:
                # Create new user
                user = User(name=name)
                db.session.add(user)
                db.session.commit()
                
                # Generate demo data for new users
                # This is only for demonstration purposes
                analytics = LearningAnalytics()
                analytics.generate_demo_data(user.id, count=50)
                
            session['user_id'] = user.id
            
        # Initialize services on first run
        init_services()
    except Exception as e:
        logging.error(f"Error in start: {str(e)}")
        logging.error(traceback.format_exc())
        
    return redirect(url_for('home'))

@app.route('/home')
def home():
    if 'name' not in session:
        return redirect(url_for('index'))
    
    return render_template('home.html', name=session.get('name', 'User'))

@app.route('/hardware_test')
def hardware_test():
    """Hardware testing page for microphone and camera"""
    if 'name' not in session:
        return redirect(url_for('index'))
    
    return render_template('hardware_test.html')

@app.route('/symbols')
def symbols():
    """Symbol-based communication interface"""
    if 'name' not in session:
        return redirect(url_for('index'))
    
    return render_template('symbols.html')

@app.route('/user_profile')
def user_profile():
    """User profile and preferences page"""
    if 'name' not in session:
        return redirect(url_for('index'))
    
    return render_template('profile.html')

@app.route('/process-input', methods=['POST'])
def process_input():
    """Process text input with improved NLP capabilities"""
    input_text = request.form.get('input_text', '')
    
    if not input_text:
        return jsonify({'error': 'No input provided'})
    
    try:
        # Initialize the processor if needed
        from alphavox_input_nlu import get_input_processor
        processor = get_input_processor()
        
        # Create the interaction
        interaction = {"type": "text", "input": input_text}
        user_id = session.get('user_id', 'anonymous')
        
        # Process through AlphaVox NLU
        result = processor.process_interaction(interaction, user_id)
        
        # Extract response fields from the processed result
        message = interaction.get('message', f"I understand you're saying: {input_text}")
        intent = interaction.get('intent', 'communicate')
        confidence = result.get('confidence', 0.9)
        expression = interaction.get('emotion', 'neutral')
        emotion_tier = 'moderate'
        
        # Create response
        response = {
            'intent': intent,
            'message': message,
            'confidence': confidence,
            'expression': expression,
            'emotion_tier': emotion_tier,
            'root_cause': result.get('root_cause', 'unknown'),
            'advanced_ai': True
        }
        
        # Log the advanced processing
        logging.info(f"Processed text with AlphaVox NLU: '{input_text}' → {intent} ({confidence:.2f})")
    
    except Exception as e:
        # Log the error and fall back to basic processing
        logging.error(f"Error using AlphaVox NLU: {str(e)}")
        logging.error(traceback.format_exc())
        
        # Fall back to basic processing
        response = process_input_basic(input_text)
    
    # Save the interaction
    save_interaction(input_text, response['intent'], response['confidence'])
    
    # Generate speech file and get URL
    speech_url = text_to_speech(response['message'], 
                              emotion=response['expression'], 
                              emotion_tier=response['emotion_tier'])
    
    # Add speech URL to the response
    response['speech_url'] = speech_url
    
    return jsonify(response)

def process_input_basic(input_text):
    """Basic text processing fallback"""
    # Process the input using nonverbal engine
    if nonverbal_engine:
        # Basic intent analysis based on keywords
        expression = 'neutral'
        emotion_tier = 'moderate'
        intent = 'communicate'
        confidence = 0.9
        
        # Simple keyword-based emotion detection
        if any(word in input_text.lower() for word in ['help', 'need', 'please', 'urgent']):
            intent = 'help'
            expression = 'urgent'
            emotion_tier = 'strong'
            message = f"I need help: {input_text}"
        elif any(word in input_text.lower() for word in ['happy', 'glad', 'thank', 'good']):
            intent = 'express_joy'
            expression = 'positive'
            emotion_tier = 'moderate'
            message = input_text
        elif any(word in input_text.lower() for word in ['sad', 'upset', 'sorry', 'bad']):
            intent = 'express_sadness'
            expression = 'negative'
            emotion_tier = 'moderate'
            message = input_text
        elif any(word in input_text.lower() for word in ['angry', 'mad', 'stop', 'no']):
            intent = 'express_anger'
            expression = 'negative'
            emotion_tier = 'strong'
            message = input_text
        elif any(word in input_text.lower() for word in ['question', '?', 'ask', 'why', 'how', 'what', 'when', 'where']):
            intent = 'ask_question'
            expression = 'inquisitive'
            emotion_tier = 'mild'
            message = input_text
        else:
            message = f"I want to say: {input_text}"
            
        # Create response
        response = {
            'intent': intent,
            'message': message,
            'confidence': confidence,
            'expression': expression,
            'emotion_tier': emotion_tier,
            'advanced_ai': False
        }
    else:
        # Fallback if engine not initialized
        response = {
            'intent': 'communicate',
            'message': f"I understand you want to say: {input_text}",
            'confidence': 0.9,
            'expression': 'neutral',
            'emotion_tier': 'moderate',
            'advanced_ai': False
        }
    
    return response

@app.route('/speak/greeting', methods=['POST'])
def speak_greeting():
    """Handle greeting messages with TTS using male voice"""
    try:
        data = request.get_json()
        message = data.get('message', 'Hello, welcome to AlphaVox.')
        
        # Use positive expression for greetings
        expression = 'positive'
        emotion_tier = 'moderate'
        
        # Generate speech with male voice
        speech_url = text_to_speech(
            text=message,
            emotion=expression,
            emotion_tier=emotion_tier,
            voice_id="us_male"  # Explicitly use male voice for greetings
        )
        
        # Log the greeting
        logging.info(f"Speaking greeting: {message[:30]}...")
        
        return jsonify({
            'status': 'success',
            'message': message,
            'speech_url': speech_url
        })
    except Exception as e:
        logging.error(f"Error in speak_greeting: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/speak/<gesture>', methods=['POST', 'GET'])
def speak(gesture):
    """Handle simulated gesture inputs with emotional context"""
    
    try:
        # Initialize the processor if needed
        from alphavox_input_nlu import get_input_processor
        processor = get_input_processor()
        
        # Get features from request if POST, or use defaults if GET
        if request.method == 'POST' and request.is_json:
            data = request.json
            features = data.get('features', [0.5, 0.5, 90, 90])
        else:
            # Default features for GET requests
            features = [0.5, 0.5, 90, 90]
        
        # Create the interaction for AlphaVox NLU
        interaction = {"type": "gesture", "input": features}
        user_id = session.get('user_id', 'anonymous')
        
        # Process through AlphaVox NLU
        result = processor.process_interaction(interaction, user_id)
        
        # Extract response fields
        message = interaction.get('message', f"I'm communicating with a {gesture} gesture.")
        intent = interaction.get('intent', 'communicate')
        confidence = result.get('confidence', 0.7)
        expression = interaction.get('emotion', 'neutral')
        emotion_tier = 'moderate'
        
        # Add advanced AI flag to response
        advanced_ai = True
        
        # Log the processing
        logging.info(f"Processed gesture with AlphaVox NLU: '{gesture}' → {intent} ({confidence:.2f})")
        
    except Exception as e:
        # Log the error and fall back to basic processing
        logging.error(f"Error using AlphaVox NLU for gesture: {str(e)}")
        logging.error(traceback.format_exc())
        
        # Fall back to basic gesture processing
        message, intent, confidence, expression, emotion_tier = process_gesture_basic(gesture)
        advanced_ai = False
    
    # Save the interaction
    save_interaction(f"gesture:{gesture}", intent, confidence)
    
    # Generate speech file and get URL
    speech_url = text_to_speech(message, emotion=expression, emotion_tier=emotion_tier)
    
    # Check if this is a GET request (direct browser access)
    if request.method == 'GET':
        # Return HTML with embedded audio player
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AlphaVox - {gesture.capitalize()} Gesture</title>
            <style>
                body {{ font-family: Arial; background: #222; color: white; padding: 20px; text-align: center; }}
                h1 {{ color: #00ccff; }}
                audio {{ margin: 20px 0; }}
                .container {{ max-width: 600px; margin: 0 auto; background: #333; padding: 20px; border-radius: 10px; }}
                .message {{ font-size: 18px; margin: 20px 0; }}
                .gesture {{ color: #00ccff; font-weight: bold; }}
                .back-link {{ margin-top: 20px; }}
                .back-link a {{ color: #00ccff; text-decoration: none; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>AlphaVox Gesture Response</h1>
                <p class="gesture">Gesture: {gesture.capitalize()}</p>
                <p class="message">"{message}"</p>
                <audio controls autoplay src="{speech_url}"></audio>
                <p class="back-link"><a href="javascript:history.back()">← Go Back</a></p>
            </div>
        </body>
        </html>
        """
    else:
        # For POST requests (API calls), return JSON as before
        return jsonify({
            'status': 'success',
            'message': message,
            'intent': intent,
            'confidence': confidence,
            'expression': expression,
            'emotion_tier': emotion_tier,
            'speech_url': speech_url,
            'advanced_ai': advanced_ai,
            'html_audio': f'<audio controls autoplay src="{speech_url}"></audio>'
        })

def process_gesture_basic(gesture):
    """Basic gesture processing fallback"""
    # Map gestures to common phrases (extended)
    gesture_map = {
        'nod': "Yes, I agree.",
        'shake': "No, I don't want that.",
        'point_up': "I need help.",
        'wave': "Hello there!",
        'thumbs_up': "That's great!",
        'thumbs_down': "I don't like that.",
        'open_palm': "Please stop.",
        'stimming': "I need to self-regulate, please give me a moment.",
        'rapid_blink': "I'm feeling overwhelmed."
    }
    
    message = gesture_map.get(gesture, "I'm trying to communicate.")
    
    # Use nonverbal engine to analyze the gesture with emotion
    if nonverbal_engine:
        result = nonverbal_engine.classify_gesture(gesture)
        intent = result.get('intent', 'communicate')
        confidence = result.get('confidence', 0.7)
        expression = result.get('expression', 'neutral')
        emotion_tier = result.get('emotion_tier', 'moderate')
        
        # Add the emotion information to the response
        logging.debug(f"Gesture {gesture} classified as {intent} with emotion {expression}, tier: {emotion_tier}")
    else:
        intent = 'communicate'
        confidence = 0.7
        expression = 'neutral'
        emotion_tier = 'moderate'
    
    return message, intent, confidence, expression, emotion_tier

@app.route('/video_feed')
def video_feed():
    """Route for streaming the video feed with eye tracking"""
    return Response(generate_simulated_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/behavior')
def behavior_capture_page():
    """Behavior capture and analysis page"""
    if 'name' not in session:
        return redirect(url_for('index'))
    
    return render_template('behavior_capture.html')
    
@app.route('/behavior-test')
def behavior_capture_test():
    """Public test version of behavior capture for interface testing"""
    return render_template('behavior_capture.html')
    
@app.route('/api/behavior/start', methods=['POST'])
def start_behavior_capture():
    """Start behavior tracking"""
    global behavior_capture
    
    if not behavior_capture:
        init_services()
    
    if behavior_capture:
        behavior_capture.start_tracking()
        return jsonify({
            'status': 'success',
            'tracking': True,
            'message': 'Behavior capture started'
        })
    else:
        return jsonify({
            'status': 'error',
            'message': 'Behavior capture system not available'
        }), 500
    
@app.route('/api/behavior/stop', methods=['POST'])
def stop_behavior_capture():
    """Stop behavior tracking"""
    global behavior_capture
    
    if not behavior_capture:
        return jsonify({
            'status': 'error',
            'message': 'Behavior capture not initialized'
        })
    
    behavior_capture.stop_tracking()
    
    return jsonify({
        'status': 'success',
        'tracking': False,
        'message': 'Behavior capture stopped'
    })
    
@app.route('/api/behavior/status', methods=['GET'])
def get_behavior_status():
    """Get behavior capture status"""
    global behavior_capture
    
    if not behavior_capture:
        return jsonify({
            'status': 'error',
            'message': 'Behavior capture not initialized'
        })
    
    status = behavior_capture.get_analysis_summary()
    
    return jsonify({
        'status': 'success',
        'tracking_status': status
    })
    
@app.route('/api/behavior/process', methods=['POST'])
def process_behavior_frame():
    """Process a frame for behavior analysis"""
    global behavior_capture
    
    if not behavior_capture:
        return jsonify({
            'status': 'error',
            'message': 'Behavior capture not initialized'
        })
    
    data = request.get_json()
    
    # Validate request
    if not data or 'frame' not in data:
        return jsonify({
            'status': 'error',
            'message': 'No frame data provided'
        })
    
    try:
        # Decode base64 image
        import base64
        import numpy as np
        import cv2
        
        frame_data = base64.b64decode(data['frame'])
        nparr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Process the frame
        results = behavior_capture.process_frame(frame)
        
        # Encode the annotated frame
        _, buffer = cv2.imencode('.jpg', results['frame'])
        annotated_frame_b64 = base64.b64encode(buffer).decode('utf-8')
        
        # Prepare and return results
        response = {
            'status': 'success',
            'tracking': results['tracking'],
            'annotated_frame': annotated_frame_b64
        }
        
        # Add analysis results if available
        if 'results' in results:
            response['results'] = results['results']
        
        return jsonify(response)
    except Exception as e:
        logging.error(f"Error processing behavior frame: {e}")
        logging.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
        
@app.route('/api/behavior/observations', methods=['GET'])
def get_behavior_observations():
    """Get recorded behavior observations"""
    global behavior_capture
    
    if not behavior_capture:
        return jsonify({
            'status': 'error',
            'message': 'Behavior capture not initialized'
        })
    
    try:
        # Look for observations file
        observations_file = os.path.join('data', 'behavior_patterns', 'behavior_observations.json')
        
        if not os.path.exists(observations_file):
            return jsonify({
                'status': 'success',
                'observations': []
            })
        
        with open(observations_file, 'r') as f:
            observations = json.load(f)
        
        return jsonify({
            'status': 'success',
            'count': len(observations),
            'observations': observations
        })
    except Exception as e:
        logging.error(f"Error loading behavior observations: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# Route for symbol-based communication
@app.route('/symbol/<symbol_name>', methods=['POST'])
def process_symbol(symbol_name):
    """Handle communication through symbols"""
    
    try:
        # Initialize the processor if needed
        from alphavox_input_nlu import get_input_processor
        processor = get_input_processor()
        
        # Create the interaction for AlphaVox NLU
        interaction = {"type": "symbol", "input": symbol_name}
        user_id = session.get('user_id', 'anonymous')
        
        # Process through AlphaVox NLU
        result = processor.process_interaction(interaction, user_id)
        
        # Default message patterns based on symbol name
        default_messages = {
            'food': "I'm hungry. I would like something to eat.",
            'drink': "I'm thirsty. I would like something to drink.",
            'bathroom': "I need to use the bathroom.",
            'medicine': "I need my medicine.",
            'happy': "I'm feeling happy!",
            'sad': "I'm feeling sad.",
            'pain': "I'm in pain or discomfort.",
            'tired': "I'm feeling tired.",
            'yes': "Yes.",
            'no': "No.",
            'help': "I need help, please.",
            'question': "I have a question.",
            'play': "I want to play.",
            'music': "I want to listen to music.",
            'book': "I want to read a book.",
            'outside': "I want to go outside."
        }
        
        # Override the message with our default message for known symbols
        if symbol_name in default_messages:
            message = default_messages[symbol_name]  # Prioritize defaults over AI-generated
        else:
            # Use result or interaction message, or a generic fallback
            message = result.get('message') or interaction.get('message')
            if not message:
                message = f"I'm communicating using the {symbol_name} symbol."
                
        intent = result.get('intent', interaction.get('intent', 'communicate'))
        confidence = result.get('confidence', 0.7)
        expression = result.get('emotion', interaction.get('emotion', 'neutral'))
        emotion_tier = 'moderate'
        
        # Log the processing
        logging.info(f"Processed symbol with AlphaVox NLU: '{symbol_name}' → {intent} ({confidence:.2f})")
        
        # Advanced AI is used
        advanced_ai = True
    
    except Exception as e:
        # Log the error and fall back to basic processing
        logging.error(f"Error using AlphaVox NLU for symbol: {str(e)}")
        logging.error(traceback.format_exc())
        
        # Fall back to basic symbol processing
        message, intent, confidence, expression, emotion_tier = process_symbol_basic(symbol_name)
        advanced_ai = False
    
    # Save the interaction
    save_interaction(f"symbol:{symbol_name}", intent, confidence)
    
    # Generate speech file and get URL
    speech_url = text_to_speech(message, emotion=expression, emotion_tier=emotion_tier)
    
    return jsonify({
        'status': 'success',
        'message': message,
        'intent': intent,
        'confidence': confidence,
        'expression': expression,
        'emotion_tier': emotion_tier,
        'symbol': symbol_name,
        'speech_url': speech_url,
        'advanced_ai': advanced_ai,
        'root_cause': result.get('root_cause', 'unknown') if 'result' in locals() else 'unknown'
    })

def process_symbol_basic(symbol_name):
    """Basic symbol processing fallback"""
    # Default symbol messages
    symbol_messages = {
        'food': "I'm hungry. I would like something to eat.",
        'drink': "I'm thirsty. I would like something to drink.",
        'bathroom': "I need to use the bathroom.",
        'medicine': "I need my medicine.",
        'happy': "I'm feeling happy!",
        'sad': "I'm feeling sad.",
        'pain': "I'm in pain or discomfort.",
        'tired': "I'm feeling tired.",
        'yes': "Yes.",
        'no': "No.",
        'help': "I need help, please.",
        'question': "I have a question.",
        'play': "I want to play.",
        'music': "I want to listen to music.",
        'book': "I want to read a book.",
        'outside': "I want to go outside."
    }
    
    # If engine has symbol mapping, use it
    if nonverbal_engine and hasattr(nonverbal_engine, 'symbol_map') and symbol_name in nonverbal_engine.symbol_map:
        result = nonverbal_engine.symbol_map[symbol_name]
        intent = result.get('intent', 'communicate')
        confidence = result.get('confidence', 0.7)
        
        message = symbol_messages.get(symbol_name, f"I'm communicating using the {symbol_name} symbol.")
        
        # Determine emotion for the symbol
        if symbol_name in ['happy', 'yes']:
            expression = 'positive'
            emotion_tier = 'moderate'
        elif symbol_name in ['sad', 'tired', 'no']:
            expression = 'negative'
            emotion_tier = 'moderate'
        elif symbol_name in ['pain', 'medicine']:
            expression = 'negative'
            emotion_tier = 'strong'
        elif symbol_name in ['bathroom', 'food', 'drink', 'help']:
            expression = 'urgent'
            emotion_tier = 'moderate'
        elif symbol_name in ['question']:
            expression = 'inquisitive'
            emotion_tier = 'mild'
        elif symbol_name in ['play', 'music', 'book', 'outside']:
            expression = 'enthusiastic'
            emotion_tier = 'moderate'
        else:
            expression = 'neutral'
            emotion_tier = 'moderate'
    else:
        # Unknown symbol
        intent = 'unknown'
        confidence = 0.4
        message = f"I'm using a symbol but I'm not sure what it means."
        expression = 'confused'
        emotion_tier = 'mild'
    
    return message, intent, confidence, expression, emotion_tier

# Route for getting a user's profile
@app.route('/profile', methods=['GET'])
def get_profile():
    """Get current user profile and preferences"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    user_id = session['user_id']
    
    # Get user from database
    from models import User, UserPreference
    user = User.query.get(user_id)
    
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    # Get user preferences
    profile = UserPreference.get_user_profile(user_id)
    
    # Add default preferences if not present
    default_prefs = {
        'gesture_sensitivity': 0.8,
        'eye_tracking_sensitivity': 0.8,
        'sound_sensitivity': 0.7,
        'preferred_emotion_display': True,
        'response_speed': 1.0,
        'symbol_system': 'default',
        'voice_type': 'default',
        'multimodal_processing': True
    }
    
    for key, value in default_prefs.items():
        if key not in profile:
            profile[key] = value
    
    return jsonify({
        'user': {
            'id': user.id,
            'name': user.name,
            'created_at': user.created_at.isoformat() if user.created_at else None
        },
        'preferences': profile
    })

@app.route('/profile/update', methods=['POST'])
def update_profile():
    """Update user preferences"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    user_id = session['user_id']
    data = request.json
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    # Get preference updates
    preferences = data.get('preferences', {})
    if not preferences:
        return jsonify({'error': 'No preferences provided'}), 400
    
    # Update each preference
    from models import UserPreference
    updated = []
    
    for pref_type, value in preferences.items():
        pref = UserPreference.set_preference(user_id, pref_type, value)
        updated.append(pref_type)
    
    return jsonify({
        'status': 'success',
        'updated': updated
    })

# Routes for starting and stopping the AI assistance
@app.route('/start-ai', methods=['POST'])
def start_ai():
    """Start the AI assistance mode"""
    global ai_running, ai_thread
    
    if ai_running:
        return jsonify({
            'status': 'already_running',
            'message': 'AI assistance is already active'
        })
    
    ai_running = True
    
    # Initialize services if needed
    if not nonverbal_engine:
        init_services()
    
    # Start AI thread if needed (would monitor user activity, etc.)
    # This is a simplified version for the demo
    logging.info("Started AI assistance mode")
    
    return jsonify({
        'status': 'started',
        'message': 'AI assistance activated'
    })

@app.route('/stop-ai', methods=['POST'])
def stop_ai():
    """Stop the AI assistance mode"""
    global ai_running, ai_thread
    
    if not ai_running:
        return jsonify({
            'status': 'already_stopped',
            'message': 'AI assistance is already inactive'
        })
    
    ai_running = False
    
    # Stop AI thread if needed
    # This is a simplified version for the demo
    logging.info("Stopped AI assistance mode")
    
    return jsonify({
        'status': 'stopped',
        'message': 'AI assistance deactivated'
    })

# AI Control routes
@app.route('/ai_control')
def ai_control():
    """AI Control Center for managing autonomous learning"""
    if 'name' not in session:
        return redirect(url_for('index'))
    
    # Get the engines
    from ai_learning_engine import get_self_improvement_engine
    from self_modifying_code import get_self_modifying_code_engine
    
    learning_engine = get_self_improvement_engine()
    code_engine = get_self_modifying_code_engine()
    
    # Get stats
    learning_active = learning_engine.learning_active
    auto_mode_active = code_engine.auto_mode_active
    
    # Prepare sample data for the UI
    recent_improvements = []
    recent_modifications = []
    learning_actions = []
    improvements = {}  # Initialize improvements as empty dict
    
    try:
        # Get real data if available
        improvements = learning_engine.model_optimizer.interaction_stats or {}
        if improvements:
            # Format improvements for display
            for key, stats in improvements.get('intents', {}).items():
                if stats.get('count', 0) > 5:
                    recent_improvements.append({
                        'description': f"Improved recognition for '{key}' intent",
                        'details': f"Based on {stats.get('count', 0)} interactions with {stats.get('success', 0)} successes",
                        'confidence': round(stats.get('confidence_sum', 0) / stats.get('count', 1) * 100) if stats.get('count', 0) > 0 else 0,
                        'timestamp': stats.get('last_used', 'Unknown')
                    })
        
        # Get pending modifications
        modifications = code_engine.code_modifier.modifications
        for mod in modifications[-5:]:  # Show last 5
            recent_modifications.append({
                'file_path': mod.get('file_path', 'Unknown'),
                'status': 'applied' if mod.get('applied', False) else 'pending',
                'description': mod.get('description', '').split('\n')[0],
                'timestamp': mod.get('timestamp', 'Unknown'),
                'diff': mod.get('diff', '')
            })
        
        # Learning actions
        if learning_active:
            learning_actions = [
                "Analyzing user interaction patterns",
                "Optimizing intent classification weights",
                "Processing emotional context correlations",
                "Updating multimodal recognition models"
            ]
    except Exception as e:
        logging.error(f"Error preparing AI control data: {str(e)}")
        logging.error(traceback.format_exc())
    
    # Get research module status
    research_status = {
        'last_update': None,
        'articles_count': 0,
        'insights_count': 0
    }
    
    try:
        if os.path.exists(os.path.join('data', 'research_cache.pkl')):
            import pickle
            with open(os.path.join('data', 'research_cache.pkl'), 'rb') as f:
                cache = pickle.load(f)
                research_status['last_update'] = cache.get('timestamp')
                research_status['articles_count'] = len(cache.get('articles', []))
    except Exception as e:
        logging.error(f"Error loading research status: {str(e)}")
    
    # Stats
    stats = {
        'interactions_count': sum(len(stats) for stats in improvements.values()) if improvements else 0,
        'pending_modifications_count': len([m for m in recent_modifications if m['status'] == 'pending']),
        'applied_modifications_count': len([m for m in recent_modifications if m['status'] == 'applied']),
        'learning_progress': 35,  # Example progress percentage
        'confidence_score': 75,
        'intent_recognition': 82,
        'emotion_processing': 68,
        'self_repair': 54,
        'research_status': research_status
    }
    
    return render_template('ai_control.html',
                          learning_active=learning_active,
                          auto_mode_active=auto_mode_active,
                          recent_improvements=recent_improvements,
                          recent_modifications=recent_modifications,
                          learning_actions=learning_actions,
                          last_optimization=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                          **stats)

# Neural Learning Core Routes

@app.route("/api/learn_root_cause", methods=["POST"])
def learn_root_cause():
    """API endpoint to process an interaction and learn its root cause"""
    data = request.json
    interaction = data.get("interaction")
    user_id = session.get("user_id", "anonymous")
    
    if not interaction:
        return jsonify({"status": "error", "message": "No interaction provided"}), 400
    
    # Import the input processor
    from alphavox_input_nlu import get_input_processor
    input_processor = get_input_processor()
    
    # Process the interaction
    result = input_processor.process_interaction(interaction, user_id)
    
    return jsonify({
        "status": "success", 
        "root_cause": result.get("root_cause", "unknown"), 
        "confidence": result.get("confidence", 0.0)
    })

@app.route("/api/user_insights/<user_id>", methods=["GET"])
def get_user_insights(user_id):
    """Get insights about a user's interactions and root causes"""
    # Check for valid user_id
    if not user_id or user_id == 'current':
        user_id = session.get("user_id", "anonymous")
    
    # Access Neural Learning Core directly for insights
    from neural_learning_core import get_neural_learning_core
    nlc = get_neural_learning_core()
    
    # Get insights
    insights = nlc.get_user_insights(user_id)
    
    return jsonify({
        "status": "success", 
        "insights": insights.get("insights", []), 
        "summary": insights.get("summary", {})
    })

# Research Module Routes
@app.route("/api/update_research", methods=["POST"])
def update_research():
    """API endpoint to update the knowledge base with new research"""
    try:
        from research_module import AlphaVoxResearchModule
        research_module = AlphaVoxResearchModule()
        result = research_module.update_knowledge_base()
        return jsonify(result)
    except Exception as e:
        logging.error(f"Error updating research: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': f"Error: {str(e)}"
        })

# Caregiver Dashboard Routes
# AI Control API routes
@app.route('/ai/start-learning', methods=['POST'])
def start_ai_learning():
    """Start the AI learning process"""
    try:
        from ai_learning_engine import get_self_improvement_engine
        learning_engine = get_self_improvement_engine()
        result = learning_engine.start_learning()
        
        return jsonify({
            'status': 'success' if result else 'already_running',
            'message': 'Learning process started' if result else 'Learning already active'
        })
    except Exception as e:
        logging.error(f"Error starting learning: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': f"Error: {str(e)}"
        })

@app.route('/ai/stop-learning', methods=['POST'])
def stop_ai_learning():
    """Stop the AI learning process"""
    try:
        from ai_learning_engine import get_self_improvement_engine
        learning_engine = get_self_improvement_engine()
        result = learning_engine.stop_learning()
        
        return jsonify({
            'status': 'success' if result else 'already_stopped',
            'message': 'Learning process stopped' if result else 'Learning already inactive'
        })
    except Exception as e:
        logging.error(f"Error stopping learning: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': f"Error: {str(e)}"
        })

@app.route('/ai/start-auto-mode', methods=['POST'])
def start_auto_mode():
    """Start the auto-modification mode"""
    try:
        from self_modifying_code import get_self_modifying_code_engine
        code_engine = get_self_modifying_code_engine()
        result = code_engine.start_auto_mode()
        
        return jsonify({
            'status': 'success' if result else 'already_running',
            'message': 'Auto-modification mode started' if result else 'Auto-modification already active'
        })
    except Exception as e:
        logging.error(f"Error starting auto mode: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': f"Error: {str(e)}"
        })

@app.route('/ai/stop-auto-mode', methods=['POST'])
def stop_auto_mode():
    """Stop the auto-modification mode"""
    try:
        from self_modifying_code import get_self_modifying_code_engine
        code_engine = get_self_modifying_code_engine()
        result = code_engine.stop_auto_mode()
        
        return jsonify({
            'status': 'success' if result else 'already_stopped',
            'message': 'Auto-modification mode stopped' if result else 'Auto-modification already inactive'
        })
    except Exception as e:
        logging.error(f"Error stopping auto mode: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': f"Error: {str(e)}"
        })

@app.route('/ai/queue-modification', methods=['POST'])
def queue_modification():
    """Queue a custom code modification"""
    try:
        data = request.get_json()
        file_path = data.get('file_path')
        issue_description = data.get('issue_description')
        modification_type = data.get('modification_type', 'feature')
        
        if not file_path or not issue_description:
            return jsonify({
                'status': 'error',
                'message': 'Missing required fields'
            })
        
        from self_modifying_code import get_self_modifying_code_engine
        code_engine = get_self_modifying_code_engine()
        
        code_engine.queue_modification(file_path, issue_description, modification_type)
        
        return jsonify({
            'status': 'success',
            'message': 'Modification queued successfully'
        })
    except Exception as e:
        logging.error(f"Error queueing modification: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': f"Error: {str(e)}"
        })

@app.route('/ai/improvements')
def get_improvements():
    """Get recent AI improvements"""
    try:
        from ai_learning_engine import get_self_improvement_engine
        learning_engine = get_self_improvement_engine()
        
        # Get real data if available
        improvements = []
        stats = learning_engine.model_optimizer.interaction_stats
        
        if stats:
            # Format improvements for display
            for key, item_stats in stats.get('intents', {}).items():
                if item_stats.get('count', 0) > 5:
                    improvements.append({
                        'description': f"Improved recognition for '{key}' intent",
                        'details': f"Based on {item_stats.get('count', 0)} interactions with {item_stats.get('success', 0)} successes",
                        'confidence': round(item_stats.get('confidence_sum', 0) / item_stats.get('count', 1) * 100) if item_stats.get('count', 0) > 0 else 0,
                        'timestamp': item_stats.get('last_used', 'Unknown')
                    })
            
            for key, item_stats in stats.get('gestures', {}).items():
                if item_stats.get('count', 0) > 3:
                    improvements.append({
                        'description': f"Optimized '{key}' gesture recognition",
                        'details': f"Based on {item_stats.get('count', 0)} uses",
                        'confidence': round(item_stats.get('confidence_sum', 0) / item_stats.get('count', 1) * 100) if item_stats.get('count', 0) > 0 else 0,
                        'timestamp': item_stats.get('last_used', 'Unknown')
                    })
        
        return jsonify({
            'status': 'success',
            'improvements': improvements
        })
    except Exception as e:
        logging.error(f"Error fetching improvements: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': f"Error: {str(e)}",
            'improvements': []
        })

@app.route('/ai/stats')
def get_ai_stats():
    """Get current AI stats for dashboard"""
    try:
        from ai_learning_engine import get_self_improvement_engine
        from self_modifying_code import get_self_modifying_code_engine
        
        learning_engine = get_self_improvement_engine()
        code_engine = get_self_modifying_code_engine()
        
        # Calculate interaction counts
        interaction_count = 0
        stats = learning_engine.model_optimizer.interaction_stats
        
        for section in stats.values():
            if isinstance(section, dict):
                interaction_count += sum(item.get('count', 0) for item in section.values())
        
        # Get modification counts
        modifications = code_engine.code_modifier.modifications
        pending_mods = len([m for m in modifications if not m.get('applied', False)])
        applied_mods = len([m for m in modifications if m.get('applied', False)])
        
        # Learning progress is a combination of data volume and model quality
        learning_progress = min(95, int(interaction_count / 10) + 30)  # Cap at 95%
        
        # Generate example learning actions
        learning_actions = []
        if learning_engine.learning_active:
            learning_actions = [
                "Analyzing user interaction patterns",
                "Optimizing intent classification weights",
                "Processing emotional context correlations",
                "Updating multimodal recognition models"
            ]
        
        return jsonify({
            'status': 'success',
            'interactions_count': interaction_count,
            'pending_modifications_count': pending_mods,
            'applied_modifications_count': applied_mods,
            'learning_progress': learning_progress,
            'last_optimization': learning_engine.last_optimization.strftime('%Y-%m-%d %H:%M:%S') if hasattr(learning_engine, 'last_optimization') else None,
            'learning_actions': learning_actions,
            'learning_active': learning_engine.learning_active,
            'auto_mode_active': code_engine.auto_mode_active
        })
    except Exception as e:
        logging.error(f"Error getting AI stats: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': f"Error: {str(e)}",
            'interactions_count': 0,
            'pending_modifications_count': 0,
            'applied_modifications_count': 0,
            'learning_progress': 30,
            'learning_actions': [],
            'learning_active': False,
            'auto_mode_active': False
        })

@app.route('/caregiver')
def caregiver_dashboard():
    """Caregiver dashboard for monitoring user communication"""
    if 'name' not in session:
        return redirect(url_for('index'))
    
    # Determine if logged in user is a caregiver (for full implementation)
    # In this demo, we'll just assume the current user can access the dashboard
    
    # Get user (for demo, we'll use the logged in user as the client)
    from models import User, UserInteraction, CaregiverNote, CommunicationProfile, SystemSuggestion
    
    user_id = session.get('user_id', 1)  # Fallback to first user for demo
    user = User.query.get(user_id)
    
    if not user:
        flash('User not found')
        return redirect(url_for('home'))
    
    # Get user data
    interactions = UserInteraction.query.filter_by(user_id=user_id).order_by(UserInteraction.timestamp.desc()).limit(20).all()
    caregiver_notes = CaregiverNote.query.filter_by(user_id=user_id).order_by(CaregiverNote.timestamp.desc()).all()
    communication_profile = CommunicationProfile.get_latest_profile(user_id)
    
    # Get suggestions
    analytics = LearningAnalytics(user_id)
    suggestions = analytics.generate_system_suggestions()
    
    # For demo, convert to SystemSuggestion objects
    system_suggestions = []
    for suggestion in suggestions:
        system_suggestions.append(SystemSuggestion(
            user_id=user_id,
            title=suggestion['title'],
            description=suggestion['description'],
            suggestion_type=suggestion['suggestion_type'],
            confidence=suggestion['confidence'],
            is_active=True,
            is_accepted=False
        ))
    
    # Get frequently used expressions
    frequent_expressions = analytics.get_frequent_expressions()
    
    # Get progress data
    progress = analytics.get_learning_progress()
    
    # Find user observations from caregiver notes
    observations = None
    for note in caregiver_notes:
        if note.tags and 'observation' in note.tags.lower():
            observations = note.content
            break
    
    return render_template('caregiver.html',
                          user=user,
                          interactions=interactions,
                          caregiver_notes=caregiver_notes,
                          communication_profile=communication_profile,
                          system_suggestions=system_suggestions,
                          frequent_expressions=frequent_expressions,
                          progress=progress,
                          observations=observations)

@app.route('/caregiver/add-note', methods=['POST'])
def add_caregiver_note():
    """Add a new caregiver note"""
    if 'name' not in session:
        return jsonify({'status': 'error', 'message': 'Not logged in'}), 401
    
    data = request.json
    content = data.get('content')
    tags = data.get('tags', [])
    
    if not content:
        return jsonify({'status': 'error', 'message': 'Note content is required'}), 400
    
    # Get user (for demo, we'll use the logged in user as the client)
    user_id = session.get('user_id', 1)
    author = session.get('name', 'Caregiver')
    
    # Add the note
    from models import CaregiverNote
    note = CaregiverNote.add_note(user_id, author, content, tags)
    
    if note:
        return jsonify({'status': 'success', 'note_id': note.id})
    else:
        return jsonify({'status': 'error', 'message': 'Failed to add note'}), 500

@app.route('/caregiver/share-data', methods=['POST'])
def share_caregiver_data():
    """Share user data with a healthcare provider"""
    if 'name' not in session:
        return jsonify({'status': 'error', 'message': 'Not logged in'}), 401
    
    data = request.json
    provider_email = data.get('provider_email')
    
    if not provider_email:
        return jsonify({'status': 'error', 'message': 'Provider email is required'}), 400
    
    # In a real implementation, this would create a secure sharing link
    # For this demo, we'll just return success
    
    return jsonify({
        'status': 'success',
        'message': f'Data access link sent to {provider_email}'
    })

@app.route('/caregiver/export', methods=['POST'])
def export_caregiver_data():
    """Export user data in various formats"""
    if 'name' not in session:
        return jsonify({'status': 'error', 'message': 'Not logged in'}), 401
    
    data = request.json
    export_format = data.get('format', 'csv')
    date_range = data.get('date_range', 'all')
    
    # Get user data
    user_id = session.get('user_id', 1)
    
    from models import UserInteraction
    
    # Get interactions based on date range
    if date_range == 'week':
        start_date = datetime.now() - timedelta(days=7)
        interactions = UserInteraction.query.filter(
            UserInteraction.user_id == user_id,
            UserInteraction.timestamp >= start_date
        ).all()
    elif date_range == 'month':
        start_date = datetime.now() - timedelta(days=30)
        interactions = UserInteraction.query.filter(
            UserInteraction.user_id == user_id,
            UserInteraction.timestamp >= start_date
        ).all()
    else:
        # All data
        interactions = UserInteraction.query.filter_by(user_id=user_id).all()
    
    # Format data based on export format
    if export_format == 'csv':
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow(['Timestamp', 'Type', 'Content', 'Intent', 'Confidence'])
        
        # Write data
        for interaction in interactions:
            writer.writerow([
                interaction.timestamp,
                'text' if not interaction.text.startswith('symbol:') and not interaction.text.startswith('gesture:') else 'symbol' if interaction.text.startswith('symbol:') else 'gesture',
                interaction.text,
                interaction.intent,
                interaction.confidence
            ])
        
        # Create response
        response = Response(
            output.getvalue(),
            mimetype='text/csv',
            headers={'Content-Disposition': 'attachment;filename=alphavox_data.csv'}
        )
        
        return response
    
    elif export_format == 'json':
        interaction_list = []
        
        for interaction in interactions:
            interaction_list.append({
                'timestamp': interaction.timestamp.isoformat(),
                'type': 'text' if not interaction.text.startswith('symbol:') and not interaction.text.startswith('gesture:') else 'symbol' if interaction.text.startswith('symbol:') else 'gesture',
                'content': interaction.text,
                'intent': interaction.intent,
                'confidence': interaction.confidence
            })
        
        # Create response
        response = Response(
            json.dumps(interaction_list, indent=2),
            mimetype='application/json',
            headers={'Content-Disposition': 'attachment;filename=alphavox_data.json'}
        )
        
        return response
    
    elif export_format == 'pdf':
        # In a real implementation, this would generate a PDF report
        # For this demo, just return a message
        return jsonify({
            'status': 'error',
            'message': 'PDF export not implemented in demo'
        }), 501
    
    return jsonify({
        'status': 'error',
        'message': f'Unsupported export format: {export_format}'
    }), 400

@app.route('/caregiver/analytics', methods=['GET'])
def get_caregiver_analytics():
    """Get analytics data for caregiver dashboard"""

    if 'name' not in session:
        return jsonify({'status': 'error', 'message': 'Not logged in'}), 401
    
    period = request.args.get('period', 'week')
    user_id = session.get('user_id', 1)
    
    # Get analytics data
    analytics = LearningAnalytics(user_id)
    frequency_data = analytics.get_interaction_frequency(period)
    methods_data = analytics.get_interaction_methods()
    
    return jsonify({
        'status': 'success',
        'frequency': frequency_data,
        'methods': methods_data
    })

# Create all tables in a Flask context
with app.app_context():
    # Import models to ensure tables are created
    from models import User, UserInteraction, UserPreference, CaregiverNote, CommunicationProfile, SystemSuggestion
    db.create_all()

# Import AI learning and self-modification modules
from ai_learning_engine import get_self_improvement_engine
from self_modifying_code import get_self_modifying_code_engine

# Lambda adapter
try:
    from mangum import Mangum
    handler = Mangum(app)
except ImportError:
    # Fallback to local dev server
    if __name__ == '__main__':
        app.run(host='0.0.0.0', port=8000, debug=True)

