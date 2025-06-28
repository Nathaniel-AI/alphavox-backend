"""
Routes for the Dynamic AI-powered conversation difficulty adjustment feature
"""

import logging
from flask import Blueprint, render_template, request, jsonify, session
from adaptive_conversation import get_complexity_engine

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create blueprint
adaptive_bp = Blueprint('adaptive_conversation', __name__, url_prefix='/adaptive')
complexity_engine = get_complexity_engine()

@adaptive_bp.route('/')
def adaptive_dashboard():
    """Dashboard for conversation complexity settings."""
    # Get user ID from session or use default
    user_id = session.get('user_id', 'default_user')
    
    # Get user profile and complexity information
    profile = complexity_engine.get_user_profile(user_id)
    complexity_levels = {i: complexity_engine.get_complexity_level_info(i) for i in range(1, 6)}
    adaptation_modes = {
        mode: complexity_engine.get_adaptation_mode_info(mode)
        for mode in ["fixed", "responsive", "progressive", "adaptive"]
    }
    
    return render_template(
        'adaptive/dashboard.html',
        profile=profile,
        complexity_levels=complexity_levels,
        adaptation_modes=adaptation_modes
    )

@adaptive_bp.route('/settings', methods=['GET'])
def get_settings():
    """API endpoint to get current complexity settings."""
    user_id = session.get('user_id', 'default_user')
    profile = complexity_engine.get_user_profile(user_id)
    
    return jsonify({
        "complexity_level": profile["complexity_level"],
        "adaptation_mode": profile["adaptation_mode"],
        "complexity_variance": profile.get("complexity_variance", 0.5),
        "progression_factor": profile.get("progression_factor", 0.1)
    })

@adaptive_bp.route('/settings', methods=['POST'])
def update_settings():
    """API endpoint to update complexity settings."""
    user_id = session.get('user_id', 'default_user')
    data = request.json or {}
    
    # Validate settings
    updates = {}
    
    if "complexity_level" in data:
        complexity = float(data["complexity_level"])
        if 1 <= complexity <= 5:
            updates["complexity_level"] = complexity
    
    if "adaptation_mode" in data:
        mode = data["adaptation_mode"]
        if mode in ["fixed", "responsive", "progressive", "adaptive"]:
            updates["adaptation_mode"] = mode
    
    if "complexity_variance" in data:
        variance = float(data["complexity_variance"])
        if 0 <= variance <= 1:
            updates["complexity_variance"] = variance
    
    if "progression_factor" in data:
        factor = float(data["progression_factor"])
        if 0 <= factor <= 0.5:
            updates["progression_factor"] = factor
    
    # Apply updates
    if updates:
        profile = complexity_engine.update_user_profile(user_id, updates)
        return jsonify({
            "status": "success",
            "message": "Settings updated successfully",
            "settings": {
                "complexity_level": profile["complexity_level"],
                "adaptation_mode": profile["adaptation_mode"],
                "complexity_variance": profile.get("complexity_variance", 0.5),
                "progression_factor": profile.get("progression_factor", 0.1)
            }
        })
    
    return jsonify({
        "status": "error",
        "message": "No valid settings provided"
    }), 400

@adaptive_bp.route('/test', methods=['POST'])
def test_adaptation():
    """Test endpoint to see how text would be adapted based on current settings."""
    user_id = session.get('user_id', 'default_user')
    data = request.json or {}
    
    original_text = data.get('text', '')
    if not original_text:
        return jsonify({
            "status": "error",
            "message": "No text provided"
        }), 400
    
    # Get context factors that might affect complexity
    context = {
        "topic": data.get('topic'),
        "emotion": data.get('emotion'),
        "time_of_day": data.get('time_of_day')
    }
    
    # Determine appropriate complexity and simplify text
    target_complexity = complexity_engine.determine_response_complexity(user_id, context)
    simplified_text = complexity_engine.simplify_text(original_text, target_complexity)
    
    # Also calculate complexity of original and simplified texts
    original_complexity = complexity_engine._calculate_text_complexity(original_text)
    simplified_complexity = complexity_engine._calculate_text_complexity(simplified_text)
    
    return jsonify({
        "status": "success",
        "original_text": original_text,
        "adapted_text": simplified_text,
        "target_complexity": target_complexity,
        "original_complexity": original_complexity,
        "adapted_complexity": simplified_complexity,
        "complexity_level_info": complexity_engine.get_complexity_level_info(int(target_complexity))
    })

@adaptive_bp.route('/log-interaction', methods=['POST'])
def log_interaction():
    """API endpoint to log a conversation interaction for adaptive learning."""
    user_id = session.get('user_id', 'default_user')
    data = request.json or {}
    
    required_fields = ['user_input', 'response']
    if not all(field in data for field in required_fields):
        return jsonify({
            "status": "error",
            "message": "Missing required fields"
        }), 400
    
    # Extract data
    user_input = data['user_input']
    response = data['response']
    topic = data.get('topic')
    comprehension_feedback = data.get('comprehension_feedback')
    
    # Log the interaction
    complexity_engine.log_interaction(
        user_id=user_id,
        user_input=user_input,
        response=response,
        topic=topic,
        comprehension_feedback=comprehension_feedback
    )
    
    return jsonify({
        "status": "success",
        "message": "Interaction logged successfully"
    })

@adaptive_bp.route('/process', methods=['POST'])
def process_adaptive_response():
    """Process a user input and return an adaptive response with appropriate complexity."""
    user_id = session.get('user_id', 'default_user')
    data = request.json or {}
    
    if 'user_input' not in data:
        return jsonify({
            "status": "error",
            "message": "No user input provided"
        }), 400
    
    user_input = data['user_input']
    context = {
        "topic": data.get('topic'),
        "emotion": data.get('emotion'),
        "time_of_day": data.get('time_of_day')
    }
    
    # Determine ideal complexity for this response
    target_complexity = complexity_engine.determine_response_complexity(user_id, context)
    
    # Generate a response (in a real system, this would call an NLP model)
    # For this example, we'll use a placeholder response
    raw_response = "This is a simulated response to demonstrate the adaptive conversation complexity features of AlphaVox. The system analyzes factors such as user input complexity, topic, emotional context, and adaptation preferences to determine the optimal linguistic complexity of responses."
    
    # Adapt response complexity
    adapted_response = complexity_engine.simplify_text(raw_response, target_complexity)
    
    # Log this interaction
    complexity_engine.log_interaction(
        user_id=user_id,
        user_input=user_input,
        response=adapted_response,
        topic=context.get('topic')
    )
    
    return jsonify({
        "status": "success",
        "user_input": user_input,
        "response": adapted_response,
        "complexity_level": target_complexity,
        "complexity_info": complexity_engine.get_complexity_level_info(int(target_complexity))
    })

@adaptive_bp.route('/history')
def view_interaction_history():
    """View a user's interaction history and complexity trends."""
    user_id = session.get('user_id', 'default_user')
    profile = complexity_engine.get_user_profile(user_id)
    
    # Limit to the last 50 interactions for display
    recent_interactions = profile["interaction_history"][-50:] if profile["interaction_history"] else []
    
    return render_template(
        'adaptive/history.html',
        user_id=user_id,
        interactions=recent_interactions,
        topic_complexities=profile["topic_complexities"]
    )

def register_adaptive_routes(app):
    """Register the adaptive conversation blueprint with the Flask app."""
    app.register_blueprint(adaptive_bp)
    logger.info("Adaptive conversation routes registered")