"""
Learning routes for AlphaVox

This module provides routes related to the learning journey functionality
of the AlphaVox application.
"""

import logging
from flask import Blueprint, jsonify, request, render_template, current_app

logger = logging.getLogger(__name__)

# Create blueprint for learning routes
learning_bp = Blueprint('learning', __name__, url_prefix='/learning')

@learning_bp.route('/')
def learning_hub():
    """Learning hub main page with personalized learning recommendations."""
    try:
        from learning_journey import get_learning_journey
        learning_journey = get_learning_journey()
        
        # Get user data and stats
        from flask import session
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
            title="Learning Hub",
            topics=topics,
            stats=stats
        )
    except Exception as e:
        import logging
        logging.error(f"Error in learning hub: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return render_template('error.html', 
                             message="The Learning Hub is currently undergoing maintenance. Please check back later.")

@learning_bp.route('/milestones')
def milestones():
    """Show learning milestones progress for the current user."""
    return render_template('learning/milestones.html', title="Learning Milestones")

@learning_bp.route('/sessions')
def sessions():
    """Show learning sessions history."""
    return render_template('learning/sessions.html', title="Learning Sessions")

@learning_bp.route('/analytics')
def analytics():
    """Show learning analytics for the current user."""
    return render_template('learning/analytics.html', title="Learning Analytics")

@learning_bp.route('/start_session', methods=['POST'])
def start_session():
    """Start a new learning session with specific focus areas."""
    data = request.get_json()
    
    # For demonstration purposes, just return success
    # In a real implementation, we would use the LearningSession model
    return jsonify({
        'success': True,
        'session_id': 12345,  # Placeholder ID
        'message': 'Session started successfully'
    })

@learning_bp.route('/end_session/<int:session_id>', methods=['POST'])
def end_session(session_id):
    """End an active learning session."""
    # For demonstration purposes, just return success
    return jsonify({
        'success': True,
        'session_id': session_id,
        'message': 'Session ended successfully'
    })

def register_learning_routes(app):
    """Register learning routes with the Flask application."""
    app.register_blueprint(learning_bp)
    logger.info("Learning routes registered")