"""
Health check routes for AlphaVox

This module provides health check endpoints for AWS load balancers and other monitoring tools
to verify that the application is running properly.
"""

import logging
import os
import sys
import time
from datetime import datetime
from flask import Blueprint, jsonify, current_app

logger = logging.getLogger(__name__)

# Create a blueprint for health check routes
health_bp = Blueprint('health', __name__)

# Track application start time for uptime reporting
APP_START_TIME = datetime.utcnow()

@health_bp.route('/health')
def health_check():
    """
    Basic health check endpoint for load balancers.
    Returns 200 OK if the application is running.
    """
    return jsonify({'status': 'ok', 'service': 'alphavox'})

@health_bp.route('/health/detailed')
def detailed_health_check():
    """
    Detailed health check with system information.
    Useful for more comprehensive monitoring.
    """
    # Calculate uptime
    uptime_seconds = (datetime.utcnow() - APP_START_TIME).total_seconds()
    uptime_formatted = str(int(uptime_seconds // 86400)) + "d " + \
                      time.strftime('%H:%M:%S', time.gmtime(uptime_seconds % 86400))
    
    # Get current environment
    environment = os.environ.get('FLASK_ENV', 'development')
    
    # Check database connectivity
    db_status = _check_database()
    
    # Get external API status
    openai_api_status = _check_external_api('OPENAI_API_KEY')
    anthropic_api_status = _check_external_api('ANTHROPIC_API_KEY')
    
    # System information
    system_info = {
        'python_version': sys.version,
        'environment': environment,
        'uptime': uptime_formatted,
        'uptime_seconds': int(uptime_seconds),
        'start_time': APP_START_TIME.isoformat(),
        'current_time': datetime.utcnow().isoformat()
    }
    
    # Collect all status information
    status_info = {
        'status': 'ok' if db_status['status'] == 'ok' else 'degraded',
        'service': 'alphavox',
        'database': db_status,
        'external_apis': {
            'openai': openai_api_status,
            'anthropic': anthropic_api_status
        },
        'system': system_info
    }
    
    return jsonify(status_info)

def _check_database():
    """Check database connectivity."""
    try:
        # Import here to avoid circular imports
        from app_init import db
        from sqlalchemy import text
        
        # Execute a simple query to check database connectivity
        with current_app.app_context():
            db.session.execute(text('SELECT 1'))
        
        return {
            'status': 'ok',
            'message': 'Database connection successful'
        }
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return {
            'status': 'error',
            'message': f"Database connection failed: {str(e)}"
        }

def _check_external_api(api_key_name):
    """Check if an external API key is configured."""
    api_key = os.environ.get(api_key_name)
    
    if not api_key:
        return {
            'status': 'not_configured',
            'message': f"{api_key_name} is not configured"
        }
    
    return {
        'status': 'configured',
        'message': f"{api_key_name} is configured"
    }

def register_health_routes(app):
    """Register health check routes with the Flask application."""
    app.register_blueprint(health_bp)
    logger.info("Health routes registered")