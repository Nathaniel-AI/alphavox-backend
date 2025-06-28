"""
Color Scheme Module for AlphaVox

This module provides color scheme functionality including:
- Color scheme management
- UI theme configuration
- Blueprint with routes for color scheme settings
"""

import logging
from flask import Blueprint, request, jsonify, session, render_template

# Create blueprint for color scheme routes
color_scheme_bp = Blueprint('color_scheme', __name__)

# Initialize default color schemes
DEFAULT_SCHEMES = {
    "default": {
        "primary": "#00ccff",
        "secondary": "#0066cc",
        "background": "#222222",
        "text": "#ffffff",
        "accent": "#ff9900"
    },
    "high_contrast": {
        "primary": "#ffffff",
        "secondary": "#ffff00",
        "background": "#000000",
        "text": "#ffffff",
        "accent": "#00ff00"
    },
    "calm": {
        "primary": "#6495ED",
        "secondary": "#87CEEB",
        "background": "#F0F8FF",
        "text": "#2F4F4F",
        "accent": "#20B2AA"
    }
}

def get_default_scheme():
    """Get the default color scheme"""
    return DEFAULT_SCHEMES["default"]

def get_scheme(scheme_name):
    """Get a specific color scheme by name"""
    return DEFAULT_SCHEMES.get(scheme_name, DEFAULT_SCHEMES["default"])

def get_user_scheme(user_id):
    """Get a user's preferred color scheme"""
    try:
        from models import UserPreference
        
        # Try to get user preference
        preference = UserPreference.query.filter_by(
            user_id=user_id,
            preference_type='color_scheme'
        ).first()
        
        if preference and preference.value:
            # If preference is a scheme name, get that scheme
            if preference.value in DEFAULT_SCHEMES:
                return DEFAULT_SCHEMES[preference.value]
            # If preference is JSON, parse it
            if isinstance(preference.value, str):
                try:
                    import json
                    return json.loads(preference.value)
                except:
                    pass
            # If preference is already a dict
            if isinstance(preference.value, dict):
                return preference.value
                
    except Exception as e:
        logging.error(f"Error getting user color scheme: {e}")
    
    # Return default if anything fails
    return get_default_scheme()

# Routes for the blueprint
@color_scheme_bp.route('/api/color-schemes', methods=['GET'])
def list_schemes():
    """List all available color schemes"""
    return jsonify({
        'status': 'success',
        'schemes': list(DEFAULT_SCHEMES.keys()),
        'current_scheme': session.get('color_scheme', 'default')
    })

@color_scheme_bp.route('/api/color-schemes/<scheme_name>', methods=['GET'])
def get_scheme_api(scheme_name):
    """Get a specific color scheme by name"""
    if scheme_name in DEFAULT_SCHEMES:
        return jsonify({
            'status': 'success',
            'scheme': DEFAULT_SCHEMES[scheme_name]
        })
    else:
        return jsonify({
            'status': 'error',
            'message': f'Scheme "{scheme_name}" not found'
        }), 404

@color_scheme_bp.route('/api/color-schemes/current', methods=['GET', 'POST'])
def current_scheme():
    """Get or set the current color scheme"""
    if request.method == 'POST':
        data = request.get_json()
        scheme_name = data.get('scheme')
        
        if not scheme_name:
            return jsonify({
                'status': 'error',
                'message': 'No scheme name provided'
            }), 400
            
        if scheme_name not in DEFAULT_SCHEMES:
            return jsonify({
                'status': 'error',
                'message': f'Scheme "{scheme_name}" not found'
            }), 404
            
        # Save to session
        session['color_scheme'] = scheme_name
        
        # If user is logged in, save to database
        if 'user_id' in session:
            try:
                from models import UserPreference
                UserPreference.set_preference(
                    session['user_id'],
                    'color_scheme',
                    scheme_name
                )
            except Exception as e:
                logging.error(f"Error saving color scheme preference: {e}")
        
        return jsonify({
            'status': 'success',
            'scheme': DEFAULT_SCHEMES[scheme_name]
        })
    else:
        # Get current scheme from session or default
        scheme_name = session.get('color_scheme', 'default')
        return jsonify({
            'status': 'success',
            'scheme': DEFAULT_SCHEMES[scheme_name]
        })

# Add a new route for the color scheme home page
@color_scheme_bp.route('/color-schemes')
def color_scheme_home():
    """Color scheme management page"""
    # Get available schemes
    schemes = list(DEFAULT_SCHEMES.keys())
    
    # Get current scheme name from session or default
    current_scheme_name = session.get('color_scheme', 'default')
    
    # Get the current scheme data
    current_scheme = DEFAULT_SCHEMES.get(current_scheme_name, DEFAULT_SCHEMES['default'])
    
    return render_template('color_schemes.html', 
                          schemes=schemes, 
                          current_scheme_name=current_scheme_name,
                          current_scheme=current_scheme)
