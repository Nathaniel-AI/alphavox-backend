"""
Color scheme routes for AlphaVox

This module provides routes related to color scheme customization for the AlphaVox application.
"""

import logging
import os
import json
import time
from flask import Blueprint, jsonify, request, render_template, session

logger = logging.getLogger(__name__)

# Default color schemes
DEFAULT_SCHEMES = {
    "default": {
        "primary": "#3498db",
        "secondary": "#2ecc71",
        "accent": "#e74c3c",
        "background": "#f5f5f5",
        "text": "#333333"
    },
    "dark": {
        "primary": "#3498db",
        "secondary": "#2ecc71",
        "accent": "#e74c3c",
        "background": "#2c3e50",
        "text": "#ecf0f1"
    },
    "high_contrast": {
        "primary": "#0066cc",
        "secondary": "#00cc66",
        "accent": "#cc0000",
        "background": "#ffffff",
        "text": "#000000"
    },
    "pastel": {
        "primary": "#a6dcef",
        "secondary": "#c7f0db",
        "accent": "#ff9a9e",
        "background": "#f9f7e8",
        "text": "#5e5e5e"
    },
    "calm": {
        "primary": "#7fb3d5",
        "secondary": "#a0d6b4",
        "accent": "#f1948a",
        "background": "#f5f5f5",
        "text": "#333333"
    }
}

# Path to store custom color schemes
SCHEMES_PATH = os.path.join(os.getcwd(), 'data', 'color_schemes.json')

# Make sure the data directory exists
os.makedirs(os.path.dirname(SCHEMES_PATH), exist_ok=True)

def get_all_schemes():
    """Get all available color schemes."""
    # Start with default schemes
    schemes = DEFAULT_SCHEMES.copy()
    
    # Add custom schemes if they exist
    if os.path.exists(SCHEMES_PATH):
        try:
            with open(SCHEMES_PATH, 'r') as f:
                custom_schemes = json.load(f)
                schemes.update(custom_schemes)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error loading custom color schemes: {e}")
    
    return schemes

def get_current_scheme(user_id=None):
    """
    Get the current color scheme for a user.
    This function is made available to templates.
    """
    # Use the session user_id if no user_id is provided
    if user_id is None:
        user_id = session.get('user_id', 'default_user')
    
    # Get all schemes
    schemes = get_all_schemes()
    
    # Default scheme key
    scheme_key = 'default'
    
    # Try to get user's preferred scheme from the database or session
    try:
        # First check session for scheme preference
        if 'color_scheme' in session:
            scheme_key = session['color_scheme']
        else:
            # Then check database
            from models import UserPreference
            preference = UserPreference.query.filter_by(
                user_id=user_id,
                type='color_scheme',
                is_active=True
            ).first()
            
            if preference:
                scheme_key = preference.value
                # Cache in session
                session['color_scheme'] = scheme_key
    except Exception as e:
        logger.error(f"Error getting user color scheme preference: {e}")
    
    # Return the scheme or default as fallback
    return schemes.get(scheme_key, schemes['default'])

# Create blueprint for color scheme routes
color_scheme_bp = Blueprint('color_scheme', __name__, url_prefix='/color-scheme')

@color_scheme_bp.route('/')
def color_scheme_home():
    """Color scheme customization page."""
    return render_template('color_scheme/home.html', title="Color Scheme")

@color_scheme_bp.route('/generate', methods=['POST'])
def generate_color_scheme():
    """Generate a new color scheme based on parameters."""
    emotion = request.form.get('emotion', '')
    base_color = request.form.get('base_color', '#3498db')
    
    # Use the base color for generating a color scheme
    # In a real implementation, we would use a color theory algorithm
    # and potentially AI to generate a harmonious color scheme
    
    # For demonstration purposes, we're using predefined schemes
    # based on emotion or variations of the base color
    
    schemes = {
        'calm': {
            'primary': '#5c9ead',
            'secondary': '#326771',
            'accent': '#e1b16a',
            'background': '#f8f9fa',
            'text': '#2c3e50'
        },
        'energetic': {
            'primary': '#e74c3c',
            'secondary': '#f39c12',
            'accent': '#3498db',
            'background': '#f8f9fa',
            'text': '#2c3e50'
        },
        'focused': {
            'primary': '#3498db',
            'secondary': '#2c3e50',
            'accent': '#e74c3c',
            'background': '#f5f7fa',
            'text': '#34495e'
        },
        'playful': {
            'primary': '#9b59b6',
            'secondary': '#3498db',
            'accent': '#f1c40f',
            'background': '#f9f9f9',
            'text': '#2c3e50'
        },
        'serene': {
            'primary': '#27ae60',
            'secondary': '#2980b9',
            'accent': '#f39c12',
            'background': '#f5f5f5',
            'text': '#2c3e50'
        },
        'professional': {
            'primary': '#2c3e50',
            'secondary': '#3498db',
            'accent': '#e67e22',
            'background': '#f5f5f5',
            'text': '#2c3e50'
        }
    }
    
    # Use the selected emotion scheme or derive from base color
    if emotion and emotion in schemes:
        scheme = schemes[emotion]
    else:
        # Derive a scheme from the base color
        # This is a simplified approach; a real implementation would use
        # color theory to generate complementary colors
        
        # Remove the # from the hex color and convert to RGB
        base_rgb = tuple(int(base_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        
        # Generate a slightly darker color for secondary
        secondary_rgb = tuple(max(0, c - 30) for c in base_rgb)
        secondary = '#%02x%02x%02x' % secondary_rgb
        
        # Generate a complementary color for accent
        # Simple complementary: invert each channel
        accent_rgb = tuple(255 - c for c in base_rgb)
        accent = '#%02x%02x%02x' % accent_rgb
        
        scheme = {
            'primary': base_color,
            'secondary': secondary,
            'accent': accent,
            'background': '#f5f5f5',
            'text': '#2c3e50'
        }
    
    return jsonify({
        'success': True,
        'scheme': scheme
    })

@color_scheme_bp.route('/apply', methods=['POST'])
def apply_color_scheme():
    """Apply a predefined color scheme."""
    scheme_key = request.form.get('scheme', 'default')
    user_id = session.get('user_id', 'default_user')
    
    # Update user preference in the database
    try:
        from models import UserPreference
        UserPreference.set_preference(user_id, 'color_scheme', scheme_key, 'manual')
        
        # Also update the session
        session['color_scheme'] = scheme_key
        
        return jsonify({
            'success': True,
            'message': 'Color scheme applied successfully'
        })
    except Exception as e:
        logger.error(f"Error applying color scheme: {e}")
        return jsonify({
            'success': False,
            'message': 'Error applying color scheme'
        }), 500

@color_scheme_bp.route('/apply-custom', methods=['POST'])
def apply_custom_scheme():
    """Apply a custom generated color scheme."""
    user_id = session.get('user_id', 'default_user')
    
    try:
        # Get the color values
        colors = request.form.getlist('colors[]')
        
        if len(colors) < 3:
            return jsonify({
                'success': False,
                'message': 'Not enough colors provided'
            }), 400
        
        # Create a custom scheme
        custom_scheme = {
            'primary': colors[0],
            'secondary': colors[1],
            'accent': colors[2],
            'background': colors[3] if len(colors) > 3 else '#f5f5f5',
            'text': colors[4] if len(colors) > 4 else '#2c3e50'
        }
        
        # Save it to the database
        custom_scheme_key = f"custom_{int(time.time())}"
        
        # Save to file
        if os.path.exists(SCHEMES_PATH):
            try:
                with open(SCHEMES_PATH, 'r') as f:
                    schemes = json.load(f)
            except json.JSONDecodeError:
                schemes = {}
        else:
            schemes = {}
        
        schemes[custom_scheme_key] = custom_scheme
        
        with open(SCHEMES_PATH, 'w') as f:
            json.dump(schemes, f)
        
        # Update user preference
        from models import UserPreference
        UserPreference.set_preference(user_id, 'color_scheme', custom_scheme_key, 'manual')
        
        # Also update the session
        session['color_scheme'] = custom_scheme_key
        
        return jsonify({
            'success': True,
            'message': 'Custom color scheme applied successfully'
        })
    except Exception as e:
        logger.error(f"Error applying custom color scheme: {e}")
        return jsonify({
            'success': False,
            'message': 'Error applying custom color scheme'
        }), 500

@color_scheme_bp.route('/save-settings', methods=['POST'])
def save_accessibility_settings():
    """Save accessibility settings."""
    user_id = session.get('user_id', 'default_user')
    
    try:
        # Get settings from form
        reduce_motion = request.form.get('reduce_motion') == 'true'
        high_contrast = request.form.get('high_contrast') == 'true'
        enlarged_text = request.form.get('enlarged_text') == 'true'
        
        # Update user preferences
        from models import UserPreference
        UserPreference.set_preference(user_id, 'reduce_motion', reduce_motion, 'manual')
        UserPreference.set_preference(user_id, 'high_contrast', high_contrast, 'manual')
        UserPreference.set_preference(user_id, 'enlarged_text', enlarged_text, 'manual')
        
        return jsonify({
            'success': True,
            'message': 'Accessibility settings saved successfully'
        })
    except Exception as e:
        logger.error(f"Error saving accessibility settings: {e}")
        return jsonify({
            'success': False,
            'message': 'Error saving accessibility settings'
        }), 500