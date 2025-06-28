"""
Routes registration for AlphaVox

This file imports all necessary routes from their respective modules
and registers them with the Flask application.
"""

# Import necessary modules and blueprints
from routes.learning_routes import register_learning_routes
from routes.health_routes import register_health_routes

# Optional imports that might not be available yet
try:
    from routes.adaptive_conversation_routes import adaptive_bp, register_adaptive_routes
    HAS_ADAPTIVE_ROUTES = True
except ImportError:
    HAS_ADAPTIVE_ROUTES = False

# Direct import from the new color scheme module
from color_scheme_module import color_scheme_bp
HAS_COLOR_SCHEME_ROUTES = True

from app_init import app

# Direct registration of blueprints for main.py imports
# These exports allow main.py to import * from app_routes
# without needing to know the specific routes files

# Register the color scheme blueprint if available
if HAS_COLOR_SCHEME_ROUTES:
    app.register_blueprint(color_scheme_bp, name='unique_color_scheme')

# Register adaptive conversation routes if available
if HAS_ADAPTIVE_ROUTES:
    # register_adaptive_routes function will register the blueprint internally
    register_adaptive_routes(app)

# Register the learning routes
register_learning_routes(app)

# Register health check routes for AWS and monitoring
register_health_routes(app)