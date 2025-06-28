"""
AlphaVox - Caregiver Dashboard Module
-------------------------------------
This module provides integration for the caregiver interface, allowing caregivers
to monitor user progress, customize settings, and review communication history.
"""

import logging
import os
import json
from typing import Dict, Any, List, Optional, Tuple
from flask import render_template, jsonify, request, session

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Singleton instance
_caregiver_dashboard = None

class CaregiverDashboard:
    """
    Provides caregiver dashboard functionality for AlphaVox.
    
    This class:
    - Manages caregiver access to user data
    - Provides progress monitoring and reporting
    - Allows customization of user settings
    - Enables communication history review
    """
    
    def __init__(self):
        """Initialize the caregiver dashboard"""
        self.caregiver_interface = None
        
        # Initialize components
        self._initialize_components()
        
        logger.info("Caregiver Dashboard module initialized")
    
    def _initialize_components(self):
        """Initialize all required components"""
        # Try to import and initialize the caregiver interface
        try:
            from attached_assets.caregiver_interface import CaregiverInterface
            self.caregiver_interface = CaregiverInterface()
            logger.info("Caregiver Interface loaded successfully")
        except ImportError as e:
            logger.warning(f"Caregiver Interface import failed: {e}")
            self.caregiver_interface = None
    
    def get_user_profiles(self) -> List[Dict[str, Any]]:
        """
        Get all user profiles.
        
        Returns:
            List of user profile dictionaries
        """
        if self.caregiver_interface:
            return self.caregiver_interface.get_all_user_profiles()
        return []
    
    def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """
        Get a specific user profile.
        
        Args:
            user_id: User ID to retrieve
            
        Returns:
            User profile dictionary
        """
        if self.caregiver_interface:
            return self.caregiver_interface.get_user_profile(user_id)
        return {}
    
    def update_user_profile(self, user_id: str, profile_data: Dict[str, Any]) -> bool:
        """
        Update a user profile.
        
        Args:
            user_id: User ID to update
            profile_data: Profile data to update
            
        Returns:
            True if successful
        """
        if self.caregiver_interface:
            return self.caregiver_interface.update_user_profile(user_id, profile_data)
        return False
    
    def create_user_profile(self, profile_data: Dict[str, Any]) -> Optional[str]:
        """
        Create a new user profile.
        
        Args:
            profile_data: Profile data for the new user
            
        Returns:
            New user ID if successful, None otherwise
        """
        if self.caregiver_interface:
            return self.caregiver_interface.create_user_profile(profile_data)
        return None
    
    def get_communication_history(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get communication history for a user.
        
        Args:
            user_id: User ID to retrieve history for
            limit: Maximum number of items to return
            
        Returns:
            List of communication history items
        """
        if self.caregiver_interface:
            return self.caregiver_interface.get_user_communication_history(user_id, limit)
        return []
    
    def add_communication_entry(self, user_id: str, entry: Dict[str, Any]) -> bool:
        """
        Add a communication entry to a user's history.
        
        Args:
            user_id: User ID to add entry for
            entry: Communication entry to add
            
        Returns:
            True if successful
        """
        if self.caregiver_interface:
            return self.caregiver_interface.add_communication_entry(user_id, entry)
        return False
    
    def get_progress_metrics(self, user_id: str) -> Dict[str, Any]:
        """
        Get progress metrics for a user.
        
        Args:
            user_id: User ID to retrieve metrics for
            
        Returns:
            Dictionary of progress metrics
        """
        if self.caregiver_interface:
            return self.caregiver_interface.get_progress_metrics(user_id)
        return {}
    
    def get_educational_resources(self, topic: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get educational resources for caregivers.
        
        Args:
            topic: Optional topic to filter resources by
            
        Returns:
            List of educational resources
        """
        if self.caregiver_interface:
            return self.caregiver_interface.get_educational_resources(topic)
        return []

def get_caregiver_dashboard():
    """Get or create the caregiver dashboard singleton"""
    global _caregiver_dashboard
    if _caregiver_dashboard is None:
        _caregiver_dashboard = CaregiverDashboard()
    return _caregiver_dashboard

# Flask routes for caregiver dashboard
def register_caregiver_routes(app):
    """
    Register Flask routes for caregiver dashboard.
    
    Args:
        app: Flask application to register routes with
    """
    @app.route('/caregiver')
    def caregiver_home():
        """Caregiver dashboard home page"""
        dashboard = get_caregiver_dashboard()
        user_profiles = dashboard.get_user_profiles()
        return render_template('caregiver/dashboard.html', user_profiles=user_profiles)
    
    @app.route('/caregiver/user/<user_id>')
    def caregiver_user_profile(user_id):
        """Caregiver user profile page"""
        dashboard = get_caregiver_dashboard()
        user_profile = dashboard.get_user_profile(user_id)
        
        if not user_profile:
            return render_template('caregiver/error.html', message="User not found")
        
        history = dashboard.get_communication_history(user_id)
        metrics = dashboard.get_progress_metrics(user_id)
        
        return render_template('caregiver/user_profile.html', 
                              user_profile=user_profile,
                              history=history,
                              metrics=metrics)
    
    @app.route('/caregiver/resources')
    def caregiver_resources():
        """Caregiver educational resources page"""
        dashboard = get_caregiver_dashboard()
        topic = request.args.get('topic')
        resources = dashboard.get_educational_resources(topic)
        return render_template('caregiver/resources.html', resources=resources)
    
    @app.route('/api/caregiver/user_profiles', methods=['GET'])
    def api_get_user_profiles():
        """API endpoint to get all user profiles"""
        dashboard = get_caregiver_dashboard()
        return jsonify(dashboard.get_user_profiles())
    
    @app.route('/api/caregiver/user_profile/<user_id>', methods=['GET'])
    def api_get_user_profile(user_id):
        """API endpoint to get a specific user profile"""
        dashboard = get_caregiver_dashboard()
        return jsonify(dashboard.get_user_profile(user_id))
    
    @app.route('/api/caregiver/user_profile/<user_id>', methods=['PUT'])
    def api_update_user_profile(user_id):
        """API endpoint to update a user profile"""
        dashboard = get_caregiver_dashboard()
        profile_data = request.json
        success = dashboard.update_user_profile(user_id, profile_data)
        return jsonify({"success": success})
    
    @app.route('/api/caregiver/user_profile', methods=['POST'])
    def api_create_user_profile():
        """API endpoint to create a new user profile"""
        dashboard = get_caregiver_dashboard()
        profile_data = request.json
        user_id = dashboard.create_user_profile(profile_data)
        if user_id:
            return jsonify({"success": True, "user_id": user_id})
        return jsonify({"success": False, "error": "Failed to create user profile"})
    
    @app.route('/api/caregiver/communication_history/<user_id>', methods=['GET'])
    def api_get_communication_history(user_id):
        """API endpoint to get communication history for a user"""
        dashboard = get_caregiver_dashboard()
        limit = request.args.get('limit', 50, type=int)
        return jsonify(dashboard.get_communication_history(user_id, limit))
    
    @app.route('/api/caregiver/progress_metrics/<user_id>', methods=['GET'])
    def api_get_progress_metrics(user_id):
        """API endpoint to get progress metrics for a user"""
        dashboard = get_caregiver_dashboard()
        return jsonify(dashboard.get_progress_metrics(user_id))
    
    @app.route('/api/caregiver/educational_resources', methods=['GET'])
    def api_get_educational_resources():
        """API endpoint to get educational resources"""
        dashboard = get_caregiver_dashboard()
        topic = request.args.get('topic')
        return jsonify(dashboard.get_educational_resources(topic))