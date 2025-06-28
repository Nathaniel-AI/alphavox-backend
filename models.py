from datetime import datetime
from app_init import db
import json

# Define SQLite/PostgreSQL models for the application

class User(db.Model):
    """User model for storing user details"""
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(64), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    interactions = db.relationship('UserInteraction', backref='user', lazy='dynamic')
    preferences = db.relationship('UserPreference', backref='user', lazy='dynamic')
    
    def __repr__(self):
        return f'<User {self.name}>'


class UserInteraction(db.Model):
    """Model for storing user interaction history"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    text = db.Column(db.String(512), nullable=True)
    intent = db.Column(db.String(64), nullable=True)
    confidence = db.Column(db.Float, nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<UserInteraction {self.intent} ({self.confidence:.2f})>'


class UserPreference(db.Model):
    """Model for storing user preferences for the adaptive profile system"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    type = db.Column(db.String(64), nullable=False)
    value = db.Column(db.String(256), nullable=False)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    source = db.Column(db.String(64), default='manual')  # 'manual', 'learned', 'system'
    confidence = db.Column(db.Float, default=1.0)  # Confidence in this preference
    
    __table_args__ = (
        db.UniqueConstraint('user_id', 'type', name='_user_preference_type_uc'),
    )
    
    def __repr__(self):
        return f'<UserPreference {self.type}: {self.value}>'
    
    @classmethod
    def get_user_profile(cls, user_id):
        """Get all active preferences for a user as a profile dictionary"""
        preferences = cls.query.filter_by(user_id=user_id, is_active=True).all()
        profile = {}
        
        for pref in preferences:
            # Convert values to appropriate types when possible
            try:
                if pref.value.lower() in ('true', 'false'):
                    # Boolean conversion
                    profile[pref.type] = pref.value.lower() == 'true'
                elif pref.value.replace('.', '', 1).isdigit():
                    # Numeric conversion (float)
                    profile[pref.type] = float(pref.value)
                else:
                    # String value
                    profile[pref.type] = pref.value
            except:
                # If any conversion fails, use the raw value
                profile[pref.type] = pref.value
                
        return profile
    
    @classmethod
    def set_preference(cls, user_id, pref_type, value, source='manual'):
        """Set or update a user preference"""
        # Find existing preference
        pref = cls.query.filter_by(user_id=user_id, type=pref_type).first()
        
        if pref:
            # Update existing
            pref.value = str(value)
            pref.source = source
            pref.is_active = True
            pref.updated_at = datetime.utcnow()
        else:
            # Create new
            pref = cls(
                user_id=user_id,
                type=pref_type,
                value=str(value),
                source=source
            )
            db.session.add(pref)
            
        db.session.commit()
        return pref


class CaregiverNote(db.Model):
    """Model for storing caregiver notes and observations"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    author = db.Column(db.String(128), nullable=False)
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    tags = db.Column(db.String(256), nullable=True)  # Comma-separated tags
    
    user = db.relationship('User', backref=db.backref('caregiver_notes', lazy='dynamic'))
    
    def __repr__(self):
        return f'<CaregiverNote {self.id}: {self.content[:20]}...>'
    
    def get_tags_list(self):
        """Return tags as a list"""
        if not self.tags:
            return []
        return [tag.strip() for tag in self.tags.split(',')]
    
    @classmethod
    def add_note(cls, user_id, author, content, tags=None):
        """Add a new caregiver note"""
        tags_str = ','.join(tags) if tags else None
        
        note = cls(
            user_id=user_id,
            author=author,
            content=content,
            tags=tags_str
        )
        
        db.session.add(note)
        db.session.commit()
        return note


class CommunicationProfile(db.Model):
    """Model for storing user's communication profile"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    primary_mode = db.Column(db.String(64), nullable=False)  # 'gesture', 'symbol', 'text', etc.
    secondary_mode = db.Column(db.String(64), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    user = db.relationship('User', backref=db.backref('communication_profiles', lazy='dynamic'))
    
    def __repr__(self):
        return f'<CommunicationProfile {self.primary_mode}/{self.secondary_mode}>'
    
    @classmethod
    def get_latest_profile(cls, user_id):
        """Get the most recent communication profile for a user"""
        return cls.query.filter_by(user_id=user_id).order_by(cls.updated_at.desc()).first()
    
    @classmethod
    def update_profile(cls, user_id, primary_mode, secondary_mode=None):
        """Update a user's communication profile"""
        profile = cls.get_latest_profile(user_id)
        
        if profile:
            # Update existing profile
            profile.primary_mode = primary_mode
            profile.secondary_mode = secondary_mode
            profile.updated_at = datetime.utcnow()
        else:
            # Create new profile
            profile = cls(
                user_id=user_id,
                primary_mode=primary_mode,
                secondary_mode=secondary_mode
            )
            db.session.add(profile)
            
        db.session.commit()
        return profile


class SystemSuggestion(db.Model):
    """Model for storing AI system suggestions for users"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title = db.Column(db.String(128), nullable=False)
    description = db.Column(db.Text, nullable=False)
    suggestion_type = db.Column(db.String(64), nullable=False)  # 'feature', 'setting', 'communication'
    confidence = db.Column(db.Float, nullable=False)
    is_active = db.Column(db.Boolean, default=True)
    is_accepted = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    user = db.relationship('User', backref=db.backref('system_suggestions', lazy='dynamic'))
    
    def __repr__(self):
        return f'<SystemSuggestion {self.id}: {self.title}>'
    
    @classmethod
    def get_active_suggestions(cls, user_id):
        """Get all active suggestions for a user"""
        return cls.query.filter_by(user_id=user_id, is_active=True).order_by(cls.confidence.desc()).all()
    
    @classmethod
    def add_suggestion(cls, user_id, title, description, suggestion_type, confidence):
        """Add a new system suggestion"""
        suggestion = cls(
            user_id=user_id,
            title=title,
            description=description,
            suggestion_type=suggestion_type,
            confidence=confidence
        )
        
        db.session.add(suggestion)
        db.session.commit()
        return suggestion
    
    @classmethod
    def accept_suggestion(cls, suggestion_id):
        """Mark a suggestion as accepted"""
        suggestion = cls.query.get(suggestion_id)
        if suggestion:
            suggestion.is_accepted = True
            db.session.commit()
        return suggestion
    
    @classmethod
    def dismiss_suggestion(cls, suggestion_id):
        """Mark a suggestion as inactive"""
        suggestion = cls.query.get(suggestion_id)
        if suggestion:
            suggestion.is_active = False
            db.session.commit()
        return suggestion


class LearningMilestone(db.Model):
    """Model for tracking learning milestones"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title = db.Column(db.String(128), nullable=False)
    description = db.Column(db.Text, nullable=False)
    criteria_type = db.Column(db.String(64), nullable=False)  # 'interaction_count', 'vocabulary_size', etc.
    target_value = db.Column(db.Integer, nullable=False)
    current_value = db.Column(db.Integer, default=0)
    is_reached = db.Column(db.Boolean, default=False)
    reached_date = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)
    
    user = db.relationship('User', backref=db.backref('learning_milestones', lazy='dynamic'))
    
    def __repr__(self):
        return f'<LearningMilestone {self.id}: {self.title}>'
    
    @property
    def progress(self):
        """Calculate progress as a percentage"""
        if self.target_value <= 0:
            return 100 if self.is_reached else 0
        
        progress = min(100, int((self.current_value / self.target_value) * 100))
        return progress
    
    @classmethod
    def get_active_milestones(cls, user_id):
        """Get all active milestones for a user"""
        return cls.query.filter_by(user_id=user_id, is_active=True).all()
    
    @classmethod
    def add_milestone(cls, user_id, title, description, criteria_type, target_value):
        """Add a new learning milestone"""
        milestone = cls(
            user_id=user_id,
            title=title,
            description=description,
            criteria_type=criteria_type,
            target_value=target_value
        )
        
        db.session.add(milestone)
        db.session.commit()
        return milestone
    
    @classmethod
    def update_progress(cls, user_id, criteria_type, value=1):
        """Update progress for all milestones of a given criteria type"""
        milestones = cls.query.filter_by(
            user_id=user_id, 
            criteria_type=criteria_type, 
            is_active=True,
            is_reached=False
        ).all()
        
        updated = []
        for milestone in milestones:
            milestone.current_value += value
            if milestone.current_value >= milestone.target_value and not milestone.is_reached:
                milestone.is_reached = True
                milestone.reached_date = datetime.utcnow()
                updated.append(milestone)
                
        if updated:
            db.session.commit()
            
        return updated


class LearningTemplate(db.Model):
    """Model for storing learning templates that customize the adaptive learning system"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    name = db.Column(db.String(128), nullable=False)
    description = db.Column(db.Text, nullable=False)
    focus_area = db.Column(db.String(64), nullable=False)  # 'symbols', 'gestures', 'intent', etc.
    parameters = db.Column(db.Text, nullable=False)  # JSON string of parameters
    is_active = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    user = db.relationship('User', backref=db.backref('learning_templates', lazy='dynamic'))
    
    def __repr__(self):
        return f'<LearningTemplate {self.id}: {self.name}>'
    
    @property
    def params(self):
        """Get parameters as a dictionary"""
        try:
            return json.loads(self.parameters)
        except:
            return {}
    
    @classmethod
    def get_active_template(cls, user_id):
        """Get the active learning template for a user"""
        return cls.query.filter_by(user_id=user_id, is_active=True).first()
    
    @classmethod
    def create_template(cls, user_id, name, description, focus_area, parameters, activate=False):
        """Create a new learning template"""
        # Deactivate other templates if needed
        if activate:
            active_templates = cls.query.filter_by(user_id=user_id, is_active=True).all()
            for template in active_templates:
                template.is_active = False
        
        # Create new template
        parameters_json = json.dumps(parameters) if isinstance(parameters, dict) else parameters
        template = cls(
            user_id=user_id,
            name=name,
            description=description,
            focus_area=focus_area,
            parameters=parameters_json,
            is_active=activate
        )
        
        db.session.add(template)
        db.session.commit()
        return template
    
    @classmethod
    def activate_template(cls, template_id):
        """Activate a specific template and deactivate others"""
        template = cls.query.get(template_id)
        if not template:
            return None
            
        # Deactivate other templates
        active_templates = cls.query.filter_by(user_id=template.user_id, is_active=True).all()
        for t in active_templates:
            t.is_active = False
            
        # Activate the requested template
        template.is_active = True
        db.session.commit()
        return template
    
    @classmethod
    def deactivate_template(cls, template_id):
        """Deactivate a specific template"""
        template = cls.query.get(template_id)
        if template:
            template.is_active = False
            db.session.commit()
        return template


class LearningSession(db.Model):
    """Model for tracking learning sessions and their outcomes"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    start_time = db.Column(db.DateTime, default=datetime.utcnow)
    end_time = db.Column(db.DateTime, nullable=True)
    is_active = db.Column(db.Boolean, default=True)
    interactions_count = db.Column(db.Integer, default=0)
    success_rate = db.Column(db.Float, nullable=True)
    template_id = db.Column(db.Integer, db.ForeignKey('learning_template.id'), nullable=True)
    notes = db.Column(db.Text, nullable=True)
    
    user = db.relationship('User', backref=db.backref('learning_sessions', lazy='dynamic'))
    template = db.relationship('LearningTemplate', backref=db.backref('sessions', lazy='dynamic'))
    
    def __repr__(self):
        duration = (self.end_time or datetime.utcnow()) - self.start_time
        minutes = round(duration.total_seconds() / 60)
        return f'<LearningSession {self.id}: {minutes} minutes, {self.interactions_count} interactions>'
    
    @classmethod
    def start_session(cls, user_id, template_id=None):
        """Start a new learning session"""
        # End any active sessions
        active_sessions = cls.query.filter_by(user_id=user_id, is_active=True).all()
        for session in active_sessions:
            session.end_time = datetime.utcnow()
            session.is_active = False
            
        # Create new session
        session = cls(
            user_id=user_id,
            template_id=template_id
        )
        
        db.session.add(session)
        db.session.commit()
        return session
    
    @classmethod
    def end_session(cls, session_id, success_rate=None, notes=None):
        """End an active learning session"""
        session = cls.query.get(session_id)
        if session and session.is_active:
            session.end_time = datetime.utcnow()
            session.is_active = False
            session.success_rate = success_rate
            session.notes = notes
            db.session.commit()
        return session
    
    @classmethod
    def record_interaction(cls, user_id, success=None):
        """Record an interaction in the active session"""
        session = cls.query.filter_by(user_id=user_id, is_active=True).first()
        if session:
            session.interactions_count += 1
            db.session.commit()
        return session
    
    @classmethod
    def get_recent_sessions(cls, user_id, limit=10):
        """Get recent learning sessions for a user"""
        return cls.query.filter_by(user_id=user_id).order_by(cls.start_time.desc()).limit(limit).all()


class SkillLevel(db.Model):
    """Model for tracking skill levels in different areas"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    skill_type = db.Column(db.String(64), nullable=False)  # 'symbol', 'gesture', 'intent', etc.
    level = db.Column(db.Integer, default=0)  # Scale of 0-100
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    user = db.relationship('User', backref=db.backref('skill_levels', lazy='dynamic'))
    
    __table_args__ = (
        db.UniqueConstraint('user_id', 'skill_type', name='_user_skill_type_uc'),
    )
    
    def __repr__(self):
        return f'<SkillLevel {self.skill_type}: {self.level}>'
    
    @classmethod
    def get_skill_levels(cls, user_id):
        """Get all skill levels for a user"""
        skills = cls.query.filter_by(user_id=user_id).all()
        result = {
            'symbol': 0,
            'gesture': 0,
            'intent': 0,
            'emotion': 0,
            'context': 0
        }
        
        for skill in skills:
            result[skill.skill_type] = skill.level
            
        return result
    
    @classmethod
    def update_skill(cls, user_id, skill_type, change=1):
        """Update a skill level"""
        skill = cls.query.filter_by(user_id=user_id, skill_type=skill_type).first()
        
        if skill:
            # Update existing skill
            skill.level = min(100, max(0, skill.level + change))
        else:
            # Create new skill
            skill = cls(
                user_id=user_id,
                skill_type=skill_type,
                level=min(100, max(0, change))
            )
            db.session.add(skill)
            
        db.session.commit()
        return skill


class RecognitionFeedback(db.Model):
    """Model for storing feedback on recognition accuracy"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    interaction_id = db.Column(db.Integer, db.ForeignKey('user_interaction.id'), nullable=False)
    feedback_type = db.Column(db.String(32), nullable=False)  # 'correct', 'partially', 'incorrect'
    correct_intent = db.Column(db.String(64), nullable=True)
    notes = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    user = db.relationship('User', backref=db.backref('recognition_feedback', lazy='dynamic'))
    interaction = db.relationship('UserInteraction', backref=db.backref('feedback', lazy='dynamic'))
    
    def __repr__(self):
        return f'<RecognitionFeedback {self.id}: {self.feedback_type}>'
    
    @classmethod
    def add_feedback(cls, user_id, interaction_id, feedback_type, correct_intent=None, notes=None):
        """Add feedback for a recognition"""
        feedback = cls(
            user_id=user_id,
            interaction_id=interaction_id,
            feedback_type=feedback_type,
            correct_intent=correct_intent,
            notes=notes
        )
        
        db.session.add(feedback)
        db.session.commit()
        return feedback
    
    @classmethod
    def get_feedback_for_interaction(cls, interaction_id):
        """Get feedback for an interaction"""
        return cls.query.filter_by(interaction_id=interaction_id).first()