"""
AI-powered Color Scheme Generator for AlphaVox

This module generates personalized color schemes based on user preferences
and accessibility needs using AI techniques.
"""

import os
import json
import logging
import random
import colorsys
from typing import Dict, List, Tuple, Optional, Any
import math

# Global singleton instance
_color_scheme_generator = None

# Check if OpenAI API key is available
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if OPENAI_API_KEY:
    from openai import OpenAI
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    AI_COLOR_GENERATION_AVAILABLE = True
else:
    AI_COLOR_GENERATION_AVAILABLE = False
    logging.warning("OpenAI API key not found. AI color generation will not be available.")

# Color scheme profiles storage location
COLOR_PROFILES_PATH = "profiles/color_schemes.json"
os.makedirs(os.path.dirname(COLOR_PROFILES_PATH), exist_ok=True)

# Predefined accessible color schemes
PREDEFINED_SCHEMES = {
    "default": {
        "name": "Cyber Midnight",
        "description": "Default dark theme with cyan accents",
        "primary": "#00b4d8",  # Bright cyan
        "secondary": "#0077b6",  # Darker blue
        "background": "#121212",  # Very dark gray
        "surface": "#1e1e1e",  # Dark gray
        "text": "#ffffff",  # White
        "accent": "#90e0ef"  # Light cyan
    },
    "high_contrast": {
        "name": "High Contrast",
        "description": "Maximum contrast for visibility",
        "primary": "#ffff00",  # Yellow
        "secondary": "#00ffff",  # Cyan
        "background": "#000000",  # Black
        "surface": "#0a0a0a",  # Very dark gray
        "text": "#ffffff",  # White
        "accent": "#ff00ff"  # Magenta
    },
    "focus": {
        "name": "Focus Mode",
        "description": "Reduced colors for minimal distraction",
        "primary": "#4a5568",  # Slate gray
        "secondary": "#2d3748",  # Darker slate
        "background": "#1a202c",  # Very dark blue-gray
        "surface": "#2d3748",  # Dark slate
        "text": "#e2e8f0",  # Light gray
        "accent": "#718096"  # Medium gray
    },
    "calm": {
        "name": "Calm Blue",
        "description": "Calming blue tones for reduced anxiety",
        "primary": "#4299e1",  # Bright blue
        "secondary": "#2b6cb0",  # Darker blue
        "background": "#1a365d",  # Very dark blue
        "surface": "#2a4365",  # Dark blue
        "text": "#ebf8ff",  # Very light blue
        "accent": "#63b3ed"  # Light blue
    },
    "warm": {
        "name": "Warm Comfort",
        "description": "Warm colors for comfortable viewing",
        "primary": "#ed8936",  # Orange
        "secondary": "#c05621",  # Dark orange
        "background": "#2d2d2d",  # Dark gray
        "surface": "#3d3d3d",  # Medium-dark gray
        "text": "#fffaf0",  # Off-white
        "accent": "#f6ad55"  # Light orange
    }
}

class ColorSchemeGenerator:
    """
    Generates personalized color schemes based on user preferences
    and accessibility requirements.
    """
    
    def __init__(self):
        """Initialize the color scheme generator."""
        self._load_color_profiles()
    
    def _load_color_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Load color profiles from the JSON file."""
        try:
            if os.path.exists(COLOR_PROFILES_PATH):
                with open(COLOR_PROFILES_PATH, 'r') as f:
                    self.color_profiles = json.load(f)
            else:
                self.color_profiles = {}
                self._save_color_profiles()
            return self.color_profiles
        except Exception as e:
            logging.error(f"Error loading color profiles: {str(e)}")
            self.color_profiles = {}
            return self.color_profiles
    
    def _save_color_profiles(self):
        """Save color profiles to the JSON file."""
        try:
            with open(COLOR_PROFILES_PATH, 'w') as f:
                json.dump(self.color_profiles, f, indent=2)
        except Exception as e:
            logging.error(f"Error saving color profiles: {str(e)}")
    
    def get_user_color_scheme(self, user_id: str) -> Dict[str, str]:
        """
        Get a user's current color scheme. If none exists, return default.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dict containing color scheme values
        """
        if user_id in self.color_profiles:
            return self.color_profiles[user_id]['scheme']
        else:
            return PREDEFINED_SCHEMES['default']
    
    def save_user_color_scheme(self, user_id: str, scheme: Dict[str, str], 
                              preferences: Optional[Dict[str, Any]] = None):
        """
        Save a user's color scheme and preferences.
        
        Args:
            user_id: User identifier
            scheme: Color scheme dictionary
            preferences: Optional user preferences that informed this scheme
        """
        if user_id not in self.color_profiles:
            self.color_profiles[user_id] = {
                'scheme': scheme,
                'preferences': preferences or {},
                'history': []
            }
        else:
            # Add current scheme to history before updating
            if 'scheme' in self.color_profiles[user_id]:
                if 'history' not in self.color_profiles[user_id]:
                    self.color_profiles[user_id]['history'] = []
                self.color_profiles[user_id]['history'].append(
                    self.color_profiles[user_id]['scheme']
                )
                # Keep only last 5 schemes in history
                self.color_profiles[user_id]['history'] = \
                    self.color_profiles[user_id]['history'][-5:]
            
        # Add RGB values to the scheme before saving
        scheme_with_rgb = self._add_rgb_values(scheme)
        
        # Update scheme and preferences
        self.color_profiles[user_id]['scheme'] = scheme_with_rgb
        if preferences:
            self.color_profiles[user_id]['preferences'] = preferences
        
        self._save_color_profiles()
    
    def generate_from_preferences(self, preferences: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate a color scheme based on user preferences.
        
        Args:
            preferences: Dictionary of user preferences
                - color_preference: Primary color preference
                - contrast: Preferred contrast level (low, medium, high)
                - brightness: Preferred brightness level (dark, medium, bright)
                - accessibility_needs: Specific accessibility needs
                - emotion: Emotional tone preference
                
        Returns:
            Dictionary containing the generated color scheme
        """
        if AI_COLOR_GENERATION_AVAILABLE:
            try:
                return self._generate_ai_color_scheme(preferences)
            except Exception as e:
                logging.error(f"Error generating AI color scheme: {str(e)}")
                return self._generate_algorithmic_color_scheme(preferences)
        else:
            return self._generate_algorithmic_color_scheme(preferences)
    
    def _generate_ai_color_scheme(self, preferences: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate a color scheme using AI (OpenAI).
        
        Args:
            preferences: Dictionary of user preferences
                
        Returns:
            Dictionary containing the generated color scheme
        """
        # Extract preferences
        color_preference = preferences.get('color_preference', 'blue')
        contrast = preferences.get('contrast', 'medium')
        brightness = preferences.get('brightness', 'dark')
        accessibility_needs = preferences.get('accessibility_needs', [])
        emotion = preferences.get('emotion', 'neutral')
        
        # Create prompt for OpenAI
        prompt = f"""
        Generate a color scheme for a user interface that meets these preferences:
        - Main color preference: {color_preference}
        - Contrast level: {contrast}
        - Brightness level: {brightness}
        - Accessibility needs: {', '.join(accessibility_needs) if accessibility_needs else 'None'}
        - Emotional tone: {emotion}
        
        This is for an assistive technology application for users with communication challenges.
        
        Respond with a JSON object containing these color values as hex codes:
        - name: A creative name for this color scheme
        - description: A brief description of the scheme
        - primary: Main brand color
        - secondary: Secondary brand color
        - background: Page background color
        - surface: Card and component background color
        - text: Main text color
        - accent: Accent color for highlights and emphasis
        
        For best readability, ensure text has at least 4.5:1 contrast ratio with backgrounds.
        If 'high contrast' is specified, ensure a minimum 7:1 contrast ratio.
        """
        
        # Call OpenAI API
        response = openai_client.chat.completions.create(
            model="gpt-4o",  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024. do not change this unless explicitly requested by the user
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        
        # Parse response
        try:
            result = json.loads(response.choices[0].message.content)
            # Ensure all required colors are present
            required_keys = ["name", "description", "primary", "secondary", 
                            "background", "surface", "text", "accent"]
            for key in required_keys:
                if key not in result:
                    if key in PREDEFINED_SCHEMES['default']:
                        result[key] = PREDEFINED_SCHEMES['default'][key]
            return result
        except Exception as e:
            logging.error(f"Error parsing AI color scheme response: {str(e)}")
            return self._generate_algorithmic_color_scheme(preferences)
    
    def _generate_algorithmic_color_scheme(self, preferences: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate a color scheme algorithmically when AI is not available.
        
        Args:
            preferences: Dictionary of user preferences
                
        Returns:
            Dictionary containing the generated color scheme
        """
        # Extract preferences
        color_preference = preferences.get('color_preference', 'blue')
        contrast = preferences.get('contrast', 'medium')
        brightness = preferences.get('brightness', 'dark')
        accessibility_needs = preferences.get('accessibility_needs', [])
        
        # Start with a base scheme based on overall preference
        if 'colorblind' in accessibility_needs:
            # Use colorblind-friendly base
            base_scheme = PREDEFINED_SCHEMES['high_contrast'].copy()
        elif contrast == 'high':
            # Use high contrast base
            base_scheme = PREDEFINED_SCHEMES['high_contrast'].copy()
        elif brightness == 'dark':
            # Use dark theme
            base_scheme = PREDEFINED_SCHEMES['default'].copy()
        else:
            # Use calm theme
            base_scheme = PREDEFINED_SCHEMES['calm'].copy()
        
        # Adjust the scheme based on color preference
        primary_color = self._get_color_from_preference(color_preference)
        base_scheme['primary'] = primary_color
        
        # Generate complementary colors
        h, s, v = self._hex_to_hsv(primary_color)
        
        # Secondary color (analogous or complementary based on preferences)
        h_secondary = (h + 0.08) % 1.0  # Slightly shifted hue
        secondary_color = self._hsv_to_hex(h_secondary, s, v)
        base_scheme['secondary'] = secondary_color
        
        # Accent color (complementary)
        h_accent = (h + 0.5) % 1.0  # Opposite hue
        accent_color = self._hsv_to_hex(h_accent, s, v)
        base_scheme['accent'] = accent_color
        
        # Adjust contrast based on preference
        if contrast == 'high':
            # Increase contrast
            base_scheme['text'] = '#ffffff'
            base_scheme['background'] = '#000000'
        elif contrast == 'low':
            # Reduce contrast
            base_scheme['text'] = '#e0e0e0'  # Light gray
            base_scheme['background'] = '#1a1a1a'  # Dark gray
        
        # Adjust brightness
        if brightness == 'bright':
            # Brighten background
            base_scheme['background'] = self._lighten_color(base_scheme['background'], 0.3)
            base_scheme['surface'] = self._lighten_color(base_scheme['surface'], 0.3)
        elif brightness == 'dark':
            # Darken background
            base_scheme['background'] = self._darken_color(base_scheme['background'], 0.1)
            base_scheme['surface'] = self._darken_color(base_scheme['surface'], 0.1)
        
        # Update name and description
        base_scheme['name'] = f"Custom {color_preference.title()}"
        base_scheme['description'] = f"Personalized {color_preference} theme with {contrast} contrast and {brightness} brightness"
        
        return base_scheme
    
    def _get_color_from_preference(self, color_preference: str) -> str:
        """Convert a color name preference to a hex color code."""
        color_map = {
            'red': '#e53e3e',
            'orange': '#dd6b20',
            'yellow': '#d69e2e',
            'green': '#38a169',
            'teal': '#319795',
            'blue': '#3182ce',
            'cyan': '#00b5d8',
            'purple': '#805ad5',
            'pink': '#d53f8c',
            'gray': '#718096'
        }
        
        return color_map.get(color_preference.lower(), '#3182ce')  # Default to blue
    
    def _hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color to RGB."""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def _rgb_to_hex(self, rgb: Tuple[int, int, int]) -> str:
        """Convert RGB to hex color."""
        return '#{:02x}{:02x}{:02x}'.format(*rgb)
    
    def _hex_to_hsv(self, hex_color: str) -> Tuple[float, float, float]:
        """Convert hex color to HSV."""
        r, g, b = self._hex_to_rgb(hex_color)
        return colorsys.rgb_to_hsv(r/255, g/255, b/255)
    
    def _hsv_to_hex(self, h: float, s: float, v: float) -> str:
        """Convert HSV to hex color."""
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return '#{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255))
    
    def _lighten_color(self, hex_color: str, amount: float) -> str:
        """Lighten a color by the given amount (0-1)."""
        r, g, b = self._hex_to_rgb(hex_color)
        r = min(255, int(r + (255 - r) * amount))
        g = min(255, int(g + (255 - g) * amount))
        b = min(255, int(b + (255 - b) * amount))
        return self._rgb_to_hex((r, g, b))
    
    def _darken_color(self, hex_color: str, amount: float) -> str:
        """Darken a color by the given amount (0-1)."""
        r, g, b = self._hex_to_rgb(hex_color)
        r = max(0, int(r * (1 - amount)))
        g = max(0, int(g * (1 - amount)))
        b = max(0, int(b * (1 - amount)))
        return self._rgb_to_hex((r, g, b))
    
    def get_predefined_schemes(self) -> Dict[str, Dict[str, str]]:
        """
        Get all predefined color schemes.
        
        Returns:
            Dictionary of predefined schemes
        """
        return PREDEFINED_SCHEMES
    
    def get_emotion_based_schemes(self) -> Dict[str, Dict[str, str]]:
        """
        Get emotion-based color schemes.
        
        Returns:
            Dictionary of emotion-based schemes
        """
        return {
            "calm": {
                "name": "Calm Serenity",
                "description": "Calming blue tones to reduce anxiety",
                "primary": "#4299e1",  # Bright blue
                "secondary": "#2b6cb0",  # Darker blue
                "background": "#1a365d",  # Very dark blue
                "surface": "#2a4365",  # Dark blue
                "text": "#ebf8ff",  # Very light blue
                "accent": "#63b3ed"  # Light blue
            },
            "energetic": {
                "name": "Energetic Spark",
                "description": "Vibrant colors for energy and motivation",
                "primary": "#ed8936",  # Orange
                "secondary": "#dd6b20",  # Dark orange
                "background": "#1a202c",  # Very dark blue-gray
                "surface": "#2d3748",  # Dark slate
                "text": "#ffffff",  # White
                "accent": "#f6ad55"  # Light orange
            },
            "focused": {
                "name": "Deep Focus",
                "description": "Minimal, low-distraction scheme for focus",
                "primary": "#667eea",  # Indigo
                "secondary": "#5a67d8",  # Deeper indigo
                "background": "#1a202c",  # Very dark blue-gray
                "surface": "#2d3748",  # Dark slate
                "text": "#e2e8f0",  # Light gray
                "accent": "#9f7aea"  # Purple
            },
            "positive": {
                "name": "Positive Growth",
                "description": "Uplifting green tones for positive energy",
                "primary": "#38a169",  # Green
                "secondary": "#2f855a",  # Darker green
                "background": "#1c2f2c",  # Very dark green-gray
                "surface": "#2c3c39",  # Dark green-gray
                "text": "#e6fffa",  # Very light cyan
                "accent": "#68d391"  # Light green
            },
            "cheerful": {
                "name": "Cheerful Sunshine",
                "description": "Bright, cheerful yellows for happy moods",
                "primary": "#ecc94b",  # Yellow
                "secondary": "#d69e2e",  # Gold
                "background": "#2d2c24",  # Dark yellow-gray
                "surface": "#3d3c34",  # Medium yellow-gray
                "text": "#fffff0",  # Ivory
                "accent": "#f6e05e"  # Light yellow
            }
        }
    
    def _add_rgb_values(self, scheme: Dict[str, str]) -> Dict[str, str]:
        """
        Add RGB value strings to a color scheme for CSS variables.
        
        Args:
            scheme: Dictionary with hex color values
            
        Returns:
            Dictionary with added RGB values for each color
        """
        # Make a copy to avoid modifying the original
        enhanced_scheme = scheme.copy()
        
        # Define color keys that need RGB values
        color_keys = ['primary', 'secondary', 'background', 'surface', 'text', 'accent']
        
        # Add RGB values for each color
        for key in color_keys:
            if key in scheme:
                try:
                    # Convert hex to RGB
                    rgb = self._hex_to_rgb(scheme[key])
                    # Add as comma-separated string for CSS variables
                    enhanced_scheme[f"{key}_rgb"] = f"{rgb[0]}, {rgb[1]}, {rgb[2]}"
                except Exception as e:
                    logging.error(f"Error converting {key} color to RGB: {str(e)}")
                    # Provide fallback values if conversion fails
                    fallbacks = {
                        'primary_rgb': '0, 180, 216',
                        'secondary_rgb': '0, 119, 182',
                        'background_rgb': '18, 18, 18',
                        'surface_rgb': '30, 30, 30',
                        'text_rgb': '255, 255, 255',
                        'accent_rgb': '144, 224, 239'
                    }
                    enhanced_scheme[f"{key}_rgb"] = fallbacks.get(f"{key}_rgb", '0, 0, 0')
        
        return enhanced_scheme
    
    def calculate_contrast_ratio(self, color1: str, color2: str) -> float:
        """
        Calculate the contrast ratio between two colors according to WCAG 2.0.
        
        Returns a value between 1 and 21, where:
        - 1:1 = no contrast
        - 21:1 = max contrast (black on white)
        - WCAG AA requires 4.5:1 for normal text
        - WCAG AAA requires 7:1 for normal text
        """
        # Convert hex to RGB
        rgb1 = self._hex_to_rgb(color1)
        rgb2 = self._hex_to_rgb(color2)
        
        # Calculate relative luminance
        def get_luminance(rgb):
            r, g, b = rgb
            r, g, b = r/255, g/255, b/255
            
            # Apply gamma correction
            r = r / 12.92 if r <= 0.03928 else ((r + 0.055) / 1.055) ** 2.4
            g = g / 12.92 if g <= 0.03928 else ((g + 0.055) / 1.055) ** 2.4
            b = b / 12.92 if b <= 0.03928 else ((b + 0.055) / 1.055) ** 2.4
            
            # Calculate luminance
            return 0.2126 * r + 0.7152 * g + 0.0722 * b
        
        l1 = get_luminance(rgb1)
        l2 = get_luminance(rgb2)
        
        # Calculate contrast ratio
        if l1 > l2:
            return (l1 + 0.05) / (l2 + 0.05)
        else:
            return (l2 + 0.05) / (l1 + 0.05)
    


# Singleton instance
_color_scheme_generator = None

def get_color_scheme_generator():
    """Get or create the color scheme generator singleton."""
    global _color_scheme_generator
    if _color_scheme_generator is None:
        _color_scheme_generator = ColorSchemeGenerator()
    return _color_scheme_generator