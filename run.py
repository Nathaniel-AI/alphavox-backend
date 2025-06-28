#!/usr/bin/env python3
"""
AlphaVox Application Runner
Run this script to start the AlphaVox application locally with debug mode enabled.
"""

import os
from app import app

if __name__ == '__main__':
    # Set Flask environment variables
    os.environ['FLASK_ENV'] = 'development'
    os.environ['FLASK_DEBUG'] = '1'
    
    print("Starting AlphaVox application...")
    print("Access the application at http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    
    # Run the Flask application
    app.run(host='0.0.0.0', port=5000, debug=True)
