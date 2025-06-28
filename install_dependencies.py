#!/usr/bin/env python3
"""
AlphaVox Dependency Installer

This script handles installing dependencies in the correct order to prevent conflicts.
"""

import os
import sys
import subprocess
import platform

def main():
    print("\n=== AlphaVox Dependency Installer ===\n")
    
    # Check if running in a virtual environment
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("❌ Not running in a virtual environment!")
        print("Please activate your virtual environment first:")
        print("   source venv/bin/activate")
        return False
    
    print("✅ Virtual environment detected")
    print(f"Python version: {platform.python_version()}")
    
    # Core dependencies first (in correct order)
    core_dependencies = [
        "numpy==1.26.4",
        "pandas",
        "six",
        "python-dateutil",
        "pytz",
        "Flask==2.3.3",
        "Werkzeug==2.3.7",
        "Jinja2==3.1.6",
        "itsdangerous==2.2.0",
        "MarkupSafe==3.0.2", 
        "opencv-python-headless==4.8.1.78"
    ]
    
    # Other common dependencies
    other_dependencies = [
        "scikit-learn",
        "matplotlib",
        "requests",
        "pillow",
        "PyYAML",
        "flask-cors",
        "Flask-SQLAlchemy",
        "pygame",
        "openai",
        "anthropic",
        "python-dotenv"
    ]
    
    print("\n--- Installing core dependencies ---\n")
    for dep in core_dependencies:
        install_package(dep)
    
    print("\n--- Installing additional dependencies ---\n")
    for dep in other_dependencies:
        install_package(dep)
    
    print("\n--- Verifying key imports ---\n")
    verify_imports(['numpy', 'pandas', 'flask', 'cv2', 'matplotlib', 'sklearn'])
    
    print("\n=== Installation Complete ===")
    print("\nYou can now run your application:")
    print("python app.py")
    
    return True

def install_package(package_name):
    """Install a Python package using pip"""
    print(f"Installing {package_name}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"✅ Successfully installed {package_name}")
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package_name}")

def verify_imports(modules):
    """Verify that modules can be imported"""
    for module in modules:
        try:
            if module == 'cv2':
                # Special handling for opencv
                __import__('cv2')
                print(f"✅ Successfully imported {module}")
            else:
                __import__(module)
                print(f"✅ Successfully imported {module}")
        except ImportError as e:
            print(f"❌ Failed to import {module}: {e}")

if __name__ == "__main__":
    main()
