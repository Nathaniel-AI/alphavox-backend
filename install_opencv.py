#!/usr/bin/env python3
"""
OpenCV Installation Helper for AlphaVox

This script helps install the correct OpenCV version for the AlphaVox application.
"""

import os
import sys
import subprocess
import platform

def main():
    print("OpenCV Installation Helper for AlphaVox")
    print("=======================================")
    
    # Check Python version
    python_version = platform.python_version()
    print(f"Python version: {python_version}")
    
    # Check OS
    os_name = platform.system()
    print(f"Operating system: {os_name}")
    
    # Check if OpenCV is already installed
    try:
        import cv2
        cv_version = cv2.__version__
        print(f"OpenCV is already installed (version {cv_version})")
        print("If you're having issues, you may want to reinstall it.")
        should_reinstall = input("Reinstall OpenCV? (y/n): ").lower().strip() == 'y'
        if not should_reinstall:
            print("Exiting without changes.")
            return
    except ImportError:
        print("OpenCV is not installed.")
    
    # Install or reinstall OpenCV
    print("\nInstalling OpenCV...")
    
    # Try different installation methods
    packages = [
        "opencv-python==4.7.0.72",
        "opencv-python-headless==4.7.0.72",
        "opencv-contrib-python==4.7.0.72"
    ]
    
    for package in packages:
        try:
            print(f"\nTrying to install {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            
            # Verify installation
            try:
                import cv2
                print(f"✅ Successfully installed OpenCV {cv2.__version__}")
                print("\nYou should now be able to run the AlphaVox app!")
                return
            except ImportError:
                print("❌ Installation seemed to succeed but import still fails.")
                continue
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package}")
    
    print("\n⚠️ All installation methods failed.")
    
    # Provide OS-specific advice
    if os_name == "Darwin":  # macOS
        print("\nFor macOS, try these steps:")
        print("1. Install Homebrew if you don't have it: /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"")
        print("2. Install dependencies: brew install cmake pkg-config jpeg libpng libtiff openexr eigen tbb")
        print("3. Try installing again: pip install opencv-python==4.7.0.72")
    elif os_name == "Linux":
        print("\nFor Linux, try these steps:")
        print("1. Install dependencies: sudo apt-get update && sudo apt-get install -y python3-dev python3-numpy build-essential cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev")
        print("2. Try installing again: pip install opencv-python==4.7.0.72")
    elif os_name == "Windows":
        print("\nFor Windows, try these steps:")
        print("1. Make sure you have the latest Visual C++ redistributable installed")
        print("2. Try installing the pre-built wheel: pip install opencv-python==4.7.0.72")
    
    print("\nIf you continue to have issues, consider using a Python virtual environment:")
    print("python -m venv venv")
    print("source venv/bin/activate  # On macOS/Linux")
    print("venv\\Scripts\\activate  # On Windows")
    print("pip install -r requirements.txt")

if __name__ == "__main__":
    main()
