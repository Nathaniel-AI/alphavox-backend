#!/usr/bin/env python3
"""
AlphaVox Setup Script

This script helps set up the Python environment and dependencies for AlphaVox.
"""

import os
import sys
import subprocess
import platform

def main():
    print("AlphaVox Setup Helper")
    print("=====================")
    
    # Check Python version
    python_version = platform.python_version()
    print(f"Python version: {python_version}")
    python_major, python_minor, _ = map(int, python_version.split('.'))
    
    if python_major < 3 or (python_major == 3 and python_minor < 8):
        print("⚠️ Warning: AlphaVox requires Python 3.8 or higher")
        response = input("Would you like to install a compatible Python version? (y/n): ")
        if response.lower() == 'y':
            install_python()
    
    # Create virtual environment if it doesn't exist
    if not os.path.exists('venv'):
        print("\nCreating virtual environment...")
        try:
            subprocess.check_call([sys.executable, '-m', 'venv', 'venv'])
            print("✅ Virtual environment created successfully")
        except subprocess.CalledProcessError:
            print("❌ Failed to create virtual environment")
            print("Try installing the venv module: pip install virtualenv")
            return
    
    # Activate virtual environment
    print("\nActivating virtual environment...")
    venv_python = os.path.join('venv', 'bin', 'python')
    venv_pip = os.path.join('venv', 'bin', 'pip')
    
    # Update pip
    print("\nUpdating pip...")
    try:
        subprocess.check_call([venv_pip, 'install', '--upgrade', 'pip'])
        print("✅ Pip updated successfully")
    except subprocess.CalledProcessError:
        print("❌ Failed to update pip")
    
    # Install dependencies
    print("\nInstalling dependencies...")
    try:
        # First try to install OpenCV separately
        subprocess.check_call([venv_pip, 'install', 'opencv-python==4.7.0.72'])
        print("✅ OpenCV installed successfully")
    except subprocess.CalledProcessError:
        print("⚠️ Failed to install OpenCV package")
        print("Trying alternative installation methods...")
        
        try:
            # Try headless version
            subprocess.check_call([venv_pip, 'install', 'opencv-python-headless==4.7.0.72'])
            print("✅ OpenCV headless version installed successfully")
        except subprocess.CalledProcessError:
            print("❌ Failed to install OpenCV")
            print("Please try installing system dependencies first:")
            print("  brew install cmake pkg-config")
            print("  brew install jpeg libpng libtiff openexr eigen tbb")
    
    # Install compatible NumPy first (crucial for Python 3.13)
    print("\nInstalling compatible NumPy version...")
    try:
        # For Python 3.13, we need to use NumPy 1.26.x for OpenCV compatibility
        subprocess.check_call([venv_pip, 'install', 'numpy==1.26.4'])
        print("✅ NumPy installed successfully")
    except subprocess.CalledProcessError:
        print("❌ Failed to install NumPy")
        print("This is critical for OpenCV compatibility")
    
    # Now try OpenCV again with the right NumPy
    try:
        subprocess.check_call([venv_pip, 'install', 'opencv-python-headless==4.8.1.78'])
        print("✅ OpenCV headless version installed successfully")
    except subprocess.CalledProcessError:
        print("❌ Still having issues with OpenCV")
        print("Attempting to install legacy web.py dependency...")
        subprocess.check_call([venv_pip, 'install', 'cgi-tools'])
    
    # Install other requirements
    try:
        subprocess.check_call([venv_pip, 'install', '-r', 'requirements.txt'])
        print("✅ All dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install some dependencies: {e}")
        print("Attempting to install individual packages...")
        try:
            # Web.py is problematic with Python 3.13, use a different approach
            subprocess.check_call([venv_pip, 'install', 'flask'])  # Alternative to web.py
            subprocess.check_call([venv_pip, 'install', 'pygame'])
            subprocess.check_call([venv_pip, 'install', 'requests'])
            print("✅ Essential packages installed individually")
        except subprocess.CalledProcessError:
            print("❌ Individual installation also failed")
    
    print("\n==================================")
    print("Setup complete! To run AlphaVox:")
    print("1. Activate the virtual environment:")
    print("   source venv/bin/activate")
    print("2. Run the application:")
    print("   python app.py")
    print("==================================")

def install_python():
    """Install Python using the appropriate method for the platform"""
    system = platform.system()
    
    if system == "Darwin":  # macOS
        print("\nInstalling Python via Homebrew...")
        try:
            # Check if Homebrew is installed
            subprocess.check_call(['which', 'brew'])
            # Install Python
            subprocess.check_call(['brew', 'install', 'python@3.11'])
            print("✅ Python installed successfully")
        except subprocess.CalledProcessError:
            print("❌ Homebrew not found or installation failed")
            print("Please install Homebrew first:")
            print("/bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"")
            print("Then run this script again")
    else:
        print(f"Automatic Python installation not supported on {system}")
        print("Please visit https://www.python.org/downloads/ to install Python 3.8 or higher")

if __name__ == "__main__":
    main()

