#!/usr/bin/env python3
"""
AlphaVox Endpoint Tester
Run this script to test key endpoints of the AlphaVox application.
"""

import requests
import json
import time
import sys
from urllib.parse import urljoin

# Configuration
BASE_URL = "http://localhost:5000"
TEST_USER = "TestUser"
ENDPOINTS = [
    # Public endpoints
    {"url": "/", "method": "GET", "name": "Homepage"},
    {"url": "/public/hardware-test", "method": "GET", "name": "Public Hardware Test"},
    {"url": "/voice_test", "method": "GET", "name": "Voice Test"},
    {"url": "/simple_voice_test", "method": "GET", "name": "Simple Voice Test"},
    
    # API endpoints
    {"url": "/api/generate_speech", "method": "POST", 
     "data": {"text": "This is a test of the speech API.", "emotion": "neutral"}, 
     "name": "Generate Speech API"},
    
    # Protected endpoints (these will create a session)
    {"url": "/start", "method": "POST", "data": {"name": TEST_USER}, "name": "Start Session"},
    {"url": "/home", "method": "GET", "name": "Home Page", "requires_session": True},
    {"url": "/hardware_test", "method": "GET", "name": "Hardware Test", "requires_session": True},
    {"url": "/symbols", "method": "GET", "name": "Symbols Page", "requires_session": True},
    
    # Gesture endpoints
    {"url": "/speak/nod", "method": "GET", "name": "Nod Gesture"},
    {"url": "/speak/wave", "method": "GET", "name": "Wave Gesture"},
    
    # Symbol endpoints
    {"url": "/symbol/yes", "method": "POST", "name": "Yes Symbol"},
    {"url": "/symbol/no", "method": "POST", "name": "No Symbol"},
    
    # AI control endpoints
    {"url": "/ai_control", "method": "GET", "name": "AI Control", "requires_session": True},
]

def run_tests():
    """Run tests against all configured endpoints"""
    print(f"AlphaVox Endpoint Tester")
    print(f"Testing against: {BASE_URL}")
    print("-" * 50)
    
    session = requests.Session()
    results = {"passed": 0, "failed": 0, "skipped": 0}
    
    # First run the start endpoint to create a session
    for endpoint in [e for e in ENDPOINTS if e["url"] == "/start"]:
        test_endpoint(endpoint, session, results)
    
    # Then test all other endpoints
    for endpoint in [e for e in ENDPOINTS if e["url"] != "/start"]:
        test_endpoint(endpoint, session, results)
    
    # Print summary
    print("-" * 50)
    print(f"Test Summary:")
    print(f"Passed: {results['passed']}")
    print(f"Failed: {results['failed']}")
    print(f"Skipped: {results['skipped']}")
    print("-" * 50)
    
    if results['failed'] > 0:
        sys.exit(1)  # Exit with error code if any tests failed

def test_endpoint(endpoint, session, results):
    """Test a single endpoint"""
    url = urljoin(BASE_URL, endpoint["url"])
    method = endpoint["method"]
    name = endpoint.get("name", url)
    data = endpoint.get("data", {})
    
    print(f"Testing: {name} ({method} {endpoint['url']})")
    
    try:
        if method == "GET":
            response = session.get(url, timeout=10)
        elif method == "POST":
            if isinstance(data, dict):
                response = session.post(url, json=data, timeout=10)
            else:
                response = session.post(url, data=data, timeout=10)
        else:
            print(f"  [SKIP] Unsupported method: {method}")
            results["skipped"] += 1
            return
        
        # Check the response
        if 200 <= response.status_code < 300:
            print(f"  [PASS] Status code: {response.status_code}")
            results["passed"] += 1
        else:
            print(f"  [FAIL] Status code: {response.status_code}")
            print(f"  Response: {response.text[:100]}...")
            results["failed"] += 1
            
    except requests.RequestException as e:
        print(f"  [FAIL] Error: {e}")
        results["failed"] += 1
    
    # Add a small delay to avoid overwhelming the server
    time.sleep(0.5)

if __name__ == "__main__":
    # Check if server is running
    try:
        requests.get(BASE_URL, timeout=2)
        run_tests()
    except requests.ConnectionError:
        print(f"ERROR: Cannot connect to {BASE_URL}")
        print("Make sure the application is running by executing 'python run.py' first")
        sys.exit(1)
