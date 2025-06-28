import os
import json
import logging
import boto3
from flask import render_template, Blueprint, request, jsonify
from logging_config import configure_logging
from app import app

# Configure application logging
configure_logging()
logger = logging.getLogger(__name__)

# Lambda setup (adjust if needed)
lambda_client = boto3.client('lambda', region_name='us-east-1')  # Set your AWS region

def send_to_lambda(user_profile, intent, system):
    """Send a secure request to the Lambda function for AlphaVox system actions."""
    payload = {
        "user_profile": user_profile,
        "intent": intent,
        "system": system
    }

    try:
        response = lambda_client.invoke(
            FunctionName='alpha_security_bridge',
            InvocationType='RequestResponse',
            Payload=json.dumps(payload)
        )
        response_payload = json.loads(response['Payload'].read())
        logger.info(f"Lambda response: {response_payload}")
        return response_payload
    except Exception as e:
        logger.error(f"Lambda invocation error: {str(e)}")
        return {"status": "error", "message": str(e)}

# ========== BLUEPRINTS ==========

# Create a test blueprint
test_bp = Blueprint('test', __name__, url_prefix='/test')

@test_bp.route('/')
def test_route():
    logger.info("Test route accessed")
    return "Learning Hub test route is working"

@test_bp.route('/trigger-lambda', methods=['POST'])
def trigger_lambda():
    logger.info("Trigger Lambda endpoint hit")
    
    user_profile = {
        "id": "alpha001",
        "cognitive_score": 0.3
    }
    intent = "lock_doors"
    system = "ring"

    result = send_to_lambda(user_profile, intent, system)
    return jsonify(result)

# Register the test blueprint
app.register_blueprint(test_bp)

# Register all other routes after test
from app_routes import *  # Your main routes

logger.info("AlphaVox application initialized")

# ========== RUN APP ==========
if __name__ == "__main__":
    logger.info(f"Starting AlphaVox server on port 5000")
    app.run(host="0.0.0.0", port=5000)

