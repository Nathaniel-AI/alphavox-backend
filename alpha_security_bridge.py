ALPHA SECURITY BRIDGE




import json
import boto3
import logging
import os
import paho.mqtt.publish as publish

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    try:
        user_profile = event.get("user_profile")
        intent = event.get("intent")  # e.g. "lock_doors", "enable_cameras", "disable_alarm"
        system = event.get("system")  # e.g. "ring", "arlo", "nest"

        if not user_profile or not intent:
            logger.error("Missing user profile or intent")
            return {"status": "error", "message": "Missing data"}

        cognitive_score = user_profile.get("cognitive_score")
        if cognitive_score is None or cognitive_score < 0.5:
            # Protect user: escalate instead of performing action
            notify_caregiver(user_profile, intent)
            return {"status": "escalated", "message": "Caregiver notified"}

        # Direct system control logic (example via MQTT)
        payload = {
            "command": intent,
            "user_id": user_profile["id"],
            "system": system
        }

        publish.single("alphavox/security", json.dumps(payload), hostname="mqtt.yourdomain.com")

        logger.info(f"Sent {intent} to {system} for user {user_profile['id']}")
        return {"status": "success", "message": f"{intent} executed"}

    except Exception as e:
        logger.error(f"Lambda error: {str(e)}")
        return {"status": "error", "message": str(e)}

def notify_caregiver(user_profile, intent):
    # Placeholder: send email, push notification, or log alert
    print(f"ALERT: User {user_profile['id']} attempted '{intent}' but requires supervision.")

