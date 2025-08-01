# Slack Integration Module with ibot trigger
# File: slack_integration.py

import requests
import os
import logging
from datetime import datetime
import json
import hashlib
import hmac

logger = logging.getLogger(__name__)

def send_to_slack(message, username="iTethr Bot", channel=None):
    """Send message to Slack using webhook"""
    webhook_url = os.getenv('SLACK_WEBHOOK_URL')
    
    if not webhook_url:
        logger.warning("SLACK_WEBHOOK_URL not configured")
        return False
    
    try:
        payload = {
            'text': message,
            'username': username,
            'icon_emoji': ':robot_face:'
        }
        
        if channel:
            payload['channel'] = channel
        
        response = requests.post(webhook_url, json=payload, timeout=10)
        
        if response.status_code == 200:
            logger.info("✅ Message sent to Slack successfully")
            return True
        else:
            logger.error(f"❌ Slack webhook failed: {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Slack integration error: {e}")
        return False

def send_slack_message(channel, message, thread_ts=None):
    """Send message to Slack using Bot token (for responses)"""
    bot_token = os.getenv('SLACK_BOT_TOKEN')
    
    if not bot_token:
        logger.warning("SLACK_BOT_TOKEN not configured")
        return False
    
    try:
        url = "https://slack.com/api/chat.postMessage"
        headers = {
            "Authorization": f"Bearer {bot_token}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "channel": channel,
            "text": message,
            "username": "iBot",
            "icon_emoji": ":robot_face:"
        }
        
        if thread_ts:
            payload["thread_ts"] = thread_ts
        
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        
        if response.status_code == 200 and response.json().get("ok"):
            logger.info("✅ Bot message sent to Slack successfully")
            return True
        else:
            logger.error(f"❌ Slack bot message failed: {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Slack bot message error: {e}")
        return False

def verify_slack_signature(request_body, timestamp, signature):
    """Verify that requests are coming from Slack"""
    signing_secret = os.getenv('SLACK_SIGNING_SECRET')
    
    if not signing_secret:
        logger.warning("SLACK_SIGNING_SECRET not configured")
        return False
    
    try:
        # Create the signature base string
        sig_basestring = f"v0:{timestamp}:{request_body}"
        
        # Create the signature
        my_signature = 'v0=' + hmac.new(
            signing_secret.encode(),
            sig_basestring.encode(),
            hashlib.sha256
        ).hexdigest()
        
        # Compare signatures
        return hmac.compare_digest(my_signature, signature)
        
    except Exception as e:
        logger.error(f"Signature verification error: {e}")
        return False

def extract_user_info(event):
    """Extract user information from Slack event"""
    try:
        user_id = event.get('user', '')
        channel = event.get('channel', '')
        text = event.get('text', '').strip()
        ts = event.get('ts', '')
        
        # Remove bot mention from text and check for ibot trigger
        text = text.replace('<@A098WUDF1QR>', '').strip()  # Replace with your actual bot user ID
        
        # Check if message starts with "ibot" (case insensitive)
        if text.lower().startswith('ibot'):
            text = text[4:].strip()  # Remove "ibot" and get the rest
        
        return {
            'user_id': user_id,
            'channel': channel,
            'message': text,
            'timestamp': ts
        }
    except Exception as e:
        logger.error(f"Error extracting user info: {e}")
        return None

def notify_startup():
    """Notify Slack that bot is starting"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = f"🚀 *iBot Started Successfully!*\n⏰ Time: {timestamp}\n🏢 AeonovX Team Ready\n💬 Type 'ibot' + your question to chat with me!"
    return send_to_slack(message)

def notify_login(name, role):
    """Notify Slack when team member logs in"""
    timestamp = datetime.now().strftime("%H:%M")
    message = f"👤 *Team Login*\n🏷️ {name} ({role})\n⏰ {timestamp}"
    return send_to_slack(message)

def notify_question(user, question):
    """Notify Slack when user asks a question"""
    truncated_question = question[:100] + "..." if len(question) > 100 else question
    message = f"🤖 *iBot Activity*\n👤 User: {user}\n❓ Question: {truncated_question}\n💡 Response sent"
    return send_to_slack(message)

def notify_error(error_message):
    """Notify Slack of system errors"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = f"⚠️ *System Error*\n🕐 {timestamp}\n💥 Error: {error_message}"
    return send_to_slack(message)

def notify_slack_chat(user_id, question, response):
    """Notify about Slack chat activity"""
    message = f"💬 *iBot Slack Chat*\n👤 User: <@{user_id}>\n❓ Question: {question[:50]}...\n✅ Responded in Slack"
    return send_to_slack(message)

def notify_daily_stats(stats):
    """Send daily statistics to Slack"""
    message = f"📊 *Daily iBot Statistics*\n{stats}"
    return send_to_slack(message)

def notify_custom(title, details, emoji="ℹ️"):
    """Send custom notification to Slack"""
    message = f"{emoji} *{title}*\n{details}"
    return send_to_slack(message)

# Quick test function
def test_slack_connection():
    """Test if Slack webhook is working"""
    test_message = "🧪 Testing iBot Slack integration"
    result = send_to_slack(test_message)
    if result:
        print("✅ Slack integration working!")
    else:
        print("❌ Slack integration failed!")
    return result