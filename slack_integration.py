# Slack Integration Module
# File: slack_integration.py

import requests
import os
import logging
from datetime import datetime

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

def notify_startup():
    """Notify Slack that bot is starting"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = f"🚀 *iTethr Bot Started Successfully!*\n⏰ Time: {timestamp}\n🏢 AeonovX Team Ready"
    return send_to_slack(message)

def notify_login(name, role):
    """Notify Slack when team member logs in"""
    timestamp = datetime.now().strftime("%H:%M")
    message = f"👤 *Team Login*\n🏷️ {name} ({role})\n⏰ {timestamp}"
    return send_to_slack(message)

def notify_question(user, question):
    """Notify Slack when user asks a question"""
    truncated_question = question[:100] + "..." if len(question) > 100 else question
    message = f"🤖 *Bot Activity*\n👤 User: {user}\n❓ Question: {truncated_question}\n💡 Response sent"
    return send_to_slack(message)

def notify_error(error_message):
    """Notify Slack of system errors"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = f"⚠️ *System Error*\n🕐 {timestamp}\n💥 Error: {error_message}"
    return send_to_slack(message)

def notify_daily_stats(stats):
    """Send daily statistics to Slack"""
    message = f"📊 *Daily Bot Statistics*\n{stats}"
    return send_to_slack(message)

def notify_custom(title, details, emoji="ℹ️"):
    """Send custom notification to Slack"""
    message = f"{emoji} *{title}*\n{details}"
    return send_to_slack(message)

# Quick test function
def test_slack_connection():
    """Test if Slack webhook is working"""
    test_message = "🧪 Testing Slack integration from iTethr Bot"
    result = send_to_slack(test_message)
    if result:
        print("✅ Slack integration working!")
    else:
        print("❌ Slack integration failed!")
    return result