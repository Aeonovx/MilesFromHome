# Slack Server - Separate FastAPI app for Slack integration
# File: slack_server.py

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import json
import os
import logging
import uvicorn
from datetime import datetime
from slack_integration import send_slack_message, notify_slack_chat

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="iBot Slack Server", version="1.0.0")

# Import the bot instance (we'll need to modify how this works)
import sys
sys.path.append('.')

@app.get("/")
async def root():
    return {
        "service": "iBot Slack Server",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "endpoints": ["/slack/events", "/slack/test", "/health"]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "ibot-slack-server",
        "timestamp": datetime.now().isoformat(),
        "slack_configured": bool(os.getenv('SLACK_BOT_TOKEN')),
        "webhook_configured": bool(os.getenv('SLACK_WEBHOOK_URL'))
    }

@app.get("/slack/test")
async def slack_test():
    return {
        "status": "iBot Slack endpoint working",
        "timestamp": datetime.now().isoformat(),
        "message": "Ready to receive Slack events"
    }

@app.post("/slack/events")
async def handle_slack_events(request: Request):
    """Handle Slack events - URL verification and messages"""
    try:
        body = await request.body()
        body_str = body.decode('utf-8')
        data = json.loads(body_str)
        
        logger.info(f"Received Slack event: {data.get('type', 'unknown')}")
        
        # Handle URL verification challenge
        if data.get("type") == "url_verification":
            challenge = data.get("challenge")
            logger.info(f"✅ Slack challenge received: {challenge}")
            return JSONResponse({"challenge": challenge})
        
        # Handle message events
        if data.get("type") == "event_callback":
            event = data.get("event", {})
            
            if (event.get("type") == "message" and 
                not event.get("bot_id") and 
                event.get("text")):
                
                text = event.get("text", "").strip().lower()
                user_id = event.get('user', '')
                channel = event.get('channel', '')
                original_message = event.get('text', '')
                
                # Check if should respond
                should_respond = (
                    event.get("channel_type") == "im" or  # Direct message
                    "ibot" in text  # Contains "ibot"
                )
                
                if should_respond:
                    # Clean the message
                    message = original_message
                    if "ibot" in message.lower():
                        message = message.lower().replace("ibot", "").strip()
                    
                    if message:
                        logger.info(f"Processing message from {user_id}: {message[:50]}...")
                        
                        # Get bot response using a simple function
                        response = get_bot_response(message)
                        
                        # Send response back to Slack
                        success = send_slack_message(channel, response)
                        
                        if success:
                            logger.info("✅ Response sent to Slack")
                            # Notify about chat activity
                            try:
                                notify_slack_chat(user_id, message, response)
                            except Exception as e:
                                logger.error(f"Notification error: {e}")
                        else:
                            logger.error("❌ Failed to send response to Slack")
        
        return JSONResponse({"status": "ok"})
        
    except Exception as e:
        logger.error(f"Slack event error: {e}")
        return JSONResponse({"error": "Internal server error"}, status_code=500)

def get_bot_response(message: str) -> str:
    """Simple bot response function - can be enhanced later"""
    message_lower = message.lower()
    
    # Simple keyword-based responses
    if any(word in message_lower for word in ['hello', 'hi', 'hey']):
        return "Hello! I'm iBot, your iTethr assistant. Ask me anything about iTethr platform!"
    
    elif "community" in message_lower or "hub" in message_lower:
        return "iTethr has a hierarchical community structure with Communities, Hubs, and Rooms. Communities can contain multiple Hubs, and Hubs contain Rooms for specific discussions."
    
    elif "authentication" in message_lower or "sign up" in message_lower:
        return "iTethr supports multiple sign-up methods: Google OAuth, Apple OAuth, and traditional email/password registration with a simplified 3-step onboarding process."
    
    elif "bubble" in message_lower or "interface" in message_lower:
        return "iTethr uses a revolutionary bubble-based interface instead of traditional navigation bars. Users interact with floating animated bubbles representing Communities, Loops, and contacts."
    
    elif "aeono" in message_lower or "ai" in message_lower:
        return "Aeono is iTethr's integrated AI assistant designed to help users connect with peers, find communities, and navigate the platform efficiently."
    
    elif "ifeed" in message_lower or "feed" in message_lower:
        return "iTethr iFeed is the global content stream where public posts, announcements, and Loop discussions are surfaced, functioning like a fusion of Reddit and Twitter."
    
    elif "help" in message_lower:
        return """I can help you with:
• iTethr platform overview
• Community structure (Communities, Hubs, Rooms)
• User authentication methods
• Bubble interface design
• Aeono AI assistant features
• iFeed functionality

Just ask me about any of these topics!"""
    
    else:
        return f"I received your message: '{message}'. I can help you with iTethr platform questions. Try asking about communities, authentication, or the bubble interface!"

if __name__ == "__main__":
    port = int(os.getenv('SLACK_PORT', '8081'))  # Different port from Gradio
    logger.info(f"🚀 Starting iBot Slack Server on port {port}")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        log_level="info"
    )