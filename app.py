import os
import json
import random
import re
from collections import Counter
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from apscheduler.schedulers.background import BackgroundScheduler
import feedparser
from groq import Groq
from dotenv import load_dotenv
import hashlib

# --- Configuration & Setup ---
load_dotenv()
app = FastAPI()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# --- Constants ---
NEWS_FILE = "news.json"
SUBSCRIBERS_FILE = "subscribers.json"
NEWS_FEEDS = { 
    "Reuters": "http://feeds.reuters.com/reuters/topNews", 
    "BBC News": "http://feeds.bbci.co.uk/news/rss.xml", 
    "The Guardian": "https://www.theguardian.com/world/rss", 
    "TechCrunch": "https://techcrunch.com/feed/", 
    "Associated Press": "https://apnews.com/hub/ap-top-news/rss.xml" 
}
CATEGORIES = {
    "Tech": ["apple", "google", "microsoft", "tech", "ai", "crypto", "startup", "software"],
    "EU News": ["europe", "european", "uk", "france", "germany", "brussels", "brexit"],
    "Travel": ["travel", "tourism", "flights", "vacation", "holiday", "hotel"]
}
STOP_WORDS = set(["the", "a", "an", "in", "on", "of", "for", "to", "with", "and", "or", "is", "are", "was", "were"])

# Enhanced source-specific styling
SOURCE_THEMES = {
    "Reuters": {
        "color": "ff6600",
        "text_color": "ffffff",
        "icon": "📰"
    },
    "BBC News": {
        "color": "bb1919", 
        "text_color": "ffffff",
        "icon": "📺"
    },
    "The Guardian": {
        "color": "052962",
        "text_color": "ffffff", 
        "icon": "🗞️"
    },
    "TechCrunch": {
        "color": "00d084",
        "text_color": "000000",
        "icon": "💻"
    },
    "Associated Press": {
        "color": "0066cc",
        "text_color": "ffffff",
        "icon": "📄"
    }
}

# --- Helper Functions ---
def categorize_article(headline, summary):
    text = f"{headline.lower()} {summary.lower()}"
    for category, keywords in CATEGORIES.items():
        if any(keyword in text for keyword in keywords):
            return category
    return "General"

def create_themed_placeholder(source, headline, size="400x180"):
    """Create a themed placeholder image URL with source branding"""
    theme = SOURCE_THEMES.get(source, {
        "color": "1a1f26",
        "text_color": "8b949e", 
        "icon": "📰"
    })
    
    # Create a short, readable text for the placeholder
    icon = theme["icon"]
    source_name = source.replace(" ", "+")
    
    # Use headline hash to create consistent but varied backgrounds
    headline_hash = hashlib.md5(headline.encode()).hexdigest()[:6]
    
    return f"https://via.placeholder.com/{size}/{theme['color']}/{theme['text_color']}?text={icon}+{source_name}"

def get_image_from_entry(entry, source, headline):
    """Extract image URL from RSS entry with comprehensive fallback methods"""
    
    # Method 1: Check media_content (most common)
    if hasattr(entry, 'media_content') and entry.media_content:
        for media in entry.media_content:
            url = media.get('url', '')
            if url and ('jpg' in url.lower() or 'jpeg' in url.lower() or 'png' in url.lower() or 'webp' in url.lower()):
                print(f"✅ Found media_content image: {url[:50]}...")
                return url
    
    # Method 2: Check media_thumbnail
    if hasattr(entry, 'media_thumbnail') and entry.media_thumbnail:
        url = entry.media_thumbnail[0].get('url', '')
        if url:
            print(f"✅ Found thumbnail image: {url[:50]}...")
            return url
    
    # Method 3: Check enclosures for images
    if hasattr(entry, 'enclosures') and entry.enclosures:
        for enclosure in entry.enclosures:
            if enclosure.get('type', '').startswith('image/'):
                url = enclosure.get('href', '')
                if url:
                    print(f"✅ Found enclosure image: {url[:50]}...")
                    return url
    
    # Method 4: Check links for image types
    if hasattr(entry, 'links') and entry.links:
        for link in entry.links:
            if link.get('type', '').startswith('image/'):
                url = link.get('href', '')
                if url:
                    print(f"✅ Found link image: {url[:50]}...")
                    return url
    
    # Method 5: Parse summary/description for img tags
    summary = entry.get('summary', '') or entry.get('description', '')
    if summary:
        # Look for img tags
        img_matches = re.findall(r'<img[^>]+src=["\']([^"\']+)["\']', summary, re.IGNORECASE)
        for img_url in img_matches:
            if img_url and ('jpg' in img_url.lower() or 'jpeg' in img_url.lower() or 'png' in img_url.lower() or 'webp' in img_url.lower()):
                print(f"✅ Found summary image: {img_url[:50]}...")
                return img_url
    
    # Method 6: Look for images in content
    if hasattr(entry, 'content') and entry.content:
        for content in entry.content:
            content_value = content.get('value', '')
            img_matches = re.findall(r'<img[^>]+src=["\']([^"\']+)["\']', content_value, re.IGNORECASE)
            for img_url in img_matches:
                if img_url and ('jpg' in img_url.lower() or 'jpeg' in img_url.lower() or 'png' in img_url.lower() or 'webp' in img_url.lower()):
                    print(f"✅ Found content image: {img_url[:50]}...")
                    return img_url
    
    # Method 7: Try to extract from link URLs (some feeds include images in URLs)
    if hasattr(entry, 'link') and entry.link:
        # Some news sites have predictable image URL patterns
        if 'bbc.co.uk' in entry.link:
            # BBC often has images we can construct
            pass
        elif 'theguardian.com' in entry.link:
            # Guardian sometimes has extractable images
            pass
    
    # If no image found, create themed placeholder
    print(f"❌ No image found for '{headline[:30]}...', using themed placeholder")
    return create_themed_placeholder(source, headline)

def explain_with_ai(summary):
    if not os.environ.get("GROQ_API_KEY"): 
        return "AI is not configured."
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system", 
                    "content": "You are a news analyst. Expand the summary into a clear, 2-3 paragraph explanation. Focus on context and importance. Use clean markdown."
                }, 
                {
                    "role": "user", 
                    "content": f"Explain: {summary}"
                }
            ], 
            model="llama3-8b-8192",
            max_tokens=int(os.environ.get("MAX_TOKENS", 200)),
            temperature=float(os.environ.get("TEMPERATURE", 0.01))
        )
        return chat_completion.choices[0].message.content
    except Exception as e: 
        print(f"AI explanation error: {e}")
        return "Could not generate AI explanation."

# --- Core News Fetching ---
def fetch_and_process_news():
    processed_news = []
    article_id = 0
    
    print(f"🚀 Starting news fetch at {datetime.now()}")
    
    for source, url in NEWS_FEEDS.items():
        try:
            print(f"📡 Fetching from {source}...")
            feed = feedparser.parse(url)
            
            if not feed.entries:
                print(f"⚠️ No entries found for {source}")
                continue
            
            for entry in feed.entries[:8]:
                try:
                    summary = entry.get('summary', entry.get('description', 'No summary available.'))
                    headline = entry.get('title', 'No title')
                    
                    # GUARANTEED image - either real or themed placeholder
                    image_url = get_image_from_entry(entry, source, headline)
                    
                    article = {
                        "id": article_id, 
                        "source": source, 
                        "headline": headline, 
                        "link": entry.get('link', '#'), 
                        "published": entry.get("published", "N/A"),
                        "image_url": image_url,  # This will ALWAYS have a value
                        "summary": summary, 
                        "explained_version": explain_with_ai(summary),
                        "hotness": random.randint(70, 100), 
                        "category": categorize_article(headline, summary),
                        "keywords": [kw.lower() for kw in re.findall(r'\b\w{4,}\b', headline.lower()) if kw.lower() not in STOP_WORDS]
                    }
                    
                    processed_news.append(article)
                    article_id += 1
                    
                    print(f"✅ Processed: {headline[:40]}... | Image: {'✅' if 'placeholder' not in image_url else '🎨'}")
                    
                except Exception as e:
                    print(f"❌ Error processing article from {source}: {e}")
                    continue
                    
        except Exception as e:
            print(f"❌ Error fetching from {source}: {e}")
            continue
    
    # Save processed news
    try:
        with open(NEWS_FILE, "w") as f:
            json.dump({
                "articles": processed_news, 
                "last_updated": datetime.utcnow().isoformat()
            }, f, indent=4)
        
        print(f"✅ News processing complete! {len(processed_news)} articles saved.")
        
    except Exception as e:
        print(f"❌ Error saving news file: {e}")

# --- API Endpoints ---
@app.on_event("startup")
def start_scheduler():
    scheduler = BackgroundScheduler(timezone="UTC")
    # Initial fetch
    fetch_and_process_news()
    # Schedule regular updates
    scheduler.add_job(fetch_and_process_news, 'interval', minutes=30)
    scheduler.start()
    print("🔄 Scheduler started - news will update every 30 minutes")

@app.get("/api/news")
async def get_news(search: str = None, category: str = None):
    if not os.path.exists(NEWS_FILE): 
        raise HTTPException(status_code=503, detail="News is being processed.")
    
    with open(NEWS_FILE, "r") as f:
        data = json.load(f)
        articles = data["articles"]
        
        if search: 
            articles = [a for a in articles if search.lower() in a['headline'].lower()]
        if category and category != "All": 
            articles = [a for a in articles if a['category'] == category]
            
        return {"articles": articles, "categories": ["All"] + list(CATEGORIES.keys())}

@app.get("/api/news/{article_id}")
async def get_article(article_id: int):
    if not os.path.exists(NEWS_FILE): 
        raise HTTPException(status_code=404, detail="News file not found.")
    
    with open(NEWS_FILE, "r") as f:
        articles = json.load(f)["articles"]
        target = next((a for a in articles if a['id'] == article_id), None)
        
        if not target: 
            raise HTTPException(status_code=404, detail="Article not found.")
            
        related = [a for a in articles if a['id'] != target['id'] and any(kw in a['keywords'] for kw in target['keywords'])][:3]
        return {"article": target, "related": related}

@app.get("/api/stats")
async def get_stats():
    if not os.path.exists(NEWS_FILE): 
        return {"error": "No data."}
    
    with open(NEWS_FILE, "r") as f:
        data = json.load(f)
    
    articles = data["articles"]
    subscribers = json.load(open(SUBSCRIBERS_FILE)) if os.path.exists(SUBSCRIBERS_FILE) else []
    
    return {
        "total_articles": len(articles), 
        "subscribers": len(subscribers), 
        "last_updated": data["last_updated"],
        "articles_per_source": dict(Counter(a['source'] for a in articles)),
        "articles_per_category": dict(Counter(a['category'] for a in articles))
    }

app.mount("/", StaticFiles(directory="static", html=True), name="static")