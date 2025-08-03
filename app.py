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

# --- Helper Functions ---
def categorize_article(headline, summary):
    text = f"{headline.lower()} {summary.lower()}"
    for category, keywords in CATEGORIES.items():
        if any(keyword in text for keyword in keywords):
            return category
    return "General"

def get_image_from_entry(entry):
    """Extract image URL from RSS entry with multiple fallback methods"""
    
    # Method 1: Check media_content (common in RSS feeds)
    if hasattr(entry, 'media_content') and entry.media_content:
        for media in entry.media_content:
            if media.get('type', '').startswith('image/'):
                return media.get('url')
    
    # Method 2: Check media_thumbnail
    if hasattr(entry, 'media_thumbnail') and entry.media_thumbnail:
        return entry.media_thumbnail[0].get('url')
    
    # Method 3: Check enclosures for images
    if hasattr(entry, 'enclosures') and entry.enclosures:
        for enclosure in entry.enclosures:
            if enclosure.get('type', '').startswith('image/'):
                return enclosure.get('href')
    
    # Method 4: Check links for image types
    if hasattr(entry, 'links') and entry.links:
        for link in entry.links:
            if link.get('type', '').startswith('image/'):
                return link.get('href')
    
    # Method 5: Parse summary/description for img tags
    summary = entry.get('summary', '') or entry.get('description', '')
    if summary:
        img_match = re.search(r'<img[^>]+src=["\']([^"\']+)["\']', summary)
        if img_match:
            return img_match.group(1)
    
    # Method 6: Look for images in content
    if hasattr(entry, 'content') and entry.content:
        for content in entry.content:
            content_value = content.get('value', '')
            img_match = re.search(r'<img[^>]+src=["\']([^"\']+)["\']', content_value)
            if img_match:
                return img_match.group(1)
    
    # Method 7: Generate a themed placeholder based on source
    source_colors = {
        "Reuters": "2e5266",
        "BBC News": "bb1919", 
        "The Guardian": "052962",
        "TechCrunch": "00d084",
        "Associated Press": "0066cc"
    }
    
    source_name = getattr(entry, 'source', 'News')
    color = source_colors.get(source_name, "1a1f26")
    
    return f"https://via.placeholder.com/800x400/{color}/ffffff?text={source_name.replace(' ', '+')}"

def explain_with_ai(summary):
    if not os.environ.get("GROQ_API_KEY"): 
        return "AI is not configured."
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a news analyst. Expand the summary into a clear, 2-3 paragraph explanation. Focus on context and importance. Use clean markdown."}, 
                {"role": "user", "content": f"Explain: {summary}"}
            ], 
            model="llama3-8b-8192"
        )
        return chat_completion.choices[0].message.content
    except Exception as e: 
        return "Could not generate AI explanation."

# --- Core News Fetching ---
def fetch_and_process_news():
    processed_news = []
    article_id = 0
    
    for source, url in NEWS_FEEDS.items():
        try:
            print(f"Fetching from {source}...")
            feed = feedparser.parse(url)
            
            for entry in feed.entries[:8]:
                summary = entry.get('summary', entry.get('description', 'No summary available.'))
                image_url = get_image_from_entry(entry)
                
                # Log image extraction for debugging
                print(f"Article: {entry.title[:50]}... | Image: {image_url}")
                
                processed_news.append({
                    "id": article_id, 
                    "source": source, 
                    "headline": entry.title, 
                    "link": entry.link, 
                    "published": entry.get("published", "N/A"),
                    "image_url": image_url, 
                    "summary": summary, 
                    "explained_version": explain_with_ai(summary),
                    "hotness": random.randint(70, 100), 
                    "category": categorize_article(entry.title, summary),
                    "keywords": [kw.lower() for kw in re.findall(r'\b\w{4,}\b', entry.title.lower()) if kw.lower() not in STOP_WORDS]
                })
                article_id += 1
        except Exception as e:
            print(f"Error fetching from {source}: {e}")
            continue
            
    with open(NEWS_FILE, "w") as f:
        json.dump({"articles": processed_news, "last_updated": datetime.utcnow().isoformat()}, f, indent=4)
    
    print(f"✅ News processing complete at {datetime.now()}. {len(processed_news)} articles fetched.")

# --- API Endpoints ---
@app.on_event("startup")
def start_scheduler():
    scheduler = BackgroundScheduler(timezone="UTC")
    fetch_and_process_news()
    scheduler.add_job(fetch_and_process_news, 'interval', minutes=30)
    scheduler.start()

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