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

# Enhanced news feeds with category-specific sources
NEWS_FEEDS = { 
    "Reuters": "http://feeds.reuters.com/reuters/topNews", 
    "BBC News": "http://feeds.bbci.co.uk/news/rss.xml", 
    "The Guardian": "https://www.theguardian.com/world/rss", 
    "TechCrunch": "https://techcrunch.com/feed/", 
    "Associated Press": "https://apnews.com/hub/ap-top-news/rss.xml",
    # Category-specific feeds
    "Tech Reuters": "http://feeds.reuters.com/reuters/technologyNews",
    "EU Guardian": "https://www.theguardian.com/world/europe-news/rss",
    "Travel Guardian": "https://www.theguardian.com/travel/rss",
    "Tech Guardian": "https://www.theguardian.com/technology/rss"
}

# Refined categorization with stronger, more specific keywords
TECH_KEYWORDS = {
    # High-value tech keywords (weight 5)
    "high_value": [
        "artificial intelligence", "machine learning", "chatgpt", "openai", "ai model", 
        "cryptocurrency", "bitcoin", "ethereum", "blockchain", "crypto", "nft",
        "iphone", "android", "ios", "app store", "google play", "smartphone",
        "tesla", "electric vehicle", "self-driving", "autonomous vehicle",
        "startup", "tech startup", "silicon valley", "venture capital",
        "software", "programming", "coding", "developer", "github",
        "cloud computing", "aws", "azure", "google cloud", "saas",
        "cybersecurity", "data breach", "hack", "cyber attack",
        "social media", "facebook", "twitter", "instagram", "tiktok", "youtube",
        "streaming", "netflix", "spotify", "gaming", "video game", "esports"
    ],
    # Medium-value tech keywords (weight 3)
    "medium_value": [
        "apple", "google", "microsoft", "amazon", "meta", "nvidia", "intel", "amd",
        "tech", "technology", "digital", "online", "internet", "website", "platform",
        "mobile", "computer", "laptop", "device", "gadget", "innovation",
        "robot", "automation", "drone", "virtual reality", "vr", "ar", "metaverse"
    ],
    # Low-value tech keywords (weight 1) - only count if multiple present
    "low_value": [
        "data", "network", "system", "electronic", "digital", "smart", "tech"
    ]
}

EU_KEYWORDS = {
    "high_value": [
        "european union", "eu parliament", "european commission", "brexit", 
        "eurozone", "european central bank", "schengen", "eu summit",
        "strasbourg", "brussels parliament", "luxembourg court"
    ],
    "medium_value": [
        "europe", "european", "eu", "britain", "british", "uk", "england", "scotland",
        "france", "french", "germany", "german", "italy", "italian", "spain", "spanish",
        "netherlands", "belgium", "austria", "poland", "portugal", "greece", "ireland",
        "london", "paris", "berlin", "rome", "madrid", "amsterdam", "brussels"
    ],
    "low_value": [
        "euro", "european", "british", "french", "german", "italian", "spanish"
    ]
}

TRAVEL_KEYWORDS = {
    "high_value": [
        "tourism", "tourist", "vacation", "holiday", "travel ban", "travel advisory",
        "airport", "airline", "flight", "cruise", "hotel", "resort", "booking",
        "destination", "sightseeing", "backpacking", "safari", "expedition"
    ],
    "medium_value": [
        "travel", "trip", "journey", "visit", "visiting", "plane", "aircraft",
        "train", "railway", "ferry", "border", "customs", "immigration",
        "passport", "visa", "luggage", "baggage", "museum", "attraction"
    ],
    "low_value": [
        "visit", "trip", "plane", "train", "border"
    ]
}

CATEGORIES = {
    "Tech": TECH_KEYWORDS,
    "EU News": EU_KEYWORDS,
    "Travel": TRAVEL_KEYWORDS
}

STOP_WORDS = set(["the", "a", "an", "in", "on", "of", "for", "to", "with", "and", "or", "is", "are", "was", "were", "will", "would", "could", "should", "has", "have", "had"])

# Enhanced source-specific styling
SOURCE_THEMES = {
    "Reuters": {"color": "ff6600", "text_color": "ffffff", "icon": "📰"},
    "BBC News": {"color": "bb1919", "text_color": "ffffff", "icon": "📺"},
    "The Guardian": {"color": "052962", "text_color": "ffffff", "icon": "🗞️"},
    "TechCrunch": {"color": "00d084", "text_color": "000000", "icon": "💻"},
    "Associated Press": {"color": "0066cc", "text_color": "ffffff", "icon": "📄"},
    "Tech Reuters": {"color": "ff6600", "text_color": "ffffff", "icon": "🔧"},
    "EU Guardian": {"color": "003399", "text_color": "ffffff", "icon": "🇪🇺"},
    "Travel Guardian": {"color": "ff6b35", "text_color": "ffffff", "icon": "✈️"},
    "Tech Guardian": {"color": "052962", "text_color": "ffffff", "icon": "💻"}
}

# --- Helper Functions ---
def calculate_category_score(text, keywords_dict):
    """Calculate weighted score for a category"""
    text_lower = text.lower()
    total_score = 0
    
    # High-value keywords
    for keyword in keywords_dict.get("high_value", []):
        if keyword in text_lower:
            total_score += 5
            
    # Medium-value keywords  
    for keyword in keywords_dict.get("medium_value", []):
        if keyword in text_lower:
            total_score += 3
            
    # Low-value keywords (only count if multiple present)
    low_value_matches = 0
    for keyword in keywords_dict.get("low_value", []):
        if keyword in text_lower:
            low_value_matches += 1
    
    # Only add low-value score if multiple matches
    if low_value_matches >= 2:
        total_score += low_value_matches
    
    return total_score

def categorize_article(headline, summary, source):
    """Enhanced categorization with weighted scoring"""
    text = f"{headline} {summary}"
    
    # Source-based categorization (highest priority)
    source_lower = source.lower()
    if "techcrunch" in source_lower or "tech" in source_lower:
        print(f"Source-based categorization: {source} -> Tech")
        return "Tech"
    elif "eu" in source_lower or ("guardian" in source_lower and "europe" in source_lower):
        print(f"Source-based categorization: {source} -> EU News")
        return "EU News"  
    elif "travel" in source_lower:
        print(f"Source-based categorization: {source} -> Travel")
        return "Travel"
    
    # Calculate scores for each category
    category_scores = {}
    for category, keywords_dict in CATEGORIES.items():
        score = calculate_category_score(text, keywords_dict)
        category_scores[category] = score
    
    # Find the best category with minimum threshold
    best_category = max(category_scores, key=category_scores.get) if category_scores else "General"
    best_score = category_scores.get(best_category, 0)
    
    # Require minimum score for categorization
    MIN_SCORE_THRESHOLD = 5  # Increased threshold
    
    if best_score >= MIN_SCORE_THRESHOLD:
        print(f"Categorized '{headline[:40]}...' as {best_category} (score: {best_score})")
        return best_category
    
    print(f"Categorized '{headline[:40]}...' as General (highest score: {best_score}, below threshold)")
    return "General"

def create_themed_placeholder(source, headline, size="800x400"):
    """Create high-quality themed placeholder"""
    theme = SOURCE_THEMES.get(source, {
        "color": "1a1f26",
        "text_color": "8b949e", 
        "icon": "📰"
    })
    
    icon = theme["icon"]
    source_name = source.replace(" ", "+")
    
    return f"https://via.placeholder.com/{size}/{theme['color']}/{theme['text_color']}?text={icon}+{source_name}"

def get_image_from_entry(entry, source, headline):
    """Extract high-quality image URL from RSS entry"""
    
    # Method 1: Check media_content for largest available image
    if hasattr(entry, 'media_content') and entry.media_content:
        best_image = None
        best_size = 0
        
        for media in entry.media_content:
            url = media.get('url', '')
            if url and any(ext in url.lower() for ext in ['.jpg', '.jpeg', '.png', '.webp']):
                # Try to determine image quality
                width = media.get('width', 0)
                height = media.get('height', 0)
                if isinstance(width, str):
                    width = int(width) if width.isdigit() else 0
                if isinstance(height, str):
                    height = int(height) if height.isdigit() else 0
                
                size_score = width * height
                if size_score > best_size:
                    best_size = size_score
                    best_image = url
                elif best_image is None:
                    best_image = url
        
        if best_image:
            print(f"✅ Found high-quality media image: {best_image[:50]}...")
            return best_image
    
    # Method 2: Check media_thumbnail (but enhance it)
    if hasattr(entry, 'media_thumbnail') and entry.media_thumbnail:
        url = entry.media_thumbnail[0].get('url', '')
        if url:
            # Try to get a larger version by modifying URL patterns
            high_res_url = url.replace('_s.jpg', '_b.jpg').replace('thumbnail', 'large').replace('150x150', '800x600')
            print(f"✅ Found thumbnail (enhanced): {high_res_url[:50]}...")
            return high_res_url
    
    # Method 3: Parse HTML content for images
    content_to_search = []
    
    if hasattr(entry, 'content') and entry.content:
        for content in entry.content:
            content_to_search.append(content.get('value', ''))
    
    if hasattr(entry, 'summary'):
        content_to_search.append(entry.summary)
    
    if hasattr(entry, 'description'):
        content_to_search.append(entry.description)
    
    best_image = None
    for content in content_to_search:
        img_matches = re.findall(r'<img[^>]+src=["\']([^"\']+)["\'][^>]*>', content, re.IGNORECASE)
        for img_url in img_matches:
            if img_url and any(ext in img_url.lower() for ext in ['.jpg', '.jpeg', '.png', '.webp']):
                if any(size in img_url for size in ['large', 'big', '800', '1200', 'full']):
                    print(f"✅ Found high-res content image: {img_url[:50]}...")
                    return img_url
                elif best_image is None:
                    best_image = img_url
    
    if best_image:
        print(f"✅ Found content image: {best_image[:50]}...")
        return best_image
    
    # Method 4: Check enclosures
    if hasattr(entry, 'enclosures') and entry.enclosures:
        for enclosure in entry.enclosures:
            if enclosure.get('type', '').startswith('image/'):
                url = enclosure.get('href', '')
                if url:
                    print(f"✅ Found enclosure image: {url[:50]}...")
                    return url
    
    # Method 5: Create high-quality themed placeholder
    print(f"❌ No image found for '{headline[:30]}...', using high-quality placeholder")
    return create_themed_placeholder(source, headline, "800x400")

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
    
    print(f"🚀 Starting refined news fetch at {datetime.now()}")
    
    for source, url in NEWS_FEEDS.items():
        try:
            print(f"📡 Fetching from {source}...")
            feed = feedparser.parse(url)
            
            if not feed.entries:
                print(f"⚠️ No entries found for {source}")
                continue
            
            # Take more articles for category-specific feeds
            max_articles = 12 if any(cat in source.lower() for cat in ['tech', 'eu', 'travel']) else 8
            
            for entry in feed.entries[:max_articles]:
                try:
                    summary = entry.get('summary', entry.get('description', 'No summary available.'))
                    headline = entry.get('title', 'No title')
                    
                    # Enhanced image extraction
                    image_url = get_image_from_entry(entry, source, headline)
                    
                    # Refined categorization
                    category = categorize_article(headline, summary, source)
                    
                    article = {
                        "id": article_id, 
                        "source": source, 
                        "headline": headline, 
                        "link": entry.get('link', '#'), 
                        "published": entry.get("published", "N/A"),
                        "image_url": image_url,
                        "summary": summary, 
                        "explained_version": explain_with_ai(summary),
                        "hotness": random.randint(70, 100), 
                        "category": category,
                        "keywords": [kw.lower() for kw in re.findall(r'\b\w{4,}\b', headline.lower()) if kw.lower() not in STOP_WORDS]
                    }
                    
                    processed_news.append(article)
                    article_id += 1
                    
                except Exception as e:
                    print(f"❌ Error processing article from {source}: {e}")
                    continue
                    
        except Exception as e:
            print(f"❌ Error fetching from {source}: {e}")
            continue
    
    # Print categorization summary
    category_counts = Counter(article['category'] for article in processed_news)
    print(f"\n📊 Final Categorization Summary:")
    for category, count in category_counts.items():
        print(f"   {category}: {count} articles")
    
    # Save processed news
    try:
        with open(NEWS_FILE, "w") as f:
            json.dump({
                "articles": processed_news, 
                "last_updated": datetime.utcnow().isoformat()
            }, f, indent=4)
        
        print(f"✅ Refined news processing complete! {len(processed_news)} articles saved.")
        
    except Exception as e:
        print(f"❌ Error saving news file: {e}")

# --- API Endpoints ---
@app.on_event("startup")
def start_scheduler():
    scheduler = BackgroundScheduler(timezone="UTC")
    fetch_and_process_news()
    scheduler.add_job(fetch_and_process_news, 'interval', minutes=30)
    scheduler.start()
    print("🔄 Refined scheduler started - news will update every 30 minutes")

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
            
        return {"articles": articles, "categories": ["All", "Tech", "EU News", "Travel", "General"]}

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