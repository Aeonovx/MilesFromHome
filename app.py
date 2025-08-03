import os
import json
import random
import re
from collections import Counter
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from apscheduler.schedulers.background import BackgroundScheduler
import feedparser
from groq import Groq
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()
app = FastAPI()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# --- Constants ---
NEWS_FILE = "news.json"
SUBSCRIBERS_FILE = "subscribers.json"
NEWS_FEEDS = {
    "New York Times": "http://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
    "BBC News": "http://feeds.bbci.co.uk/news/rss.xml",
    "The Guardian": "https://www.theguardian.com/world/rss",
    "TechCrunch": "https://techcrunch.com/feed/",
    "Reuters": "http://feeds.reuters.com/reuters/topNews",
}
STOP_WORDS = set(["the", "a", "an", "in", "on", "of", "for", "to", "with", "and", "or", "is", "are", "was", "were"])

# --- Helper Functions ---
def get_image_from_entry(entry):
    if 'media_content' in entry and entry.media_content:
        return entry.media_content[0]['url']
    if 'links' in entry:
        for link in entry.links:
            if link.get('type', '').startswith('image/'):
                return link.href
    return None

def explain_with_ai(summary):
    if not os.environ.get("GROQ_API_KEY"):
        return "AI is not configured."
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a news analyst. Expand the summary into a clear, 2-3 paragraph explanation. Focus on context, key players, and importance. Use clean markdown."},
                {"role": "user", "content": f"Explain this: {summary}"}
            ],
            model="llama3-8b-8192",
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return "Could not generate AI explanation."

def find_trending_topics(news_data):
    all_text = " ".join([article['headline'] for article in news_data])
    words = re.findall(r'\b\w+\b', all_text.lower())
    filtered_words = [word for word in words if word not in STOP_WORDS and len(word) > 3]
    return [word for word, count in Counter(filtered_words).most_common(5)]

# --- Core News Fetching ---
def fetch_and_process_news():
    processed_news = []
    article_id = 0
    for source, url in NEWS_FEEDS.items():
        feed = feedparser.parse(url)
        for entry in feed.entries[:7]: # Fetch more articles per feed
            summary = entry.get('summary', 'No summary available.')
            explained_version = explain_with_ai(summary)
            
            processed_news.append({
                "id": article_id,
                "source": source,
                "headline": entry.title,
                "link": entry.link,
                "published": entry.get("published", "N/A"),
                "image_url": get_image_from_entry(entry),
                "summary": summary,
                "explained_version": explained_version,
                "hotness": random.randint(70, 100),
                "keywords": [kw.lower() for kw in re.findall(r'\b\w{4,}\b', entry.title.lower()) if kw.lower() not in STOP_WORDS]
            })
            article_id += 1
            
    trending_topics = find_trending_topics(processed_news)
    
    with open(NEWS_FILE, "w") as f:
        json.dump({"articles": processed_news, "trending": trending_topics}, f, indent=4)
    print("✅ News and trending topics processing complete.")

# --- API Endpoints ---
@app.on_event("startup")
def start_scheduler():
    scheduler = BackgroundScheduler()
    fetch_and_process_news()
    scheduler.add_job(fetch_and_process_news, 'interval', minutes=30)
    scheduler.start()

@app.get("/api/news")
async def get_news(search: str = None, source: str = None):
    if not os.path.exists(NEWS_FILE):
        raise HTTPException(status_code=503, detail="News is being processed.")
    with open(NEWS_FILE, "r") as f:
        data = json.load(f)
        articles = data["articles"]
        if search:
            articles = [a for a in articles if search.lower() in a['headline'].lower()]
        if source:
            articles = [a for a in articles if a['source'] == source]
        return {"articles": articles, "trending": data["trending"], "sources": list(NEWS_FEEDS.keys())}

@app.get("/api/news/{article_id}")
async def get_article(article_id: int):
    if not os.path.exists(NEWS_FILE):
        raise HTTPException(status_code=404, detail="News file not found.")
    with open(NEWS_FILE, "r") as f:
        data = json.load(f)
        articles = data["articles"]
        target_article = next((a for a in articles if a['id'] == article_id), None)
        if not target_article:
            raise HTTPException(status_code=404, detail="Article not found.")
        
        # Find related articles
        related = []
        for a in articles:
            if a['id'] != target_article['id'] and any(kw in a['keywords'] for kw in target_article['keywords']):
                related.append(a)
            if len(related) == 3:
                break
        
        return {"article": target_article, "related": related}

@app.post("/api/subscribe")
async def subscribe(request: Request):
    data = await request.json()
    email = data.get("email")
    if not email:
        raise HTTPException(status_code=400, detail="Email is required.")
        
    subscribers = []
    if os.path.exists(SUBSCRIBERS_FILE):
        with open(SUBSCRIBERS_FILE, "r") as f:
            subscribers = json.load(f)
    
    if email not in subscribers:
        subscribers.append(email)
        with open(SUBSCRIBERS_FILE, "w") as f:
            json.dump(subscribers, f)
            
    return {"message": "Subscription successful!"}

@app.get("/api/stats")
async def get_stats():
    if not os.path.exists(NEWS_FILE):
        return {"error": "No data available."}
    with open(NEWS_FILE, "r") as f:
        news_data = json.load(f)["articles"]
    
    subscribers_count = 0
    if os.path.exists(SUBSCRIBERS_FILE):
        with open(SUBSCRIBERS_FILE, "r") as f:
            subscribers_count = len(json.load(f))

    return {
        "total_articles": len(news_data),
        "articles_per_source": dict(Counter(a['source'] for a in news_data)),
        "subscribers": subscribers_count
    }

app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))