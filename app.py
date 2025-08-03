import os
import feedparser
import json
import random
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from apscheduler.schedulers.background import BackgroundScheduler
from groq import Groq
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()

NEWS_FEEDS = {
    "New York Times": "http://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
    "BBC News": "http://feeds.bbci.co.uk/news/rss.xml",
    "The Guardian": "https://www.theguardian.com/world/rss",
    "TechCrunch": "https://techcrunch.com/feed/",
    "Reuters": "http://feeds.reuters.com/reuters/topNews",
}
NEWS_FILE = "news.json"
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# --- FastAPI App ---
app = FastAPI()

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
        return "AI is not configured. Please set the GROQ_API_KEY."
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a news analyst. Expand the following summary into a clear, easy-to-read explanation of 2-3 paragraphs. Focus on context, key players, and importance. Present it in clean markdown format."
                },
                {
                    "role": "user",
                    "content": f"Please explain this news summary: {summary}",
                }
            ],
            model="llama3-8b-8192",
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"Error calling Groq API: {e}")
        return "Could not generate AI explanation."


def fetch_and_process_news():
    processed_news = []
    article_id = 0
    for source, url in NEWS_FEEDS.items():
        feed = feedparser.parse(url)
        for entry in feed.entries[:5]:
            summary = entry.summary
            explained_version = explain_with_ai(summary)
            image_url = get_image_from_entry(entry)

            processed_news.append({
                "id": article_id,
                "source": source,
                "headline": entry.title,
                "link": entry.link,
                "published": entry.get("published", "N/A"),
                "image_url": image_url,
                "summary": summary,
                "explained_version": explained_version,
                "hotness": random.randint(70, 100) # Simulate a "hotness" score
            })
            article_id += 1
            
    with open(NEWS_FILE, "w") as f:
        json.dump(processed_news, f, indent=4)
    print("✅ News processing complete.")


@app.on_event("startup")
def start_scheduler():
    scheduler = BackgroundScheduler()
    fetch_and_process_news()
    scheduler.add_job(fetch_and_process_news, 'interval', minutes=30)
    scheduler.start()

@app.get("/api/news")
async def get_all_news():
    if not os.path.exists(NEWS_FILE):
        return {"error": "News is being processed. Please try again."}
    with open(NEWS_FILE, "r") as f:
        return json.load(f)

@app.get("/api/news/{article_id}")
async def get_article(article_id: int):
    if not os.path.exists(NEWS_FILE):
        raise HTTPException(status_code=404, detail="News file not found.")
    with open(NEWS_FILE, "r") as f:
        news_data = json.load(f)
        for article in news_data:
            if article['id'] == article_id:
                return article
    raise HTTPException(status_code=404, detail="Article not found.")

app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))