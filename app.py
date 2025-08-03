import os
import feedparser
import json
from fastapi import FastAPI
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
    """Extracts an image URL from an RSS feed entry."""
    if 'media_content' in entry and entry.media_content:
        return entry.media_content[0]['url']
    if 'links' in entry:
        for link in entry.links:
            if link.get('type', '').startswith('image/'):
                return link.href
    return None # Return None if no image is found

def explain_with_ai(summary):
    """Uses Groq AI to generate a more detailed explanation of a news summary."""
    if not os.environ.get("GROQ_API_KEY"):
        return "AI is not configured. Please set the GROQ_API_KEY."

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a news analyst. Your task is to take a brief news summary and expand it into a clear, easy-to-understand explanation of about 2-3 paragraphs. Focus on explaining the context, the key players, and why the story is important. Present the result in a clean, markdown-ready format."
            },
            {
                "role": "user",
                "content": f"Please explain this news summary: {summary}",
            }
        ],
        model="llama3-8b-8192",
    )
    return chat_completion.choices[0].message.content

def fetch_and_process_news():
    """Fetches news, processes it with AI, and stores it in a JSON file."""
    processed_news = []
    for source, url in NEWS_FEEDS.items():
        feed = feedparser.parse(url)
        # Limit to the latest 5 articles per feed to keep it fresh
        for entry in feed.entries[:5]:
            summary = entry.summary
            explained_version = explain_with_ai(summary)
            image_url = get_image_from_entry(entry)

            processed_news.append({
                "source": source,
                "headline": entry.title,
                "link": entry.link,
                "published": entry.get("published", "N/A"),
                "image_url": image_url,
                "explained_version": explained_version,
            })
    with open(NEWS_FILE, "w") as f:
        json.dump(processed_news, f, indent=4)
    print("✅ News processing complete.")


@app.on_event("startup")
def start_scheduler():
    """Starts the background scheduler to fetch news every 30 minutes."""
    scheduler = BackgroundScheduler()
    # Run once on startup
    fetch_and_process_news()
    # Then run every 30 minutes
    scheduler.add_job(fetch_and_process_news, 'interval', minutes=30)
    scheduler.start()

@app.get("/api/news")
async def get_news():
    """API endpoint to get the latest, AI-processed news."""
    if not os.path.exists(NEWS_FILE):
        return {"error": "News is being processed. Please try again in a moment."}
    with open(NEWS_FILE, "r") as f:
        return json.load(f)

# Mount the static directory to serve the frontend files
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))