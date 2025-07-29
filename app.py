# Optimized Fast & Accurate iTethr Bot
# File: app.py

import gradio as gr
import os
import chromadb
from sentence_transformers import SentenceTransformer
import ollama
import json
import yaml
from flask import Flask, request, jsonify
import threading
import time
import logging
from datetime import datetime
from typing import List, Tuple, Optional
import uuid
import hashlib

# Add at the top after imports
import signal
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add Railway-specific configurations
def setup_railway_config():
    """Configure app for Railway deployment"""
    if os.getenv('RAILWAY_ENVIRONMENT'):
        # Railway-specific settings
        os.environ['GRADIO_SHARE'] = 'false'
        os.environ['DEBUG'] = 'false'
        
        # Keep your optimized settings
        os.environ['MAX_TOKENS'] = '200'  # Your setting
        os.environ['TEMPERATURE'] = '0.01'  # Your setting
        
        # Railway port handling
        port = os.getenv('PORT', '7860')
        os.environ['GRADIO_SERVER_PORT'] = port
        
        logger.info(f"🚂 Railway deployment detected - Port: {port}")

# Add graceful shutdown handler
def signal_handler(sig, frame):
    logger.info('🛑 Graceful shutdown initiated...')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Update the main runner section:
if __name__ == "__main__":
    # Setup Railway configuration
    setup_railway_config()
    
    logger.info(f"🚀 Starting {bot.bot_name} v{bot.version} - Railway Mode")
    logger.info(f"⚡ Model: {bot.model_name} | Max Tokens: {bot.max_tokens}")
    
    # Get Railway environment variables
    port = int(os.getenv('PORT', '7860'))
    railway_url = os.getenv('RAILWAY_STATIC_URL', 'localhost')
    
    logger.info(f"🌐 Railway URL: {railway_url}")
    logger.info(f"🔌 Deployment Port: {port}")
    
    # Start health API in background
    try:
        api_thread = threading.Thread(target=run_api, daemon=True)
        api_thread.start()
        logger.info("🔗 Health API started")
    except Exception as e:
        logger.warning(f"API startup issue: {e}")
    
    # Launch optimized interface for Railway
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,  # Railway handles public URLs
        debug=False,
        show_error=True,
        prevent_thread_lock=False,
        quiet=True  # Reduce Railway logs
    )

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('itethr_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class OptimizediTethrBot:
    """Optimized iTethr Bot - Fast, Accurate, No Hallucination"""
    
    def __init__(self):
        self.version = "6.0.0"
        self.bot_name = "iTethr Assistant"
        
        # Optimized AI settings for speed and accuracy
        self.model_name = os.getenv('AI_MODEL', 'qwen2.5:1.5b')
        self.max_tokens = int(os.getenv('MAX_TOKENS', '200'))  # Reduced for speed
        self.temperature = float(os.getenv('TEMPERATURE', '0.01'))  # Very low to prevent hallucination
        
        # Setup bot
        self._setup_components()
        logger.info(f"🚀 {self.bot_name} v{self.version} optimized and ready!")
    
    def _setup_components(self):
        """Setup bot components"""
        try:
            # Load embeddings
            logger.info("Loading embeddings model...")
            self.embeddings = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Setup database
            self.client = chromadb.PersistentClient(path='./knowledge_base')
            self.collection = self.client.get_or_create_collection(name="docs")
            
            # Load documents
            self._load_all_documents()
            
            # Stats
            self.total_questions = 0
            
            logger.info("✅ All components optimized and ready!")
            
        except Exception as e:
            logger.error(f"Setup failed: {e}")
            raise
    
    def _load_all_documents(self):
        """Load all documents from documents folder"""
        docs_folder = './documents'
        
        if not os.path.exists(docs_folder):
            os.makedirs(docs_folder)
            logger.info(f"📁 Created documents folder: {docs_folder}")
            return
        
        # Clear and reload all documents
        try:
            self.collection.delete(where={})
            logger.info("🗑️ Cleared old documents")
        except:
            pass
        
        supported = ('.txt', '.md', '.py', '.js', '.json', '.yml', '.yaml', '.rst')
        loaded_count = 0
        total_chunks = 0
        
        for filename in os.listdir(docs_folder):
            if filename.endswith(supported) and not filename.startswith('.'):
                filepath = os.path.join(docs_folder, filename)
                
                try:
                    content = self._read_file(filepath)
                    if content and len(content.strip()) > 50:
                        chunks = self._create_chunks(content, filename)
                        
                        for i, chunk in enumerate(chunks):
                            embedding = self.embeddings.encode(chunk).tolist()
                            chunk_id = f"{filename}_{i}_{hashlib.md5(chunk.encode()).hexdigest()[:6]}"
                            
                            self.collection.add(
                                embeddings=[embedding],
                                documents=[chunk],
                                metadatas=[{
                                    "filename": filename, 
                                    "chunk_id": i,
                                    "source": filename
                                }],
                                ids=[chunk_id]
                            )
                        
                        loaded_count += 1
                        total_chunks += len(chunks)
                        logger.info(f"📄 Loaded: {filename} ({len(chunks)} chunks)")
                
                except Exception as e:
                    logger.error(f"Failed loading {filename}: {e}")
        
        logger.info(f"✅ Total: {loaded_count} documents, {total_chunks} chunks loaded")
    
    def _read_file(self, filepath: str) -> Optional[str]:
        """Read file content efficiently"""
        try:
            filename = os.path.basename(filepath)
            _, ext = os.path.splitext(filepath.lower())
            
            with open(filepath, 'r', encoding='utf-8') as f:
                if ext == '.json':
                    data = json.load(f)
                    return f"Document: {filename}\nContent:\n{json.dumps(data, indent=2)}"
                elif ext in ['.yml', '.yaml']:
                    data = yaml.safe_load(f)
                    return f"Document: {filename}\nContent:\n{yaml.dump(data, indent=2)}"
                else:
                    content = f.read()
                    return f"Document: {filename}\nContent:\n{content}"
                    
        except Exception as e:
            logger.error(f"Error reading {filepath}: {e}")
            return None
    
    def _create_chunks(self, content: str, filename: str) -> List[str]:
        """Create optimized text chunks"""
        # Smaller chunks for faster processing
        chunk_size = 250  # Reduced from 300
        overlap = 30      # Reduced overlap
        
        words = content.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words)
            if chunk_text.strip():
                chunks.append(chunk_text.strip())
        
        return chunks if chunks else [content]
    
    def _search_knowledge(self, question: str) -> List[str]:
        """Optimized search with strict relevance"""
        try:
            # Quick embedding
            question_embedding = self.embeddings.encode(question.lower()).tolist()
            
            results = self.collection.query(
                query_embeddings=[question_embedding],
                n_results=2,  # Reduced for speed
                include=['documents', 'distances', 'metadatas']
            )
            
            relevant_docs = []
            if results['documents'] and results['distances']:
                for doc, distance, metadata in zip(
                    results['documents'][0], 
                    results['distances'][0],
                    results['metadatas'][0]
                ):
                    # More lenient threshold but log for debugging
                    if distance < 1.5:
                        relevant_docs.append(doc)
                        logger.info(f"Found relevant content (distance: {distance:.2f}) from {metadata.get('filename', 'unknown')}")
            
            return relevant_docs
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
    
    def get_response(self, message: str) -> str:
        """Optimized response generation - Fast & Accurate"""
        start_time = time.time()
        self.total_questions += 1
        
        if not message.strip():
            return "Hi! Ask me anything about iTethr platform. I'll give you accurate answers based on the documentation."
        
        # Quick search
        relevant_docs = self._search_knowledge(message)
        
        if not relevant_docs:
            return """🧠 Uh-oh... my digital brain just blanked on that one!

That info's either top secret, lost in the Matrix, or my hamster-powered memory ran out of juice. 🐹⚡

**BUT here's what I *do* know without glitching:**

• iTethr platform overview and features  
• User authentication and sign-up processes 
• Community structure (Communities, Hubs, Rooms) 
• Bubble-based interface design
• I assistant capabilities (Aeono)  
• Technical implementation details
• iTethr iFeed functionality


Try asking about any of these — I'll respond faster than your group chat drama. 📲🔥"""

        # Use only the most relevant document for speed and accuracy
        context = relevant_docs[0]
        
        # Strict prompt to prevent hallucination
        prompt = f"""You are iTethr Assistant. Answer ONLY using the exact information from the documentation below.

DOCUMENTATION:
{context}

QUESTION: {message}

STRICT RULES:
- Use ONLY information directly stated in the documentation above
- Do not add any details not mentioned in the docs
- Do not make assumptions or expand beyond what's written
- If the documentation doesn't contain the answer, say "The documentation doesn't provide that specific information"
- Be direct and factual

ANSWER:"""
        
        try:
            # Optimized generation parameters
            result = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    'temperature': 0.01,    # Almost no creativity
                    'max_tokens': 200,      # Shorter for speed
                    'num_predict': 200,     # Match max_tokens
                    'top_p': 0.1,          # Very focused
                    'repeat_penalty': 1.0,  # No penalty
                    'num_ctx': 1024        # Smaller context window
                }
            )
            
            response = result['response'].strip()
            
            # Add source attribution
            response += f"\n\n*Based on iTethr documentation*"
            
            # Log response time
            response_time = time.time() - start_time
            logger.info(f"⚡ Response generated in {response_time:.2f}s")
            
            return response
            
        except Exception as e:
            logger.error(f"AI generation error: {e}")
            return f"I'm experiencing technical difficulties. Please try again.\n\nError: {str(e)}"
    
    def chat(self, message: str, history: List[Tuple[str, str]]) -> Tuple[str, List[Tuple[str, str]]]:
        """Optimized chat function"""
        if message.strip():
            response = self.get_response(message)
            history.append((message, response))
        return "", history

# Initialize optimized bot
bot = OptimizediTethrBot()

# Health API
app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "bot": bot.bot_name,
        "version": bot.version,
        "model": bot.model_name,
        "questions_answered": bot.total_questions,
        "timestamp": datetime.now().isoformat()
    })

def run_api():
    try:
        app.run(host='0.0.0.0', port=int(os.getenv('API_PORT', '5001')), debug=False)
    except Exception as e:
        logger.warning(f"API port issue: {e}")

# Fixed Gradio Interface
def create_interface():
    # Get team password from environment
    TEAM_PASSWORD = os.getenv('BOT_ACCESS_PASSWORD', 'itethr2024')
    
    def authenticate(password):
        """Check if password is correct"""
        if password.strip() == TEAM_PASSWORD:
            return (
                gr.update(visible=False),  # Hide login screen
                gr.update(visible=True),   # Show bot interface
                ""
            )
        else:
            return (
                gr.update(visible=True),   # Keep login screen
                gr.update(visible=False),  # Hide bot interface  
                "❌ Incorrect password. Please try again."
            )
    
    with gr.Blocks(
        title="iTethr Assistant - Optimized",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="blue"
        )
    ) as interface:
        
        # LOGIN SCREEN
        with gr.Column(visible=True) as login_screen:
            gr.Markdown("# 🔐 iTethr Assistant - Team Access")
            password_input = gr.Textbox(label="Team Password", type="password")
            login_btn = gr.Button("🚀 Access Bot", variant="primary")
            login_status = gr.Markdown("")
        
        # BOT INTERFACE - FIXED STRUCTURE
        with gr.Column(visible=False) as bot_interface:
            # Clean header
            gr.Markdown(f"""
            # 🤖 iTethr Assistant
            **Accurate, on-demand insights from iTethr docs — powered by AeonovX**
            
            *Optimized for speed and accuracy - v{bot.version}*
            """)
            
            # Main chat interface
            with gr.Row():
                with gr.Column():
                    chatbot = gr.Chatbot(
                        height=550,
                        label="💬 Chat with iTethr Assistant",
                        show_copy_button=True,
                        placeholder="Ask me anything about iTethr platform..."
                    )
                    
                    with gr.Row():
                        msg = gr.Textbox(
                            placeholder="Type your question about iTethr...",
                            label="",
                            scale=5,
                            max_lines=3
                        )
                        send = gr.Button("Send ⚡", variant="primary", scale=1)
            
            # Optimized quick suggestions
            gr.Markdown("### 💡 Quick Questions (Click to Ask)")
            with gr.Row():
                btn1 = gr.Button("What is iTethr platform?", size="sm", variant="secondary")
                btn2 = gr.Button("How does authentication work?", size="sm", variant="secondary")
                btn3 = gr.Button("Explain Communities and Hubs", size="sm", variant="secondary")
            
            with gr.Row():
                btn4 = gr.Button("What is the bubble interface?", size="sm", variant="secondary")
                btn5 = gr.Button("Tell me about Aeono AI", size="sm", variant="secondary")
                btn6 = gr.Button("What is iTethr iFeed?", size="sm", variant="secondary")
            
            # Simple instructions
            gr.Markdown("""
            ### ⚡ Built for Speed, Trained for Truth
            
            **Features:**
            • **Blazing-fast replies** – Like, caffeine-injected fast ☕⚡
            • **Seriously accurate** – No guesswork, no hallucinations, just facts
            • **One-click copy** – Hit that copy button and boom, info's yours  

            
            **Tips:**
            • Be specific – Vague questions confuse my circuits 🌀 
            • Add "explain" for deep dives and nerd-level clarity  
            • I only spill knowledge from official iTethr docs (no gossip here 😇)
            """)
            
            # Connect events with error handling - MOVED INSIDE bot_interface
            def safe_chat(message, history):
                try:
                    return bot.chat(message, history)
                except Exception as e:
                    logger.error(f"Chat error: {e}")
                    if message.strip():
                        history.append((message, f"Sorry, I encountered an error: {str(e)}"))
                    return "", history
            
            # Event connections
            send.click(safe_chat, [msg, chatbot], [msg, chatbot])
            msg.submit(safe_chat, [msg, chatbot], [msg, chatbot])
            
            # Connect suggestion buttons
            btn1.click(lambda: "What is iTethr platform?", outputs=msg)
            btn2.click(lambda: "How does authentication work?", outputs=msg)
            btn3.click(lambda: "Explain Communities and Hubs", outputs=msg)
            btn4.click(lambda: "What is the bubble interface?", outputs=msg)
            btn5.click(lambda: "Tell me about Aeono AI", outputs=msg)
            btn6.click(lambda: "What is iTethr iFeed?", outputs=msg)
        
        # CONNECT LOGIN - MOVED TO END
        login_btn.click(
            authenticate,
            [password_input],
            [login_screen, bot_interface, login_status]
        )
        
    return interface

# Main runner
if __name__ == "__main__":
    logger.info(f"🚀 Starting {bot.bot_name} v{bot.version} - Optimized Mode")
    logger.info(f"⚡ Model: {bot.model_name} | Max Tokens: {bot.max_tokens} | Temperature: {bot.temperature}")
    
    # Start health API in background
    try:
        api_thread = threading.Thread(target=run_api, daemon=True)
        api_thread.start()
        logger.info("🔗 Health API started on port 5001")
    except Exception as e:
        logger.warning(f"API startup issue: {e}")
    
    # Launch optimized interface
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv('PORT', '7860')),
        share=os.getenv('GRADIO_SHARE', 'false').lower() == 'true',
        debug=False,
        show_error=True
    )