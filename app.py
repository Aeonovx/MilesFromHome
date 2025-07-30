# iTethr Bot - AeonovX Team Version (Railway Compatible)
# File: app.py

import gradio as gr
import os
from sentence_transformers import SentenceTransformer
from groq import Groq
import json
import yaml
import time
import logging
from datetime import datetime
from typing import List, Tuple, Optional
import hashlib
import signal
import sys
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from team_members import AEONOVX_TEAM, USER_WELCOMES


# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("python-dotenv not available, using system environment variables")
    

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

class iTethrBot:
    """iTethr Bot - Powered by AeonovX"""
    
    def __init__(self):
        self.version = "7.1.0"
        self.bot_name = "AeonovX iTethr Assistant"
        
        # Setup Groq API
        self.groq_api_key = os.getenv('GROQ_API_KEY', '')
        if not self.groq_api_key:
            logger.warning("⚠️ GROQ_API_KEY not found. Add it to your Railway environment variables.")
            logger.info("Get your free API key from: https://console.groq.com/keys")
        
        self.groq_client = Groq(api_key=self.groq_api_key) if self.groq_api_key else None
        
        # In-memory storage for embeddings
        self.documents = []  # Store document chunks
        self.embeddings = []  # Store their embeddings
        self.metadata = []   # Store metadata
        
        # Setup bot
        self._setup_components()
        logger.info(f"🚀 {self.bot_name} v{self.version} ready!")
    
    def _setup_components(self):
        """Setup bot components"""
        try:
            # Load embeddings model (lightweight)
            logger.info("Loading embeddings model...")
            self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Load documents and create embeddings
            self._load_all_documents()
            
            self.total_questions = 0
            logger.info("✅ All components ready!")
            
        except Exception as e:
            logger.error(f"Setup failed: {e}")
            raise
    
    def _load_all_documents(self):
        """Load all documents and create embeddings in memory"""
        docs_folder = './documents'
        
        if not os.path.exists(docs_folder):
            os.makedirs(docs_folder)
            logger.info(f"📁 Created documents folder: {docs_folder}")
            return
        
        supported = ('.txt', '.md', '.py', '.js', '.json', '.yml', '.yaml', '.rst')
        loaded_count = 0
        total_chunks = 0
        
        # Clear existing data
        self.documents = []
        self.embeddings = []
        self.metadata = []
        
        for filename in os.listdir(docs_folder):
            if filename.endswith(supported) and not filename.startswith('.'):
                filepath = os.path.join(docs_folder, filename)
                
                try:
                    content = self._read_file(filepath)
                    if content and len(content.strip()) > 50:
                        chunks = self._create_chunks(content, filename)
                        
                        # Create embeddings for all chunks at once (faster)
                        chunk_embeddings = self.embeddings_model.encode(chunks)
                        
                        for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
                            self.documents.append(chunk)
                            self.embeddings.append(embedding)
                            self.metadata.append({
                                "filename": filename,
                                "chunk_id": i,
                                "source": filename
                            })
                        
                        loaded_count += 1
                        total_chunks += len(chunks)
                        logger.info(f"📄 Loaded: {filename} ({len(chunks)} chunks)")
                
                except Exception as e:
                    logger.error(f"Failed loading {filename}: {e}")
        
        # Convert embeddings to numpy array for faster similarity search
        if self.embeddings:
            self.embeddings = np.array(self.embeddings)
        
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
        chunk_size = 300  # Slightly larger for better context
        overlap = 50
        
        words = content.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words)
            if chunk_text.strip():
                chunks.append(chunk_text.strip())
        
        return chunks if chunks else [content]
    
    def _search_knowledge(self, question: str) -> List[str]:
        """Fast semantic search using cosine similarity"""
        try:
            if len(self.embeddings) == 0:
                return []
            
            # Create embedding for the question
            question_embedding = self.embeddings_model.encode([question])
            
            # Calculate cosine similarity with all document embeddings
            similarities = cosine_similarity(question_embedding, self.embeddings)[0]
            
            # Get top 3 most similar documents (increased for better context)
            top_indices = np.argsort(similarities)[-3:][::-1]  # Top 3, highest first
            
            relevant_docs = []
            for idx in top_indices:
                similarity = similarities[idx]
                if similarity > 0.25:  # Lowered threshold for more results
                    relevant_docs.append(self.documents[idx])
                    logger.info(f"Found relevant content (similarity: {similarity:.3f}) from {self.metadata[idx]['filename']}")
            
            return relevant_docs
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
    
    def _generate_groq_response(self, context: str, question: str) -> str:
        """Generate AI response using Groq API"""
        try:
            if not self.groq_client:
                return self._fallback_response(context, question)
            
            # Create a focused prompt for iTethr documentation
            prompt = f"""You are the AeonovX iTethr Assistant, an expert on the iTethr platform. Answer the user's question based ONLY on the provided documentation.

DOCUMENTATION:
{context}

USER QUESTION: {question}

INSTRUCTIONS:
- Answer based ONLY on the provided documentation
- Be helpful and conversational but accurate
- If the documentation doesn't contain the answer, say "I don't have that specific information in the iTethr documentation"
- Keep responses focused and under 300 words
- Be friendly and use a helpful tone
- You are built by AeonovX team for internal use

ANSWER:"""

            # Use Groq API with fast Llama model
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model="llama3-8b-8192",  # Fast and smart Llama model
                temperature=0.1,  # Low temperature for factual responses
                max_tokens=400,
                top_p=0.9
            )
            
            response = chat_completion.choices[0].message.content.strip()
            return response if response else self._fallback_response(context, question)
            
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            return self._fallback_response(context, question)
    
    def _fallback_response(self, context: str, question: str) -> str:
        """Fallback response when Groq API fails"""
        # Simple keyword matching fallback
        question_lower = question.lower()
        
        if "community" in question_lower or "hub" in question_lower:
            return "iTethr has a hierarchical community structure with Communities, Hubs, and Rooms. Communities can contain multiple Hubs, and Hubs contain Rooms for specific discussions."
        elif "authentication" in question_lower or "sign up" in question_lower:
            return "iTethr supports multiple sign-up methods: Google OAuth, Apple OAuth, and traditional email/password registration with a simplified 3-step onboarding process."
        elif "bubble" in question_lower or "interface" in question_lower:
            return "iTethr uses a revolutionary bubble-based interface instead of traditional navigation bars. Users interact with floating animated bubbles representing Communities, Loops, and contacts."
        elif "aeono" in question_lower or "ai" in question_lower:
            return "Aeono is iTethr's integrated AI assistant designed to help users connect with peers, find communities, and navigate the platform efficiently."
        elif "ifeed" in question_lower or "feed" in question_lower:
            return "iTethr iFeed is the global content stream where public posts, announcements, and Loop discussions are surfaced, functioning like a fusion of Reddit and Twitter."
        else:
            # Return relevant snippet from context
            sentences = context.split('. ')
            for sentence in sentences[:3]:
                if any(word in sentence.lower() for word in question_lower.split()):
                    return sentence.strip() + "."
            
            return "I can help you with iTethr platform features, communities, authentication, and more. Try asking about specific topics like 'What is iTethr?' or 'How do communities work?'"
    
    def get_response(self, message: str) -> str:
        """Get response from bot"""
        start_time = time.time()
        self.total_questions += 1
        
        if not message.strip():
            return "Hi! Ask me anything about iTethr platform. I'll give you accurate answers based on the documentation."
        
        # Search knowledge base
        relevant_docs = self._search_knowledge(message)
        
        if not relevant_docs:
            return """🧠 Hmm, I couldn't find specific information about that in the iTethr documentation!

**Here's what I can definitely help you with:**

• **iTethr platform overview** - What is iTethr and how it works
• **Community structure** - Communities, Hubs, and Rooms explained  
• **User authentication** - Sign-up processes and account management
• **Bubble interface** - The unique UI design and navigation
• **Aeono AI assistant** - Built-in AI features and capabilities
• **iFeed functionality** - Global content streams and social features

Try asking about any of these topics! 🚀"""

        # Combine multiple relevant documents for better context
        context = "\n\n".join(relevant_docs[:2])  # Use top 2 results
        
        # Generate response using Groq API
        response = self._generate_groq_response(context, message)
        response += f"\n\n*Based on iTethr documentation*"
        
        # Log response time
        response_time = time.time() - start_time
        logger.info(f"⚡ Response generated in {response_time:.2f}s")
        
        return response
    
    def chat(self, message: str, history: List[Tuple[str, str]]) -> Tuple[str, List[Tuple[str, str]]]:
        """Chat function for Gradio"""
        if message.strip():
            response = self.get_response(message)
            history.append((message, response))
        return "", history

# Initialize bot
bot = iTethrBot()

def authenticate(name, password):
    """Authenticate AeonovX team members - both name and password must match"""
    
    if name in AEONOVX_TEAM and AEONOVX_TEAM[name]["password"] == password:
        welcome_msg = USER_WELCOMES.get(name, f"Welcome {name}! 🚀")
        role = AEONOVX_TEAM[name]["role"]
        full_welcome = f"{welcome_msg}\n\n**Role:** {role}\n**Access Level:** AeonovX Team Member"
        
        return (
            gr.update(visible=False),  # Hide login
            gr.update(visible=True),   # Show bot
            full_welcome
        )
    else:
        return (
            gr.update(visible=True),   # Keep login visible
            gr.update(visible=False),  # Hide bot
            "❌ Invalid AeonovX credentials. Both name and password must be correct."
        )

def create_interface():
    """Create Gradio interface"""
    
    with gr.Blocks(
        title="AeonovX iTethr Assistant",
        theme=gr.themes.Soft(primary_hue="blue")
    ) as interface:
        
        # LOGIN SCREEN
        with gr.Column(visible=True) as login_screen:
            gr.Markdown("""
            # 🏢 AeonovX iTethr Assistant
            **Internal Team Access Only - Authorized Personnel Required**
            
            *Secure access for AeonovX team members and authorized personnel*
            """)
            
            name_input = gr.Textbox(
                label="Full Name", 
                placeholder="John Smith",
                info="Enter your full name as registered in AeonovX team database"
            )
            password_input = gr.Textbox(
                label="Password", 
                type="password",
                placeholder="Enter your team password",
                info="Use your assigned AeonovX team password"
            )
            login_btn = gr.Button("🔐 Access AeonovX Assistant", variant="primary", size="lg")
            login_status = gr.Markdown("")
        
        # BOT INTERFACE
        with gr.Column(visible=False) as bot_interface:
            # Header with team branding
            gr.Markdown(f"""
            # 🚀 AeonovX iTethr Assistant
            **Internal Knowledge Base - Powered by AeonovX Intelligence**
            
            *Lightning fast responses - v{bot.version} | Team Edition*
            """)
            
            chatbot = gr.Chatbot(
                height=550,
                label="💬 Chat with AeonovX iTethr Assistant",
                show_copy_button=True,
                avatar_images=("👤", "🤖")
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Ask me about iTethr platform, projects, or team resources...",
                    label="",
                    scale=5,
                    max_lines=3
                )
                send = gr.Button("Send ⚡", variant="primary", scale=1)
            
            # Quick suggestions
            gr.Markdown("### 💡 Quick Team Questions")
            with gr.Row():
                btn1 = gr.Button("What is iTethr?", size="sm")
                btn2 = gr.Button("Community structure?", size="sm")
                btn3 = gr.Button("Authentication process?", size="sm")
            
            with gr.Row():
                btn4 = gr.Button("Bubble interface design?", size="sm")
                btn5 = gr.Button("Aeono AI features?", size="sm")
                btn6 = gr.Button("iFeed functionality?", size="sm")
            
            gr.Markdown("""
            ### ⚡ Powered by AeonovX Intelligence
            
            **Team Features:**
            • **Ultra-fast responses** – Groq-powered AI inference ⚡
            • **Smart understanding** – Advanced Llama models for precise answers
            • **Semantic search** – Finds relevant info with natural language
            • **Always current** – Based on latest iTethr documentation
            • **Team-optimized** – Built specifically for AeonovX workflow
            • **Secure access** – Protected team-only environment 🔒
                        
            **Available 24/7 for the AeonovX team** - Your AI-powered knowledge companion.
            """)
            
            def safe_chat(message, history):
                try:
                    return bot.chat(message, history)
                except Exception as e:
                    logger.error(f"Chat error: {e}")
                    if message.strip():
                        history.append((message, f"Sorry, I encountered an error: {str(e)}"))
                    return "", history
            
            # Connect events
            send.click(safe_chat, [msg, chatbot], [msg, chatbot])
            msg.submit(safe_chat, [msg, chatbot], [msg, chatbot])
            
            btn1.click(lambda: "What is iTethr platform?", outputs=msg)
            btn2.click(lambda: "Explain iTethr community structure", outputs=msg)
            btn3.click(lambda: "How does iTethr authentication work?", outputs=msg)
            btn4.click(lambda: "What is the bubble interface?", outputs=msg)
            btn5.click(lambda: "Tell me about Aeono AI assistant", outputs=msg)
            btn6.click(lambda: "What is iTethr iFeed?", outputs=msg)
        
        # Connect login with both inputs
        login_btn.click(
            authenticate,
            [name_input, password_input],
            [login_screen, bot_interface, login_status]
        )
    
    # Health check
    @interface.app.get("/health")
    async def health_check():
        from fastapi.responses import JSONResponse
        return JSONResponse(content={
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "service": "aeonovx-itethr-assistant",
            "version": bot.version,
            "documents_loaded": len(bot.documents),
            "groq_api_configured": bool(bot.groq_api_key),
            "team_members_active": len(AEONOVX_TEAM)
        }, status_code=200)
    
    return interface

def setup_railway_config():
    """Configure for Railway"""
    port = int(os.getenv('PORT', '7860'))
    logger.info(f"🔌 Using port: {port}")
    return port

# Graceful shutdown
def signal_handler(sig, frame):
    logger.info('🛑 Shutting down AeonovX Assistant...')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

if __name__ == "__main__":
    try:
        logger.info(f"🚀 Starting {bot.bot_name} v{bot.version}")
        logger.info(f"👥 Team members configured: {len(AEONOVX_TEAM)}")
        
        interface = create_interface()
        port = setup_railway_config()

        interface.launch(
            server_name="0.0.0.0",
            server_port=port,
            share=False,
            debug=False,
            show_error=True,
            prevent_thread_lock=False,
            quiet=os.getenv('RAILWAY_ENVIRONMENT') == 'production'
        )
    except Exception as e:
        logger.error(f"Failed to start: {e}")
        sys.exit(1)