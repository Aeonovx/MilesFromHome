# iTethr Bot - ChromaDB Free Version (Railway Compatible)
# File: app.py

import gradio as gr
import os
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
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
    """iTethr Bot - Smart Search without ChromaDB"""
    
    def __init__(self):
        self.version = "6.2.0"
        self.bot_name = "iTethr Assistant"
        
        # AI settings
        self.max_tokens = int(os.getenv('MAX_TOKENS', '200'))
        self.temperature = float(os.getenv('TEMPERATURE', '0.1'))
        
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
            
            # Setup AI model (lightweight)
            logger.info("Loading AI model...")
            self._setup_ai_model()
            
            self.total_questions = 0
            logger.info("✅ All components ready!")
            
        except Exception as e:
            logger.error(f"Setup failed: {e}")
            raise
    
    def _setup_ai_model(self):
        """Setup lightweight AI model"""
        try:
            # Use a small, fast model that works well on CPU
            model_name = "microsoft/DialoGPT-small"
            
            logger.info(f"Loading {model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            
            # Add padding token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info("✅ AI model loaded successfully")
            
        except Exception as e:
            logger.error(f"AI model setup failed: {e}")
            # Fallback to no AI
            self.model = None
            self.tokenizer = None
    
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
        chunk_size = 250
        overlap = 30
        
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
            
            # Get top 2 most similar documents
            top_indices = np.argsort(similarities)[-2:][::-1]  # Top 2, highest first
            
            relevant_docs = []
            for idx in top_indices:
                similarity = similarities[idx]
                if similarity > 0.3:  # Threshold for relevance
                    relevant_docs.append(self.documents[idx])
                    logger.info(f"Found relevant content (similarity: {similarity:.3f}) from {self.metadata[idx]['filename']}")
            
            return relevant_docs
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
    
    def _generate_response(self, context: str, question: str) -> str:
        """Generate AI response using Hugging Face model"""
        try:
            if not self.model or not self.tokenizer:
                return self._fallback_response(context, question)
            
            # Create a simple prompt
            prompt = f"Based on this iTethr documentation:\n\n{context[:500]}\n\nQuestion: {question}\nAnswer:"
            
            # Tokenize
            inputs = self.tokenizer.encode(prompt, return_tensors='pt', truncate=True, max_length=512)
            
            # Generate response
            import torch
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 100,
                    temperature=self.temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part
            if "Answer:" in response:
                response = response.split("Answer:")[-1].strip()
            
            return response[:300] if response else self._fallback_response(context, question)
            
        except Exception as e:
            logger.error(f"AI generation error: {e}")
            return self._fallback_response(context, question)
    
    def _fallback_response(self, context: str, question: str) -> str:
        """Fallback response when AI fails"""
        # Simple keyword matching fallback
        question_lower = question.lower()
        context_lower = context.lower()
        
        if "community" in question_lower or "hub" in question_lower:
            return "iTethr has a hierarchical community structure with Communities, Hubs, and Rooms. Communities can contain multiple Hubs, and Hubs contain Rooms for specific discussions."
        elif "authentication" in question_lower or "sign up" in question_lower:
            return "iTethr supports multiple sign-up methods: Google OAuth, Apple OAuth, and traditional email/password registration with a simplified 3-step onboarding process."
        elif "bubble" in question_lower or "interface" in question_lower:
            return "iTethr uses a revolutionary bubble-based interface instead of traditional navigation bars. Users interact with floating animated bubbles representing Communities, Loops, and contacts."
        elif "aeono" in question_lower or "ai" in question_lower:
            return "Aeono is iTethr's integrated AI assistant designed to help users connect with peers, find communities, and navigate the platform efficiently."
        else:
            # Return relevant snippet from context
            sentences = context.split('. ')
            for sentence in sentences[:3]:
                if any(word in sentence.lower() for word in question_lower.split()):
                    return sentence.strip() + "."
            
            return "Based on the iTethr documentation, I can help you with platform features, communities, authentication, and AI assistance."
    
    def get_response(self, message: str) -> str:
        """Get response from bot"""
        start_time = time.time()
        self.total_questions += 1
        
        if not message.strip():
            return "Hi! Ask me anything about iTethr platform. I'll give you accurate answers based on the documentation."
        
        # Search knowledge base
        relevant_docs = self._search_knowledge(message)
        
        if not relevant_docs:
            return """🧠 Uh-oh... my digital brain just blanked on that one!

That info's either top secret, lost in the Matrix, or my hamster-powered memory ran out of juice. 🐹⚡

**BUT here's what I *do* know without glitching:**

• iTethr platform overview and features  
• User authentication and sign-up processes 
• Community structure (Communities, Hubs, Rooms) 
• Bubble-based interface design
• AI assistant capabilities (Aeono)  
• Technical implementation details
• iTethr iFeed functionality

Try asking about any of these — I'll respond faster than your group chat drama. 📲🔥"""

        # Use the most relevant document
        context = relevant_docs[0]
        
        # Generate response
        response = self._generate_response(context, message)
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

def create_interface():
    """Create Gradio interface"""
    
    with gr.Blocks(
        title="iTethr Assistant",
        theme=gr.themes.Soft(primary_hue="blue")
    ) as interface:
        
        # Skip login - go straight to bot
        gr.Markdown(f"""
        # 🤖 iTethr Assistant
        **Accurate insights from iTethr docs — powered by Semantic Search**
        
        *Fast and reliable - v{bot.version}*
        """)
        
        chatbot = gr.Chatbot(
            height=550,
            label="💬 Chat with iTethr Assistant",
            show_copy_button=True
        )
        
        with gr.Row():
            msg = gr.Textbox(
                placeholder="Type your question about iTethr...",
                label="",
                scale=5,
                max_lines=3
            )
            send = gr.Button("Send ⚡", variant="primary", scale=1)
        
        # Quick suggestions
        gr.Markdown("### 💡 Quick Questions")
        with gr.Row():
            btn1 = gr.Button("What is iTethr?", size="sm")
            btn2 = gr.Button("Community structure?", size="sm")
            btn3 = gr.Button("How to sign up?", size="sm")
        
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
    
    # Health check
    @interface.app.get("/health")
    async def health_check():
        from fastapi.responses import JSONResponse
        return JSONResponse(content={
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "service": "itethr-bot",
            "version": bot.version,
            "documents_loaded": len(bot.documents)
        }, status_code=200)
    
    return interface

def setup_railway_config():
    """Configure for Railway"""
    port = int(os.getenv('PORT', '7860'))
    logger.info(f"🔌 Using port: {port}")
    return port

# Graceful shutdown
def signal_handler(sig, frame):
    logger.info('🛑 Shutting down...')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

if __name__ == "__main__":
    try:
        logger.info(f"🚀 Starting {bot.bot_name} v{bot.version}")
        
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