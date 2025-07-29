# Production iTethr Bot - Company Ready
# File: app.py

import gradio as gr
import os
import chromadb
from sentence_transformers import SentenceTransformer
import ollama
import json
import yaml
from flask import Flask, request, jsonify
import pywhatkit as kit
import threading
import time
import logging
from datetime import datetime
from typing import List, Tuple, Optional
import uuid
import hashlib

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('itethr_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class iTethrBot:
    """Production-ready iTethr Documentation Bot for AeonovX"""
    
    def __init__(self):
        """Initialize the bot with production configurations"""
        self.version = "1.0.0"
        self.company = "AeonovX"
        self.bot_name = "iTethr Assistant"
        self.session_id = uuid.uuid4().hex[:8]
        
        # Production settings
        self.max_tokens = int(os.getenv('MAX_TOKENS', '300'))
        self.temperature = float(os.getenv('TEMPERATURE', '0.3'))
        self.model_name = os.getenv('AI_MODEL', 'tinyllama:1.1b')
        
        # Initialize components
        self._setup_components()
        
        logger.info(f"🚀 {self.bot_name} v{self.version} initialized (Session: {self.session_id})")
    
    def _setup_components(self):
        """Setup all bot components"""
        try:
            # Setup embeddings
            logger.info("Loading embeddings model...")
            self.embeddings = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Setup database
            db_path = os.getenv('DATABASE_PATH', './itethr_knowledge_base')
            self.client = chromadb.PersistentClient(path=db_path)
            self.collection = self.client.get_or_create_collection(
                name="itethr_docs",
                metadata={
                    "description": "iTethr documentation knowledge base",
                    "version": self.version,
                    "created": datetime.now().isoformat()
                }
            )
            
            # Setup suggestions
            self.suggestions = [
                "What is iTethr platform?",
                "How does user sign-up work?",
                "What is Aeono AI assistant?", 
                "Difference between Community and Hub?",
                "How to deploy iTethr?",
                "What are the main UI components?",
                "How does the marketplace work?",
                "Explain user authentication system",
                "What are Loops in iTethr?",
                "How do Spaces work?"
            ]
            
            # Load documents
            self.load_documents()
            
            # Usage tracking
            self.usage_stats = {
                "total_questions": 0,
                "successful_responses": 0,
                "failed_responses": 0,
                "documents_loaded": self.collection.count(),
                "session_start": datetime.now().isoformat()
            }
            
            logger.info("✅ All components initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize components: {e}")
            raise
    
    def read_file_content(self, filepath: str) -> Optional[str]:
        """Read different file types with better error handling"""
        try:
            # Get file extension
            _, ext = os.path.splitext(filepath.lower())
            
            if ext == '.json':
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return f"JSON Configuration:\n{json.dumps(data, indent=2)}"
            
            elif ext in ['.yml', '.yaml']:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                    return f"YAML Configuration:\n{yaml.dump(data, indent=2)}"
            
            elif ext in ['.py', '.js', '.ts', '.jsx', '.tsx']:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    return f"Code File ({ext}):\n{content}"
            
            else:  # .txt, .md, .rst, etc.
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    return content
                    
        except UnicodeDecodeError:
            logger.warning(f"Unicode decode error for {filepath}, trying with latin-1")
            try:
                with open(filepath, 'r', encoding='latin-1') as f:
                    return f.read()
            except Exception as e:
                logger.error(f"Failed to read {filepath} with latin-1: {e}")
                return None
        except Exception as e:
            logger.error(f"Error reading {filepath}: {e}")
            return None
    
    def load_documents(self):
        """Load all documents from documents folder with production features"""
        docs_folder = os.getenv('DOCS_FOLDER', './documents')
        
        if not os.path.exists(docs_folder):
            os.makedirs(docs_folder)
            logger.warning(f"Created documents folder: {docs_folder}")
            
            # Create sample documentation
            self._create_sample_docs(docs_folder)
            return
        
        # Supported file types for documentation
        supported_files = (
            '.txt', '.md', '.rst',           # Documentation
            '.py', '.js', '.ts', '.jsx', '.tsx',  # Code files
            '.json', '.yml', '.yaml',        # Configuration
            '.env', '.config'                # Environment files
        )
        
        loaded_count = 0
        total_chunks = 0
        failed_files = []
        
        for filename in os.listdir(docs_folder):
            if filename.endswith(supported_files) and not filename.startswith('.'):
                filepath = os.path.join(docs_folder, filename)
                
                try:
                    content = self.read_file_content(filepath)
                    if content and len(content.strip()) > 50:  # Only process substantial content
                        chunks = self.chunk_text(content, filename)
                        
                        # Store chunks with metadata
                        for i, chunk in enumerate(chunks):
                            embedding = self.embeddings.encode(chunk).tolist()
                            
                            # Create unique ID
                            chunk_id = f"{filename}_{i}_{hashlib.md5(chunk.encode()).hexdigest()[:8]}"
                            
                            self.collection.add(
                                embeddings=[embedding],
                                documents=[chunk],
                                metadatas=[{
                                    "filename": filename,
                                    "chunk_index": i,
                                    "file_type": os.path.splitext(filename)[1],
                                    "loaded_at": datetime.now().isoformat()
                                }],
                                ids=[chunk_id]
                            )
                        
                        loaded_count += 1
                        total_chunks += len(chunks)
                        logger.info(f"📄 Loaded: {filename} ({len(chunks)} chunks)")
                    else:
                        logger.warning(f"⚠️ Skipped {filename}: insufficient content")
                        
                except Exception as e:
                    failed_files.append(filename)
                    logger.error(f"❌ Failed to load {filename}: {e}")
        
        # Log summary
        logger.info(f"✅ Document loading complete:")
        logger.info(f"   - Successfully loaded: {loaded_count} files")
        logger.info(f"   - Total chunks created: {total_chunks}")
        if failed_files:
            logger.warning(f"   - Failed files: {failed_files}")
    
    def _create_sample_docs(self, docs_folder: str):
        """Create sample documentation for demonstration"""
        sample_doc = f"""# iTethr Platform Documentation

## Overview
iTethr is a next-generation community infrastructure platform designed for international students, organizations, creators, and global users.

## Key Features
- Community building tools
- Real-time social engagement
- AI-powered assistance with Aeono
- Marketplace for services
- Multi-layer organizational structure

## User Authentication
- Sign up with Google (OAuth2)
- Sign up with Apple (OAuth2)  
- Email + Password registration

## Community Structure
- Community Plus: Multi-layered for universities and large organizations
- Hub: Single-layer community groups
- Rooms: Granular units within Hubs

## AI Assistant - Aeono
Aeono helps users connect with peers, find communities, and navigate the platform efficiently.

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Bot Version: {self.version}
"""
        
        sample_path = os.path.join(docs_folder, 'itethr-sample-docs.txt')
        with open(sample_path, 'w', encoding='utf-8') as f:
            f.write(sample_doc)
        
        logger.info(f"📝 Created sample documentation: {sample_path}")
    
    def chunk_text(self, content: str, filename: str) -> List[str]:
        """Enhanced text chunking with better context preservation"""
        # Adaptive chunk size based on file type
        if filename.endswith(('.py', '.js', '.ts')):
            chunk_size = 800  # Larger chunks for code
            overlap = 100
        else:
            chunk_size = 600  # Standard for documentation
            overlap = 80
        
        words = content.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            if chunk_text.strip():
                # Add source metadata to chunk
                formatted_chunk = f"Document: {filename}\n\n{chunk_text.strip()}"
                chunks.append(formatted_chunk)
        
        return chunks
    
    def search_docs(self, query: str, n_results: int = 3) -> List[str]:
        """Enhanced document search with better relevance"""
        try:
            query_embedding = self.embeddings.encode(query).tolist()
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=['documents', 'metadatas']
            )
            
            # Return documents if found
            return results['documents'][0] if results['documents'] else []
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
    
    def get_response(self, message: str) -> str:
        """Enhanced response generation with better error handling"""
        start_time = time.time()
        self.usage_stats["total_questions"] += 1
        
        try:
            if not message.strip():
                return "Please ask me something about iTethr! You can use the quick questions below for examples."
            
            # Search for relevant documents
            relevant_docs = self.search_docs(message, n_results=3)
            
            if not relevant_docs:
                self.usage_stats["failed_responses"] += 1
                return """I don't have specific information about that in the documentation. 

Try asking about:
• iTethr platform features
• User authentication and sign-up
• Community and Hub differences  
• Aeono AI assistant
• Marketplace functionality
• Deployment procedures

Or check the quick questions below!"""
            
            # Create context from relevant documents
            context = "\n\n---\n\n".join(relevant_docs[:2])  # Use top 2 most relevant
            
            # Enhanced prompt for better responses
            prompt = f"""You are iTethr Assistant, an AI helper for the iTethr platform development team at AeonovX.

Documentation Context:
{context}

User Question: {message}

Instructions:
- Provide a clear, helpful answer based on the documentation
- Keep responses concise but informative (2-3 paragraphs max)
- If it's about technical implementation, include relevant details
- Be friendly and professional
- If you're not sure about something, say so

Answer:"""

            try:
                # Generate response using AI model
                result = ollama.generate(
                    model=self.model_name,
                    prompt=prompt,
                    options={
                        'temperature': self.temperature,
                        'max_tokens': self.max_tokens,
                        'top_p': 0.9
                    }
                )
                
                response = result['response'].strip()
                
                # Add helpful footer
                response += f"\n\n💡 *Need more details? Ask a follow-up question!*\n\n_- {self.bot_name} by {self.company}_"
                
                self.usage_stats["successful_responses"] += 1
                
                # Log response time
                response_time = time.time() - start_time
                logger.info(f"Response generated in {response_time:.2f}s")
                
                return response
                
            except Exception as e:
                logger.error(f"AI model error: {e}")
                self.usage_stats["failed_responses"] += 1
                return f"⚠️ AI service temporarily unavailable. Please try again in a moment.\n\nError details: {str(e)}"
                
        except Exception as e:
            logger.error(f"Response generation error: {e}")
            self.usage_stats["failed_responses"] += 1
            return "❌ Something went wrong. Please try again or contact support if the issue persists."
    
    def chat(self, message: str, history: List[Tuple[str, str]]) -> Tuple[str, List[Tuple[str, str]]]:
        """Main chat function for Gradio interface"""
        response = self.get_response(message)
        history.append((message, response))
        return "", history
    
    def get_stats(self) -> dict:
        """Get usage statistics for monitoring"""
        self.usage_stats["current_documents"] = self.collection.count()
        self.usage_stats["uptime"] = datetime.now().isoformat()
        return self.usage_stats

# Initialize bot
bot = iTethrBot()

# Enhanced WhatsApp Integration
class ProductionWhatsAppBot:
    """Production-ready WhatsApp integration"""
    
    def __init__(self, bot_instance):
        self.bot = bot_instance
        self.app = Flask(__name__)
        self.setup_routes()
        
        # Rate limiting (simple implementation)
        self.rate_limit = {}
        self.max_requests_per_hour = 50
        
    def setup_routes(self):
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint for monitoring"""
            return jsonify({
                "status": "healthy",
                "bot_version": self.bot.version,
                "timestamp": datetime.now().isoformat(),
                "stats": self.bot.get_stats()
            })
        
        @self.app.route('/whatsapp', methods=['POST'])
        def whatsapp_webhook():
            """Enhanced WhatsApp webhook with rate limiting"""
            try:
                data = request.json
                message = data.get('message', '').strip()
                phone = data.get('phone', '').strip()
                
                if not message or not phone:
                    return jsonify({
                        "status": "error", 
                        "message": "Missing message or phone number"
                    }), 400
                
                # Simple rate limiting
                if not self._check_rate_limit(phone):
                    return jsonify({
                        "status": "error",
                        "message": "Rate limit exceeded. Please try again later."
                    }), 429
                
                # Get bot response
                response = self.bot.get_response(message)
                
                # Send WhatsApp message
                success = self.send_whatsapp_message(phone, response)
                
                if success:
                    return jsonify({"status": "success", "message": "Response sent"})
                else:
                    return jsonify({"status": "error", "message": "Failed to send message"}), 500
                    
            except Exception as e:
                logger.error(f"WhatsApp webhook error: {e}")
                return jsonify({"status": "error", "message": str(e)}), 500
        
        @self.app.route('/send', methods=['POST'])
        def send_message():
            """Direct message sending endpoint"""
            try:
                data = request.json
                phone = data.get('phone', '').strip()
                message = data.get('message', '').strip()
                
                if not phone or not message:
                    return jsonify({
                        "status": "error",
                        "message": "Phone and message are required"
                    }), 400
                
                # Get bot response for the message
                bot_response = self.bot.get_response(message)
                
                # Send response via WhatsApp
                success = self.send_whatsapp_message(phone, bot_response)
                
                if success:
                    return jsonify({"status": "success"})
                else:
                    return jsonify({"status": "error", "message": "Failed to send"}), 500
                    
            except Exception as e:
                logger.error(f"Send message error: {e}")
                return jsonify({"status": "error", "message": str(e)}), 500
    
    def _check_rate_limit(self, phone: str) -> bool:
        """Simple rate limiting implementation"""
        current_time = time.time()
        
        if phone not in self.rate_limit:
            self.rate_limit[phone] = []
        
        # Remove old requests (older than 1 hour)
        self.rate_limit[phone] = [
            req_time for req_time in self.rate_limit[phone] 
            if current_time - req_time < 3600
        ]
        
        # Check if under limit
        if len(self.rate_limit[phone]) < self.max_requests_per_hour:
            self.rate_limit[phone].append(current_time)
            return True
        
        return False
    
    def send_whatsapp_message(self, phone: str, message: str) -> bool:
        """Enhanced WhatsApp message sending with error handling"""
        try:
            # Format phone number
            if not phone.startswith('+'):
                phone = '+' + phone
            
            # Limit message length for WhatsApp
            if len(message) > 1500:
                message = message[:1500] + "...\n\n(Message truncated for WhatsApp)"
            
            # Send message
            kit.sendwhatmsg_instantly(phone, message, 15, True, 2)
            logger.info(f"✅ WhatsApp message sent to {phone}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to send WhatsApp message to {phone}: {e}")
            return False
    
    def run(self, port: int = 5000):
        """Run WhatsApp bot server"""
        logger.info(f"🚀 WhatsApp API server starting on port {port}")
        self.app.run(host='0.0.0.0', port=port, debug=False)

# Enhanced Gradio Interface
def create_production_interface():
    """Create production-ready Gradio interface"""
    
    # Custom CSS for professional look
    custom_css = """
    .gradio-container {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .stats-container {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
    """
    
    with gr.Blocks(
        title=f"{bot.bot_name} - {bot.company}",
        theme=gr.themes.Soft(),
        css=custom_css
    ) as app:
        
        # Header
        with gr.Row():
            gr.Markdown(
                f"""
                <div class="header">
                    <h1>🤖 {bot.bot_name}</h1>
                    <h3>Powered by {bot.company} • Version {bot.version}</h3>
                    <p>Your intelligent iTethr documentation assistant</p>
                </div>
                """,
                elem_classes=["header"]
            )
        
        # Main chat interface
        with gr.Row():
            with gr.Column(scale=2):
                # Chat section
                chatbot = gr.Chatbot(
                    height=450,
                    label="💬 Chat with iTethr Assistant",
                    placeholder="Ask me anything about iTethr platform, development, or deployment..."
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Type your question about iTethr...",
                        label="",
                        scale=4,
                        max_lines=3
                    )
                    send = gr.Button("Send", variant="primary", scale=1)
                
                # Quick suggestions
                gr.Markdown("### 🚀 Quick Questions")
                with gr.Row():
                    for suggestion in bot.suggestions[:5]:
                        btn = gr.Button(suggestion, size="sm", variant="secondary")
                        btn.click(lambda s=suggestion: s, outputs=msg)
                
                with gr.Row():
                    for suggestion in bot.suggestions[5:]:
                        btn = gr.Button(suggestion, size="sm", variant="secondary")
                        btn.click(lambda s=suggestion: s, outputs=msg)
            
            with gr.Column(scale=1):
                # WhatsApp Integration Panel
                gr.Markdown("### 📱 WhatsApp Integration")
                
                with gr.Group():
                    phone_input = gr.Textbox(
                        placeholder="+1234567890",
                        label="Phone Number",
                        info="Include country code"
                    )
                    whatsapp_msg = gr.Textbox(
                        placeholder="Ask a question...",
                        label="Message",
                        lines=2
                    )
                    whatsapp_send = gr.Button(
                        "Send Answer to WhatsApp",
                        variant="secondary"
                    )
                    whatsapp_status = gr.Markdown("")
                
                # Bot Statistics
                gr.Markdown("### 📊 Bot Statistics")
                stats_display = gr.Markdown(
                    f"""
                    <div class="stats-container">
                    <strong>Current Status:</strong><br>
                    • Documents: {bot.collection.count()} chunks loaded<br>
                    • AI Model: {bot.model_name}<br>
                    • Session: {bot.session_id}<br>
                    • Status: 🟢 Online<br>
                    • Version: {bot.version}
                    </div>
                    """,
                    elem_classes=["stats-container"]
                )
                
                # Refresh stats button
                refresh_btn = gr.Button("🔄 Refresh Stats", size="sm")
                
                def refresh_stats():
                    stats = bot.get_stats()
                    return f"""
                    <div class="stats-container">
                    <strong>Live Statistics:</strong><br>
                    • Questions: {stats['total_questions']}<br>
                    • Successful: {stats['successful_responses']}<br>
                    • Failed: {stats['failed_responses']}<br>
                    • Documents: {stats['current_documents']} chunks<br>
                    • Session: {bot.session_id}<br>
                    • Status: 🟢 Online
                    </div>
                    """
                
                refresh_btn.click(refresh_stats, outputs=stats_display)
        
        # Instructions
        with gr.Row():
            gr.Markdown("""
            ### 📖 How to Use
            1. **Ask Questions**: Type any question about iTethr platform, features, or development
            2. **Quick Buttons**: Click on suggested questions for common topics
            3. **WhatsApp**: Send answers directly to team members via WhatsApp
            4. **Team Access**: Share this URL with your team for global access
            
            ### 🔗 For Developers
            - **API Health**: `/health` endpoint for monitoring
            - **WhatsApp Webhook**: `/whatsapp` for automated responses
            - **Direct Send**: `/send` for programmatic messaging
            """)
        
        # Event handlers
        def send_whatsapp_answer(phone, message):
            try:
                if not phone or not message:
                    return "❌ Please enter both phone number and message"
                
                # Get bot response
                bot_response = bot.get_response(message)
                
                # Send via WhatsApp
                success = whatsapp_bot.send_whatsapp_message(phone, bot_response)
                
                if success:
                    return "✅ Answer sent to WhatsApp successfully!"
                else:
                    return "❌ Failed to send WhatsApp message"
                    
            except Exception as e:
                logger.error(f"WhatsApp send error: {e}")
                return f"❌ Error: {str(e)}"
        
        # Connect events
        send.click(bot.chat, [msg, chatbot], [msg, chatbot])
        msg.submit(bot.chat, [msg, chatbot], [msg, chatbot])
        
        whatsapp_send.click(
            send_whatsapp_answer,
            [phone_input, whatsapp_msg],
            whatsapp_status
        )
        
        return app

# Initialize WhatsApp bot
whatsapp_bot = ProductionWhatsAppBot(bot)

# Main application launcher
if __name__ == "__main__":
    logger.info(f"🚀 Starting {bot.bot_name} v{bot.version} - Production Mode")
    
    # Start WhatsApp API server in background
    if os.getenv('ENABLE_WHATSAPP', 'true').lower() == 'true':
        whatsapp_thread = threading.Thread(
            target=whatsapp_bot.run, 
            args=(int(os.getenv('WHATSAPP_PORT', '5000')),)
        )
        whatsapp_thread.daemon = True
        whatsapp_thread.start()
        logger.info("📱 WhatsApp API server started")
    
    # Launch Gradio interface
    app = create_production_interface()
    
    # Production launch configuration
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv('PORT', '7860')),
        share=os.getenv('GRADIO_SHARE', 'false').lower() == 'true',
        debug=os.getenv('DEBUG', 'false').lower() == 'true',
        show_error=True,
        quiet=False
    )