# iBot - AeonovX Team Assistant (Web-Only Version)
# File: app.py
# Description: A streamlined, web-only chatbot for the AeonovX team.
# All Slack integrations have been removed for a pure, interactive web experience.

import gradio as gr
import os
from sentence_transformers import SentenceTransformer
from groq import Groq
import json
import yaml
import time
import logging
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from team_manager import AEONOVX_TEAM
import pickle
from collections import defaultdict, deque

# --- Configuration ---
BOT_NAME = "iBot"
BOT_VERSION = "1.0.0 (Web-Only)"
LOG_FILE = "ibot_web.log"
MEMORY_FILE = './data/ibot_memory.pkl'
DOCUMENTS_FOLDER = './documents'

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Conversation Memory ---
class ConversationMemory:
    """Manages conversation history and user context."""
    def __init__(self, memory_file, max_history=50):
        self.memory_file = memory_file
        self.max_history_per_user = max_history
        self.user_conversations = defaultdict(lambda: deque(maxlen=max_history))
        self._load_memory()

    def _load_memory(self):
        """Loads memory from a pickle file."""
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'rb') as f:
                    self.user_conversations = pickle.load(f)
                logger.info("🧠 Conversation memory loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading memory: {e}")

    def _save_memory(self):
        """Saves memory to a pickle file."""
        try:
            os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
            with open(self.memory_file, 'wb') as f:
                pickle.dump(self.user_conversations, f)
        except Exception as e:
            logger.error(f"Error saving memory: {e}")

    def add_conversation(self, username: str, question: str, response: str):
        """Adds a new interaction to the user's history."""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'question': question,
            'response': response
        }
        self.user_conversations[username].append(entry)
        self._save_memory()

    def get_recent_conversations(self, username: str, limit: int = 5) -> List[Dict]:
        """Retrieves recent conversations for a user."""
        return list(self.user_conversations[username])[-limit:]

# --- The iBot Core ---
class IBot:
    """The main chatbot class for the AeonovX team."""
    def __init__(self):
        self.bot_name = BOT_NAME
        self.version = BOT_VERSION
        self.groq_client = None
        self.embeddings_model = None
        self.documents = []
        self.embeddings = np.array([])
        self.metadata = []
        self.current_user = None
        self.memory = ConversationMemory(MEMORY_FILE)
        self._initialize_bot()

    def _initialize_bot(self):
        """Initializes all components of the bot."""
        logger.info(f"🚀 Initializing {self.bot_name} v{self.version}...")
        
        # Groq API Client
        groq_api_key = os.getenv('GROQ_API_KEY')
        if not groq_api_key:
            logger.warning("GROQ_API_KEY is not set. The bot will not be able to generate AI responses.")
        self.groq_client = Groq(api_key=groq_api_key)

        # Sentence Transformer Model
        try:
            self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ Embeddings model loaded.")
        except Exception as e:
            logger.error(f"Failed to load embeddings model: {e}")
            # The bot can't function without this, so we stop.
            raise

        # Load and process documents
        self._load_and_embed_documents()
        logger.info(f"✅ {self.bot_name} is ready to assist!")

    def _load_and_embed_documents(self):
        """Loads documents from the specified folder and creates vector embeddings."""
        if not os.path.exists(DOCUMENTS_FOLDER):
            logger.warning(f"Documents folder not found at '{DOCUMENTS_FOLDER}'. Creating it.")
            os.makedirs(DOCUMENTS_FOLDER)
            return

        supported_exts = ('.txt', '.md', '.py', '.json', '.yaml')
        doc_paths = [os.path.join(DOCUMENTS_FOLDER, f) for f in os.listdir(DOCUMENTS_FOLDER) if f.endswith(supported_exts)]

        for path in doc_paths:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Simple chunking by paragraph
                chunks = [chunk for chunk in content.split('\n\n') if chunk.strip()]
                
                for i, chunk in enumerate(chunks):
                    self.documents.append(chunk)
                    self.metadata.append({'source': os.path.basename(path), 'chunk': i})

            except Exception as e:
                logger.error(f"Failed to read or process {path}: {e}")

        if self.documents:
            logger.info(f"Creating embeddings for {len(self.documents)} document chunks...")
            self.embeddings = self.embeddings_model.encode(self.documents, show_progress_bar=True)
            logger.info("✅ Document embeddings created.")
        else:
            logger.warning("No documents loaded. The bot's knowledge base will be empty.")
            
    def _search_knowledge(self, query: str, top_k: int = 3) -> str:
        """Searches the knowledge base for relevant information using semantic search."""
        if self.embeddings.size == 0 or not query:
            return ""

        query_embedding = self.embeddings_model.encode([query])
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get the indices of the top_k most similar chunks
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Filter out low-similarity results and gather context
        context = ""
        for idx in top_indices:
            if similarities[idx] > 0.3: # Similarity threshold
                context += self.documents[idx] + "\n\n"
        
        return context.strip()

    def _generate_response(self, question: str, context: str, history: List[Dict]) -> str:
        """Generates a response using the Groq API."""
        if not self.groq_client:
            return "I'm sorry, my connection to the AI model is not configured. Please contact the administrator."

        # Prepare conversation history for the prompt
        history_str = "\n".join([f"User: {h['question']}\niBot: {h['response']}" for h in history])

        # Construct the prompt
        prompt = f"""
        You are iBot, a friendly and highly intelligent assistant for the AeonovX team.
        Your goal is to provide accurate, helpful, and conversational answers based on the provided documentation and conversation history.

        **Conversation History:**
        {history_str}

        **Relevant Documentation:**
        ---
        {context}
        ---

        **User's Current Question:**
        {question}

        **Instructions:**
        1.  Synthesize information from the documentation and the conversation history to answer the question.
        2.  If the documentation does not contain the answer, say so clearly and politely. Do not invent information.
        3.  Maintain a conversational and helpful tone.
        4.  Keep your answers concise and to the point.
        
        **Answer:**
        """

        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-8b-8192",
                temperature=0.2,
                max_tokens=500
            )
            return chat_completion.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Groq API call failed: {e}")
            return "I'm sorry, I encountered an error while trying to generate a response. Please try again."

    def get_bot_response(self, message: str) -> str:
        """The main function to get a response from the bot."""
        if not message.strip():
            return "Hello! How can I assist you today?"
        
        # 1. Search for relevant context in the knowledge base
        context = self._search_knowledge(message)
        
        # 2. Get recent conversation history for context
        history = self.memory.get_recent_conversations(self.current_user)
        
        # 3. Generate the AI response
        response = self._generate_response(message, context, history)
        
        # 4. Save the new interaction to memory
        self.memory.add_conversation(self.current_user, message, response)
        
        return response

    def authenticate_user(self, name: str, password: str) -> Tuple[gr.update, gr.update]:
        """Authenticates a user against the team database."""
        user_info = AEONOVX_TEAM.get(name)
        if user_info and user_info["password"] == password:
            self.current_user = name
            logger.info(f"✅ Authentication successful for user: {name}")
            return gr.update(visible=False), gr.update(visible=True)
        else:
            logger.warning(f"Failed authentication attempt for user: {name}")
            # You can add a more explicit error message for the user if you want
            return gr.update(visible=True), gr.update(visible=False)

# --- Gradio UI ---
def create_gradio_interface(bot_instance: IBot):
    """Creates and configures the Gradio web interface."""
    
    with gr.Blocks(
        title=f"{BOT_NAME} - AeonovX Assistant",
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="sky")
    ) as interface:

        # --- Login Screen ---
        with gr.Column(visible=True) as login_screen:
            gr.Markdown(f"""
            <div style='text-align: center;'>
                <h1>Welcome to {BOT_NAME}</h1>
                <p>Your intelligent assistant for the AeonovX team.</p>
            </div>
            """)
            with gr.Row():
                with gr.Column(scale=1):
                    pass
                with gr.Column(scale=2):
                    name_input = gr.Textbox(label="Team Member Name", placeholder="e.g., Sharjan")
                    password_input = gr.Textbox(label="Password", type="password", placeholder="Enter your password")
                    login_btn = gr.Button("🔐 Login", variant="primary")
                with gr.Column(scale=1):
                    pass

        # --- Chat Interface ---
        with gr.Column(visible=False) as chat_screen:
            gr.Markdown(f"""
            <div style='text-align: center;'>
                <h1>{BOT_NAME}</h1>
                <p>Ready to assist you. Ask me anything about our projects or documentation!</p>
            </div>
            """)
            
            chatbot = gr.Chatbot(
                height=500,
                label="Conversation",
                bubble_full_width=False,
                avatar_images=("👤", "🤖")
            )
            
            msg_input = gr.Textbox(
                placeholder="Type your question here...",
                label="Your Message",
                lines=2
            )
            
            send_btn = gr.Button("✉️ Send", variant="primary")

            gr.Examples(
                examples=[
                    "What is iTethr?",
                    "Explain the Community Plus structure.",
                    "How does user authentication work in iTethr?",
                    "Summarize the iTethr overview document.",
                ],
                inputs=msg_input,
                label="💡 Quick Questions"
            )

        # --- Event Handlers ---
        def chat_interface_logic(message, history):
            """Handles the chat logic for the Gradio interface."""
            if not message.strip():
                return "", history
            history = history or []
            response = bot_instance.get_bot_response(message)
            history.append((message, response))
            return "", history

        # Login button action
        login_btn.click(
            bot_instance.authenticate_user,
            [name_input, password_input],
            [login_screen, chat_screen]
        )

        # Send button action
        send_btn.click(
            chat_interface_logic,
            [msg_input, chatbot],
            [msg_input, chatbot]
        )
        
        # Enter key action
        msg_input.submit(
            chat_interface_logic,
            [msg_input, chatbot],
            [msg_input, chatbot]
        )

    return interface


# --- Main Execution ---
if __name__ == "__main__":
    ibot_instance = IBot()
    web_interface = create_gradio_interface(ibot_instance)
    
    PORT = int(os.getenv('PORT', 7860))
    logger.info(f"🌐 Launching the web interface on port {PORT}")
    
    web_interface.launch(
        server_name="0.0.0.0",
        server_port=PORT,
        share=False # Set to True if you need to share it publicly via a Gradio link
    )