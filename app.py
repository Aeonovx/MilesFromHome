# iBot - AeonovX Team Assistant (Web-Only Version)
# File: app.py
# Version: 1.3.0 (Personalization Release)
# Description: A web-only chatbot with conversation memory and role-based personalization
# for a tailored user experience.

import gradio as gr
import os
from sentence_transformers import SentenceTransformer
from groq import Groq
import logging
from datetime import datetime
from typing import List, Tuple, Dict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from team_manager import AEONOVX_TEAM
import pickle
from collections import defaultdict, deque

# --- Configuration ---
BOT_NAME = "iBot"
BOT_VERSION = "1.3.0 (Personalization Release)"
LOG_FILE = "ibot_web.log"
MEMORY_FILE = './data/ibot_memory.pkl'
DOCUMENTS_FOLDER = './documents'
MAX_CONTEXT_LENGTH = 3800

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

# --- Conversation Memory (Pickle-Safe) ---
def create_deque():
    return deque(maxlen=50)

class ConversationMemory:
    """Manages conversation history and user context."""
    def __init__(self, memory_file):
        self.memory_file = memory_file
        self.user_conversations = defaultdict(create_deque)
        self._load_memory()

    def _load_memory(self):
        if not os.path.exists(self.memory_file): return
        try:
            with open(self.memory_file, 'rb') as f:
                self.user_conversations = pickle.load(f)
            logger.info("🧠 Conversation memory loaded successfully.")
        except Exception as e:
            logger.error(f"Could not load memory file: {e}. Starting fresh.")
            self.user_conversations = defaultdict(create_deque)

    def _save_memory(self):
        try:
            os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
            with open(self.memory_file, 'wb') as f:
                pickle.dump(self.user_conversations, f)
        except Exception as e:
            logger.error(f"Error saving memory: {e}")

    def add_conversation(self, username: str, question: str, response: str):
        self.user_conversations[username].append({'role': 'user', 'content': question})
        self.user_conversations[username].append({'role': 'assistant', 'content': response})
        self._save_memory()

    def get_history_as_messages(self, username: str) -> List[Dict]:
        return list(self.user_conversations.get(username, deque()))

# --- The iBot Core ---
class IBot:
    """The main chatbot class with personalization."""
    def __init__(self):
        self.bot_name = BOT_NAME
        self.version = BOT_VERSION
        self.current_user = None
        self.current_user_role = None # To store the user's role
        self.memory = ConversationMemory(MEMORY_FILE)
        self._initialize_bot()

    def _initialize_bot(self):
        """Initializes all components of the bot."""
        logger.info(f"🚀 Initializing {self.bot_name} v{self.version}...")
        groq_api_key = os.getenv('GROQ_API_KEY')
        if not groq_api_key: logger.critical("FATAL: GROQ_API_KEY not set.")
        self.groq_client = Groq(api_key=groq_api_key)

        try:
            self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ Embeddings model loaded.")
        except Exception as e:
            logger.error(f"Failed to load embeddings model: {e}"); raise

        self._load_and_embed_documents()
        logger.info(f"✅ {self.bot_name} is ready to assist!")

    def _load_and_embed_documents(self):
        """Loads documents and creates smaller, more effective chunks."""
        if not os.path.exists(DOCUMENTS_FOLDER):
            os.makedirs(DOCUMENTS_FOLDER); logger.warning(f"'{DOCUMENTS_FOLDER}' created.")
            return

        self.documents, self.metadata = [], []
        supported_exts = ('.txt', '.md')
        for filename in os.listdir(DOCUMENTS_FOLDER):
            if filename.endswith(supported_exts):
                path = os.path.join(DOCUMENTS_FOLDER, filename)
                try:
                    with open(path, 'r', encoding='utf-8') as f: content = f.read()
                    paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
                    for i, para in enumerate(paragraphs):
                        self.documents.append(para)
                        self.metadata.append({'source': filename, 'chunk': i})
                except Exception as e: logger.error(f"Failed to process {path}: {e}")

        if self.documents:
            logger.info(f"Creating embeddings for {len(self.documents)} document chunks...")
            self.embeddings = self.embeddings_model.encode(self.documents, show_progress_bar=True)
            logger.info("✅ Document embeddings created.")
        else:
            logger.warning("No documents loaded. Knowledge base is empty.")

    def _search_knowledge(self, query: str) -> str:
        """Searches knowledge base and returns a context string with a controlled length."""
        if not hasattr(self, 'embeddings') or self.embeddings.size == 0: return ""
        query_embedding = self.embeddings_model.encode([query])
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        top_indices = np.argsort(similarities)[-5:][::-1]
        
        context = ""
        for idx in top_indices:
            if similarities[idx] > 0.35:
                chunk = self.documents[idx]
                if len(context) + len(chunk) < MAX_CONTEXT_LENGTH:
                    context += chunk + "\n\n"
                else: break
        logger.info(f"Retrieved context of {len(context)} characters for query.")
        return context.strip()

    def _generate_response(self, question: str, context: str, history: List[Dict], user_name: str, user_role: str) -> str:
        """Generates a response using the Groq API with user personalization."""
        if not self.groq_client or not os.getenv('GROQ_API_KEY'):
            return "I am unable to connect to the AI service. Please ensure the API key is configured."

        # Personalized system prompt
        system_prompt = f"""You are iBot, a friendly and highly intelligent assistant for the AeonovX team.
        You are currently speaking with {user_name}, who is a {user_role}.
        Tailor your response to be relevant to their role. For example, for a 'Developer' or 'AI Developer', you can be more technical. For a 'Financial Analyst', focus on business implications. For a 'UI Designer', focus on user experience and design principles.
        Provide accurate, conversational answers based on the provided documentation. If the documentation doesn't contain the answer, say so politely. Do not invent information."""
        
        messages = [{'role': 'system', 'content': system_prompt}]
        messages.extend(history)
        
        prompt_with_context = f"""
        **Relevant Documentation:**
        ---
        {context if context else "No relevant documentation found for this query."}
        ---
        **User's Question:** {question}
        """
        messages.append({'role': 'user', 'content': prompt_with_context})
        
        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=messages, model="llama3-8b-8192", temperature=0.2, max_tokens=1000
            )
            return chat_completion.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Groq API call failed: {e}")
            return "I'm sorry, an error occurred with the AI service. Please try again."

    def get_bot_response(self, message: str) -> str:
        """Main function to get a personalized response from the bot."""
        if not message.strip(): return "Hello! How can I assist you today?"
        if not self.current_user or not self.current_user_role:
            return "Authentication error. Please log in again."

        context = self._search_knowledge(message)
        history = self.memory.get_history_as_messages(self.current_user)
        response = self._generate_response(message, context, history, self.current_user, self.current_user_role)
        self.memory.add_conversation(self.current_user, message, response)
        return response

    def authenticate_user(self, name: str, password: str) -> Tuple[gr.update, gr.update]:
        """Authenticates a user and stores their name and role."""
        user_info = AEONOVX_TEAM.get(name)
        if user_info and user_info["password"] == password:
            self.current_user = name
            self.current_user_role = user_info.get("role", "Team Member") # Get role
            logger.info(f"✅ Authentication successful for {name} ({self.current_user_role}).")
            return gr.update(visible=False), gr.update(visible=True)
        else:
            logger.warning(f"Failed authentication attempt for user: {name}")
            return gr.update(visible=True), gr.update(visible=False)

# --- Gradio UI ---
def create_gradio_interface(bot_instance: IBot):
    """Creates and configures the Gradio web interface."""
    with gr.Blocks(title=f"{BOT_NAME} - AeonovX Assistant", theme=gr.themes.Soft(primary_hue="blue", secondary_hue="sky")) as interface:
        with gr.Column(visible=True) as login_screen:
            gr.Markdown(f"<div style='text-align: center;'><h1>Welcome to {BOT_NAME}</h1><p>Your intelligent assistant for the AeonovX team.</p></div>")
            with gr.Row():
                with gr.Column(scale=1): pass
                with gr.Column(scale=2):
                    name_input = gr.Textbox(label="Team Member Name", placeholder="e.g., Sharjan")
                    password_input = gr.Textbox(label="Password", type="password", placeholder="Enter your password")
                    login_btn = gr.Button("🔐 Login", variant="primary")
                with gr.Column(scale=1): pass

        with gr.Column(visible=False) as chat_screen:
            gr.Markdown(f"<div style='text-align: center;'><h1>{BOT_NAME}</h1><p>Ready to assist. Ask me anything about our projects or documentation!</p></div>")
            
            chatbot = gr.Chatbot(height=500, label="Conversation", avatar_images=("👤", "🤖"), type='messages')
            msg_input = gr.Textbox(placeholder="Type your question here...", label="Your Message", lines=2)
            send_btn = gr.Button("✉️ Send", variant="primary")
            gr.Examples(examples=["What is iTethr?", "Explain the Community Plus structure.", "How does user authentication work?"], inputs=msg_input, label="💡 Quick Questions")

        def chat_interface_logic(message: str, history: List[Dict]):
            history = history or []
            if not message.strip(): return history
            history.append({"role": "user", "content": message})
            # The bot instance already knows the user and their role
            response = bot_instance.get_bot_response(message)
            history.append({"role": "assistant", "content": response})
            return history

        # Event Handlers
        login_btn.click(bot_instance.authenticate_user, [name_input, password_input], [login_screen, chat_screen])
        send_btn.click(chat_interface_logic, [msg_input, chatbot], chatbot).then(lambda: "", outputs=msg_input)
        msg_input.submit(chat_interface_logic, [msg_input, chatbot], chatbot).then(lambda: "", outputs=msg_input)
    return interface

# --- Main Execution ---
if __name__ == "__main__":
    try:
        ibot_instance = IBot()
        web_interface = create_gradio_interface(ibot_instance)
        PORT = int(os.getenv('PORT', 7860))
        logger.info(f"🌐 Launching the web interface on port {PORT}")
        web_interface.launch(server_name="0.0.0.0", server_port=PORT, share=False)
    except Exception as e:
        logger.critical(f"A fatal error occurred during startup: {e}", exc_info=True)