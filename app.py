# iTethr Bot - AeonovX Team Version with Phase 2: Intelligence & Memory
# File: app.py

import gradio as gr
import os
from sentence_transformers import SentenceTransformer
from groq import Groq
import json
import yaml
import time
import logging
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Dict, Any
import hashlib
import signal
import sys
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from team_manager import AEONOVX_TEAM
import pickle
from collections import defaultdict, deque

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

class ConversationMemory:
    """Enhanced conversation memory system for tracking user context across sessions"""
    
    def __init__(self, max_history_per_user=50):
        self.max_history_per_user = max_history_per_user
        self.user_conversations = defaultdict(deque)  # username -> conversation history
        self.user_preferences = defaultdict(dict)     # username -> preferences
        self.user_context = defaultdict(dict)         # username -> current context
        self.session_data = {}                        # temporary session storage
        
        # Load existing memory if available
        self._load_memory()
    
    def _load_memory(self):
        """Load conversation memory from file"""
        try:
            if os.path.exists('./data/memory.pkl'):
                with open('./data/memory.pkl', 'rb') as f:
                    data = pickle.load(f)
                    self.user_conversations = data.get('conversations', defaultdict(deque))
                    self.user_preferences = data.get('preferences', defaultdict(dict))
                    self.user_context = data.get('context', defaultdict(dict))
                logger.info("💾 Loaded conversation memory")
        except Exception as e:
            logger.error(f"Failed to load memory: {e}")
    
    def _save_memory(self):
        """Save conversation memory to file"""
        try:
            os.makedirs('./data', exist_ok=True)
            with open('./data/memory.pkl', 'wb') as f:
                pickle.dump({
                    'conversations': dict(self.user_conversations),
                    'preferences': dict(self.user_preferences),
                    'context': dict(self.user_context)
                }, f)
        except Exception as e:
            logger.error(f"Failed to save memory: {e}")
    
    def add_conversation(self, username: str, question: str, response: str, topics: List[str] = None):
        """Add conversation to user's history"""
        conversation_entry = {
            'timestamp': datetime.now().isoformat(),
            'question': question,
            'response': response,
            'topics': topics or [],
            'session_id': self.session_data.get('session_id', 'unknown')
        }
        
        # Add to conversation history with max limit
        if len(self.user_conversations[username]) >= self.max_history_per_user:
            self.user_conversations[username].popleft()
        
        self.user_conversations[username].append(conversation_entry)
        
        # Update context tracking
        self._update_user_context(username, question, topics)
        
        # Save memory
        self._save_memory()
    
    def _update_user_context(self, username: str, question: str, topics: List[str]):
        """Update user context based on recent questions"""
        context = self.user_context[username]
        
        # Track frequently asked topics
        if 'frequent_topics' not in context:
            context['frequent_topics'] = defaultdict(int)
        
        for topic in (topics or []):
            context['frequent_topics'][topic] += 1
        
        # Track recent question patterns
        question_lower = question.lower()
        if 'question_patterns' not in context:
            context['question_patterns'] = []
        
        context['question_patterns'].append({
            'pattern': question_lower,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep only last 10 patterns
        context['question_patterns'] = context['question_patterns'][-10:]
        
        # Update last activity
        context['last_activity'] = datetime.now().isoformat()
    
    def get_user_context(self, username: str) -> Dict[str, Any]:
        """Get user's conversation context"""
        return self.user_context.get(username, {})
    
    def get_recent_conversations(self, username: str, limit: int = 5) -> List[Dict]:
        """Get recent conversations for context"""
        conversations = list(self.user_conversations.get(username, []))
        return conversations[-limit:] if conversations else []
    
    def get_conversation_summary(self, username: str) -> str:
        """Generate a summary of user's conversation patterns"""
        context = self.get_user_context(username)
        recent = self.get_recent_conversations(username, 3)
        
        summary_parts = []
        
        # Add user role context
        user_role = None
        for name, info in AEONOVX_TEAM.items():
            if name == username:
                user_role = info.get('role', 'Team Member')
                break
        
        if user_role:
            summary_parts.append(f"Role: {user_role}")
        
        # Add frequent topics
        if 'frequent_topics' in context:
            top_topics = sorted(context['frequent_topics'].items(), 
                              key=lambda x: x[1], reverse=True)[:3]
            if top_topics:
                topics_str = ", ".join([topic for topic, _ in top_topics])
                summary_parts.append(f"Frequent topics: {topics_str}")
        
        # Add recent conversation context
        if recent:
            recent_topics = []
            for conv in recent:
                if conv.get('topics'):
                    recent_topics.extend(conv['topics'])
            
            if recent_topics:
                unique_recent = list(set(recent_topics))[:3]
                summary_parts.append(f"Recent focus: {', '.join(unique_recent)}")
        
        return " | ".join(summary_parts) if summary_parts else "New user"


class SmartSuggestionsEngine:
    """Intelligent suggestions engine for contextual recommendations"""
    
    def __init__(self, embeddings_model, groq_client=None):
        self.embeddings_model = embeddings_model
        self.groq_client = groq_client
        
        # Domain categories for suggestions
        self.domain_categories = {
            'development': ['coding', 'programming', 'development', 'api', 'database', 'frontend', 'backend'],
            'design': ['ui', 'ux', 'design', 'interface', 'user experience', 'visual'],
            'project': ['project', 'management', 'planning', 'timeline', 'team'],
            'documentation': ['docs', 'documentation', 'guide', 'tutorial', 'help'],
            'iTethr': ['community', 'hub', 'room', 'authentication', 'bubble', 'aeono', 'ifeed']
        }
    
    def generate_suggestions(self, question: str, response: str, user_context: Dict, 
                           available_documents: List[str]) -> List[Dict[str, str]]:
        """Generate smart suggestions based on context"""
        suggestions = []
        
        # 1. Follow-up questions based on response
        follow_ups = self._generate_followup_questions(question, response)
        suggestions.extend(follow_ups)
        
        # 2. Domain-aware suggestions
        domain_suggestions = self._generate_domain_suggestions(question, user_context)
        suggestions.extend(domain_suggestions)
        
        # 3. Related document suggestions
        doc_suggestions = self._suggest_related_documents(question, available_documents)
        suggestions.extend(doc_suggestions)
        
        # 4. Role-based suggestions
        role_suggestions = self._generate_role_suggestions(user_context)
        suggestions.extend(role_suggestions)
        
        # Return top 6 suggestions
        return suggestions[:6]
    
    def _generate_followup_questions(self, question: str, response: str) -> List[Dict[str, str]]:
        """Generate intelligent follow-up questions"""
        followups = []
        question_lower = question.lower()
        
        # Pattern-based follow-ups
        if any(word in question_lower for word in ['what is', 'explain', 'tell me about']):
            followups.append({
                'type': 'followup',
                'text': 'How do I implement this?',
                'icon': '🔧'
            })
            followups.append({
                'type': 'followup', 
                'text': 'Show me examples',
                'icon': '📝'
            })
        
        if 'community' in question_lower or 'hub' in question_lower:
            followups.append({
                'type': 'followup',
                'text': 'How do users join communities?',
                'icon': '👥'
            })
        
        if 'authentication' in question_lower:
            followups.append({
                'type': 'followup',
                'text': 'What are the security features?',
                'icon': '🔒'
            })
        
        return followups
    
    def _generate_domain_suggestions(self, question: str, user_context: Dict) -> List[Dict[str, str]]:
        """Generate domain-aware suggestions"""
        suggestions = []
        question_lower = question.lower()
        
        # Detect primary domain
        detected_domain = None
        for domain, keywords in self.domain_categories.items():
            if any(keyword in question_lower for keyword in keywords):
                detected_domain = domain
                break
        
        # Generate suggestions based on domain
        if detected_domain == 'iTethr':
            suggestions.extend([
                {'type': 'domain', 'text': 'iTethr bubble interface design', 'icon': '🫧'},
                {'type': 'domain', 'text': 'Aeono AI assistant features', 'icon': '🤖'}
            ])
        elif detected_domain == 'development':
            suggestions.extend([
                {'type': 'domain', 'text': 'API integration best practices', 'icon': '⚡'},
                {'type': 'domain', 'text': 'Database optimization tips', 'icon': '🗄️'}
            ])
        elif detected_domain == 'design':
            suggestions.extend([
                {'type': 'domain', 'text': 'User experience guidelines', 'icon': '🎨'},
                {'type': 'domain', 'text': 'Interface design patterns', 'icon': '📱'}
            ])
        
        return suggestions
    
    def _suggest_related_documents(self, question: str, available_docs: List[str]) -> List[Dict[str, str]]:
        """Suggest related documents based on semantic similarity"""
        suggestions = []
        
        try:
            if not available_docs or not self.embeddings_model:
                return suggestions
            
            # Create embedding for question
            question_embedding = self.embeddings_model.encode([question])
            
            # Simple keyword matching for documents (fallback)
            question_words = set(question.lower().split())
            
            for doc in available_docs[:3]:  # Limit to top 3 docs
                # Simple relevance check
                doc_words = set(doc.lower().split())
                if question_words.intersection(doc_words):
                    suggestions.append({
                        'type': 'document',
                        'text': f'Read: {doc}',
                        'icon': '📄'
                    })
        
        except Exception as e:
            logger.error(f"Document suggestion error: {e}")
        
        return suggestions
    
    def _generate_role_suggestions(self, user_context: Dict) -> List[Dict[str, str]]:
        """Generate role-specific suggestions"""
        suggestions = []
        
        # Extract user role from context
        context_str = str(user_context)
        
        if 'Developer' in context_str:
            suggestions.append({
                'type': 'role',
                'text': 'Review code architecture patterns',
                'icon': '👨‍💻'
            })
        elif 'Designer' in context_str:
            suggestions.append({
                'type': 'role', 
                'text': 'Check design system guidelines',
                'icon': '🎨'
            })
        elif 'Manager' in context_str or 'Financial' in context_str:
            suggestions.append({
                'type': 'role',
                'text': 'View project analytics dashboard',
                'icon': '📊'
            })
        
        return suggestions


def generate_ai_welcome(name, role):
    """Generate personalized AI welcome message using Groq"""
    try:
        groq_api_key = os.getenv('GROQ_API_KEY', '')
        if not groq_api_key:
            return f"Welcome {name}! Great to have you on the AeonovX team! 🚀"
        
        groq_client = Groq(api_key=groq_api_key)
        
        prompt = f"""Generate a personalized, friendly welcome message for {name} who is a {role} at AeonovX company. 

Requirements:
- Keep it under 50 words
- Make it specific to their role and what they might need help with
- Be professional but warm and encouraging
- Include relevant emojis (1-2 max)
- Mention they can ask about iTethr platform, projects, or team resources
- Sound natural and conversational

Name: {name}
Role: {role}
Company: AeonovX

Generate a unique welcome message:"""

        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-8b-8192",
            temperature=0.8,
            max_tokens=80,
            top_p=0.9
        )
        
        ai_welcome = response.choices[0].message.content.strip()
        return ai_welcome if ai_welcome else f"Welcome {name}! Great to have you on the AeonovX team! 🚀"
        
    except Exception as e:
        logger.error(f"AI welcome generation failed: {e}")
        role_welcomes = {
            "Lead Developer": f"Welcome {name}! Ready to architect amazing solutions? 🚀",
            "UI/UX Designer": f"Hi {name}! Time to create beautiful user experiences! 🎨",
            "Backend Developer": f"Hey {name}! Let's build robust systems together! ⚙️",
            "Frontend Developer": f"Welcome {name}! Ready to bring designs to life? 💻",
            "Project Manager": f"Hi {name}! Let's keep projects on track and teams aligned! 📊",
            "DevOps Engineer": f"Hey {name}! Infrastructure and deployments await! 🔧",
            "QA Engineer": f"Welcome {name}! Quality and testing excellence ahead! 🧪",
            "Administrator": f"Welcome {name}! Full system control at your fingertips! 👑"
        }
        return role_welcomes.get(role, f"Welcome {name}! Great to have you on the AeonovX team! 🚀")


class iTethrBot:
    """iTethr Bot - Enhanced with Phase 2: Intelligence & Memory"""
    
    def __init__(self):
        self.version = "8.0.0 - Phase 2"
        self.bot_name = "AeonovX iTethr Assistant"
        
        # Setup Groq API
        self.groq_api_key = os.getenv('GROQ_API_KEY', '')
        if not self.groq_api_key:
            logger.warning("⚠️ GROQ_API_KEY not found. Add it to your Railway environment variables.")
            logger.info("Get your free API key from: https://console.groq.com/keys")
        
        self.groq_client = Groq(api_key=self.groq_api_key) if self.groq_api_key else None
        
        # Phase 2: Initialize memory and suggestions systems
        self.memory = ConversationMemory()
        
        # In-memory storage for embeddings
        self.documents = []
        self.embeddings = []
        self.metadata = []
        
        # Current session tracking
        self.current_user = None
        self.session_id = None
        
        # Setup bot
        self._setup_components()
        logger.info(f"🚀 {self.bot_name} v{self.version} ready with Intelligence & Memory!")
    
    def _setup_components(self):
        """Setup bot components"""
        try:
            # Load embeddings model
            logger.info("Loading embeddings model...")
            self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Initialize suggestions engine
            self.suggestions_engine = SmartSuggestionsEngine(
                self.embeddings_model, 
                self.groq_client
            )
            
            # Load documents and create embeddings
            self._load_all_documents()
            
            self.total_questions = 0
            logger.info("✅ All components ready with Phase 2 features!")
            
        except Exception as e:
            logger.error(f"Setup failed: {e}")
            raise
    
    def set_current_user(self, username: str):
        """Set current user for session tracking"""
        self.current_user = username
        self.session_id = f"{username}_{int(time.time())}"
        self.memory.session_data['session_id'] = self.session_id
        logger.info(f"👤 Session started for {username}")
    
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
                        
                        # Create embeddings for all chunks at once
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
        
        # Convert embeddings to numpy array
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
        chunk_size = 300
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
            
            # Calculate cosine similarity
            similarities = cosine_similarity(question_embedding, self.embeddings)[0]
            
            # Get top 3 most similar documents
            top_indices = np.argsort(similarities)[-3:][::-1]
            
            relevant_docs = []
            for idx in top_indices:
                similarity = similarities[idx]
                if similarity > 0.25:
                    relevant_docs.append(self.documents[idx])
                    logger.info(f"Found relevant content (similarity: {similarity:.3f}) from {self.metadata[idx]['filename']}")
            
            return relevant_docs
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
    
    def _extract_topics(self, question: str) -> List[str]:
        """Extract key topics from question for memory tracking"""
        topics = []
        question_lower = question.lower()
        
        # Define topic keywords
        topic_map = {
            'community': ['community', 'communities', 'hub', 'hubs'],
            'authentication': ['auth', 'authentication', 'login', 'signup', 'password'],
            'ui_design': ['ui', 'interface', 'design', 'bubble', 'layout'],
            'api': ['api', 'endpoint', 'integration', 'backend'],
            'database': ['database', 'data', 'storage', 'sql'],
            'project': ['project', 'management', 'planning', 'team'],
            'aeono': ['aeono', 'ai', 'assistant', 'artificial intelligence'],
            'ifeed': ['ifeed', 'feed', 'content', 'posts'],
            'loops': ['loops', 'threads', 'discussions']
        }
        
        for topic, keywords in topic_map.items():
            if any(keyword in question_lower for keyword in keywords):
                topics.append(topic)
        
        return topics
    
    def _generate_groq_response(self, context: str, question: str, user_context: Dict) -> str:
        """Generate AI response using Groq API with memory context"""
        try:
            if not self.groq_client:
                return self._fallback_response(context, question)
            
            # Build context-aware prompt
            context_summary = ""
            if user_context:
                context_summary = f"User context: {user_context}\n"
            
            prompt = f"""You are the AeonovX iTethr Assistant, an expert on the iTethr platform with memory of past conversations.

USER CONTEXT:
{context_summary}

DOCUMENTATION:
{context}

USER QUESTION: {question}

INSTRUCTIONS:
- Answer based on the provided documentation and user context
- Reference previous conversations if relevant
- Be helpful, conversational, and accurate
- If the documentation doesn't contain the answer, say "I don't have that specific information in the iTethr documentation"
- Keep responses focused and under 300 words
- Be friendly and use a helpful tone
- You are built by AeonovX team for internal use

ANSWER:"""

            chat_completion = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-8b-8192",
                temperature=0.1,
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
            sentences = context.split('. ')
            for sentence in sentences[:3]:
                if any(word in sentence.lower() for word in question_lower.split()):
                    return sentence.strip() + "."
            
            return "I can help you with iTethr platform features, communities, authentication, and more. Try asking about specific topics like 'What is iTethr?' or 'How do communities work?'"
    
    def get_response(self, message: str) -> Tuple[str, List[Dict[str, str]]]:
        """Get response from bot with smart suggestions"""
        start_time = time.time()
        self.total_questions += 1
        
        if not message.strip():
            return "Hi! Ask me anything about iTethr platform. I'll give you accurate answers based on the documentation.", []
        
        # Get user context from memory
        user_context = {}
        if self.current_user:
            user_context = self.memory.get_user_context(self.current_user)
            context_summary = self.memory.get_conversation_summary(self.current_user)
            if context_summary != "New user":
                user_context['summary'] = context_summary
        
        # Search knowledge base
        relevant_docs = self._search_knowledge(message)
        
        if not relevant_docs:
            response = """🧠 Hmm, I couldn't find specific information about that in the iTethr documentation!

**Here's what I can definitely help you with:**

• **iTethr platform overview** - What is iTethr and how it works
• **Community structure** - Communities, Hubs, and Rooms explained  
• **User authentication** - Sign-up processes and account management
• **Bubble interface** - The unique UI design and navigation
• **Aeono AI assistant** - Built-in AI features and capabilities
• **iFeed functionality** - Global content streams and social features

Try asking about any of these topics! 🚀"""
            suggestions = []
        else:
            # Combine relevant documents for better context
            context = "\n\n".join(relevant_docs[:2])
            
            # Generate response using Groq API with memory context
            response = self._generate_groq_response(context, message, user_context)
            response += f"\n\n*Based on iTethr documentation*"
            
            # Generate smart suggestions
            topics = self._extract_topics(message)
            available_docs = [meta['filename'] for meta in self.metadata]
            suggestions = self.suggestions_engine.generate_suggestions(
                message, response, user_context, available_docs
            )
            
            # Save conversation to memory
            if self.current_user:
                self.memory.add_conversation(self.current_user, message, response, topics)
        
        # Log response time
        response_time = time.time() - start_time
        logger.info(f"⚡ Response generated in {response_time:.2f}s")
        
        return response, suggestions
    
    def chat(self, message: str, history: List[Tuple[str, str]]) -> Tuple[str, List[Tuple[str, str]], str]:
        """Chat function for Gradio with suggestions"""
        if message.strip():
            response, suggestions = self.get_response(message)
            history.append((message, response))
            
            # Format suggestions for display
            suggestions_html = self._format_suggestions(suggestions)
            return "", history, suggestions_html
        return "", history, ""
    
    def _format_suggestions(self, suggestions: List[Dict[str, str]]) -> str:
        """Format suggestions as HTML"""
        if not suggestions:
            return ""
        
        html = "<div style='margin-top: 15px;'><h4>💡 Smart Suggestions:</h4>"
        html += "<div style='display: flex; flex-wrap: wrap; gap: 8px;'>"
        
        for suggestion in suggestions:
            icon = suggestion.get('icon', '💭')
            text = suggestion.get('text', '')
            suggestion_type = suggestion.get('type', 'general')
            
            # Color coding by type
            color_map = {
                'followup': '#4CAF50',
                'domain': '#2196F3', 
                'document': '#FF9800',
                'role': '#9C27B0',
                'general': '#607D8B'
            }
            
            color = color_map.get(suggestion_type, '#607D8B')
            
            html += f"""
            <span style='
                background: {color}15; 
                border: 1px solid {color}; 
                border-radius: 15px; 
                padding: 5px 10px; 
                font-size: 12px; 
                color: {color};
                cursor: pointer;
                display: inline-block;
                margin: 2px;
            '>
                {icon} {text}
            </span>
            """
        
        html += "</div></div>"
        return html


# Initialize bot
bot = iTethrBot()

def authenticate(name, password):
    """Authenticate AeonovX team members with AI-generated welcome"""
    
    if name in AEONOVX_TEAM and AEONOVX_TEAM[name]["password"] == password:
        role = AEONOVX_TEAM[name]["role"]
        
        # Set current user for memory tracking
        bot.set_current_user(name)
        
        # Generate AI welcome message
        logger.info(f"Generating AI welcome for {name} ({role})")
        ai_welcome = generate_ai_welcome(name, role)
        
        # Get user's conversation history summary
        context_summary = bot.memory.get_conversation_summary(name)
        
        # Build personalized welcome
        if context_summary != "New user":
            full_welcome = f"🤖 **AI Assistant**: {ai_welcome}\n\n**Your Role:** {role}\n**Previous Context:** {context_summary}\n\n*Ready to continue helping with iTethr platform questions, building on our previous conversations!*"
        else:
            full_welcome = f"🤖 **AI Assistant**: {ai_welcome}\n\n**Your Role:** {role}\n**Access Level:** AeonovX Team Member\n\n*Ready to help with iTethr platform questions, project info, and team resources!*"
        
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
    """Create Gradio interface with Phase 2 features"""
    
    with gr.Blocks(
        title="AeonovX iTethr Assistant",
        theme=gr.themes.Soft(primary_hue="blue")
    ) as interface:
        
        # LOGIN SCREEN
        with gr.Column(visible=True) as login_screen:
            gr.Markdown("""
            # 🏢 AeonovX iTethr Assistant - Phase 2
            **Enhanced with Intelligence & Memory - Authorized Personnel Required**
            
            *Phase 2 Features: Conversation Memory • Smart Suggestions • Context Awareness*
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
            # 🚀 AeonovX iTethr Assistant - Phase 2
            **Enhanced Intelligence & Memory System - Powered by AeonovX**
            
            *AI responses with conversation memory and smart suggestions - v{bot.version}*
            """)
            
            chatbot = gr.Chatbot(
                height=450,
                label="💬 Chat with Enhanced AeonovX Assistant",
                show_copy_button=True,
                avatar_images=("👤", "🤖")
            )
            
            # Smart suggestions display
            suggestions_display = gr.HTML(label="Smart Suggestions")
            
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Ask me anything - I'll remember our conversation and provide smart suggestions...",
                    label="",
                    scale=5,
                    max_lines=3
                )
                send = gr.Button("Send ⚡", variant="primary", scale=1)
            
            # Quick suggestions with Phase 2 examples
            gr.Markdown("### 💡 Try Phase 2 Features")
            with gr.Row():
                btn1 = gr.Button("What is iTethr?", size="sm")
                btn2 = gr.Button("Remember my questions?", size="sm")
                btn3 = gr.Button("Smart suggestions demo", size="sm")
            
            with gr.Row():
                btn4 = gr.Button("Community structure?", size="sm")
                btn5 = gr.Button("Authentication process?", size="sm")
                btn6 = gr.Button("Show my context", size="sm")
            
            gr.Markdown("""
            ### ⚡ Phase 2: Enhanced Intelligence Features
            
            **🧠 New Memory Capabilities:**
            • **Conversation History** – Remembers all your questions across sessions 💾
            • **Context Awareness** – Builds on previous conversations naturally 🔄
            • **User Preferences** – Tracks your interests and frequent topics 📊
            • **Role Adaptation** – Personalizes responses based on your team role 👤
            
            **💡 Smart Suggestions Engine:**
            • **Follow-up Questions** – Suggests natural next steps after responses ➡️
            • **Domain-aware Recommendations** – Context-specific suggestions by topic 🎯
            • **Related Documents** – Finds relevant docs automatically 📄
            • **Role-based Suggestions** – Tailored to your responsibilities 👔
            
            **🚀 Always Learning:**
            • Each conversation improves future interactions
            • Personalized experience that grows with usage
            • Smart context detection across all iTethr topics
                        
            **Available 24/7 for the AeonovX team** - Your evolving AI companion with memory and intelligence!
            """)
            
            def safe_chat(message, history):
                try:
                    return bot.chat(message, history)
                except Exception as e:
                    logger.error(f"Chat error: {e}")
                    if message.strip():
                        history.append((message, f"Sorry, I encountered an error: {str(e)}"))
                    return "", history, ""
            
            # Connect events
            send.click(safe_chat, [msg, chatbot], [msg, chatbot, suggestions_display])
            msg.submit(safe_chat, [msg, chatbot], [msg, chatbot, suggestions_display])
            
            btn1.click(lambda: "What is iTethr platform?", outputs=msg)
            btn2.click(lambda: "Do you remember our previous conversations?", outputs=msg)
            btn3.click(lambda: "Show me smart suggestions after this response", outputs=msg)
            btn4.click(lambda: "Explain iTethr community structure", outputs=msg)
            btn5.click(lambda: "How does iTethr authentication work?", outputs=msg)
            btn6.click(lambda: "What context do you have about me?", outputs=msg)
        
        # Connect login with both inputs
        login_btn.click(
            authenticate,
            [name_input, password_input],
            [login_screen, bot_interface, login_status]
        )
    
    # Health check with Phase 2 info
    @interface.app.get("/health")
    async def health_check():
        from fastapi.responses import JSONResponse
        return JSONResponse(content={
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "service": "aeonovx-itethr-assistant",
            "version": bot.version,
            "phase": "2 - Intelligence & Memory",
            "documents_loaded": len(bot.documents),
            "groq_api_configured": bool(bot.groq_api_key),
            "team_members_active": len(AEONOVX_TEAM),
            "features": [
                "ai_welcome", 
                "semantic_search", 
                "team_auth",
                "conversation_memory",
                "smart_suggestions",
                "context_awareness",
                "user_preferences"
            ],
            "memory_users": len(bot.memory.user_conversations),
            "total_conversations": sum(len(convs) for convs in bot.memory.user_conversations.values())
        }, status_code=200)
    
    return interface

def setup_railway_config():
    """Configure for Railway"""
    port = int(os.getenv('PORT', '7860'))
    logger.info(f"🔌 Using port: {port}")
    return port

# Graceful shutdown with memory save
def signal_handler(sig, frame):
    logger.info('🛑 Shutting down AeonovX Assistant...')
    if bot and bot.memory:
        bot.memory._save_memory()
        logger.info('💾 Conversation memory saved')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

if __name__ == "__main__":
    try:
        logger.info(f"🚀 Starting {bot.bot_name} v{bot.version}")
        logger.info(f"👥 Team members configured: {len(AEONOVX_TEAM)}")
        logger.info(f"🧠 Memory system: Enabled")
        logger.info(f"💡 Smart suggestions: Enabled")
        logger.info(f"🤖 AI welcome messages: {'Enabled' if os.getenv('GROQ_API_KEY') else 'Disabled (fallback)'}")
        
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