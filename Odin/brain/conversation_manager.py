import os
import datetime
import sqlite3
import logging
import json
import uuid
import numpy as np
from openai import OpenAI as Client
from .memory_handler import MemoryHandler
from .file_picker import FilePicker

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

class ConversationManager:
    def __init__(self):
        self.memory_dir = os.path.join(os.path.dirname(__file__), "Memory")
        os.makedirs(self.memory_dir, exist_ok=True)
        self.conv_folder = None
        self.db_path = None
        self.MODEL_NAME = self._load_model_name()
        self.OPEN_ROUTER_API_KEY = None
        self.client = None
        self.memory_handler = MemoryHandler(self.memory_dir)
        self.file_picker = FilePicker(self)  # Initialize FilePicker
        self.chatbot_ui = None
        self.file_chunks = []  # Store file chunks in memory
        self.file_ids = {}
        self.agentic_role = None  # Agentic role for the current file
        self.init_conversation()

    def _load_model_name(self):
        config_path = os.path.join(os.path.dirname(__file__), "..", "config.json")
        try:
            with open(config_path, "r", encoding="utf-8") as config_file:
                config = json.load(config_file)
                return config["MODEL_NAME"]
        except Exception as e:
            logging.error(f"Failed to load MODEL_NAME from config.json: {str(e)}")
            raise ValueError("MODEL_NAME must be defined in config.json")

    def init_conversation(self):
        """
        Initialize a new conversation with a new folder, database, and Word2Vec model.
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.conv_folder = os.path.join(self.memory_dir, f"memory_{timestamp}")
        os.makedirs(self.conv_folder, exist_ok=True)
        self.db_path = os.path.join(self.conv_folder, "conversations.db")
        self.memory_handler = MemoryHandler(self.conv_folder)
        self.init_db()
        if not os.path.exists(self.memory_handler.model_path):
            logging.info("Initializing Word2Vec model with default data.")
            self.memory_handler.load_or_train_word2vec_model()

    def init_db(self):
        """
        Initialize the SQLite database for the conversation.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                message TEXT,
                message_summary TEXT,
                embedding TEXT,
                role TEXT
            )''')
        conn.commit()
        conn.close()
        logging.info(f"Database initialized at {self.db_path}")

    def clear_conversation(self, new_conversation=False):
        """
        Clear the current conversation and optionally start a new one.
        If new_conversation is True, create a new folder with a new database and model file.
        """
        if new_conversation:
            # Create a new folder for the new conversation
            self.init_conversation()
            logging.info("New conversation started.")
        else:
            # Clear the current conversation without starting a new one
            if self.db_path and os.path.exists(self.db_path):
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute("DELETE FROM conversations")
                conn.commit()
                conn.close()
                logging.info("Current conversation cleared.")
            else:
                logging.warning("No active conversation to clear.")

    def set_openrouter_api_key(self, api_key):
        self.OPEN_ROUTER_API_KEY = api_key
        self.update_client()

    def set_model_name(self, model_name):
        self.MODEL_NAME = model_name
        self.update_client()

    def update_client(self):
        if self.MODEL_NAME and self.OPEN_ROUTER_API_KEY:
            self.client = Client(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.OPEN_ROUTER_API_KEY,
            )
            logging.info("OpenAI client updated with new API key and model name.")

    def set_agentic_role(self, role):
        """
        Set the agentic role for the current file.
        """
        self.agentic_role = role
        logging.info(f"Agentic role set to: {role}")

    def pick_file(self):
        """
        Delegate file picking to the FilePicker instance.
        """
        self.file_picker.pick_file()

    def save_file_chunks(self, file_name, chunks):
        """
        Save file chunks to the database and store them in memory.
        """
        file_id = str(uuid.uuid4())
        self.file_ids[file_name] = file_id
        self.file_chunks = chunks  # Store chunks in memory

        # Retrain Word2Vec on the new chunks
        self.memory_handler.train_word2vec(chunks)

        # Save each chunk with the file_id as the role
        for chunk in chunks:
            self.save_conversation(chunk, None, None, file_id)

    def get_file_chunks(self):
        """
        Retrieve all file chunks from memory.
        """
        return self.file_chunks

    def process_query(self, user_message):
        try:
            context_messages = []
            # Include saved file chunks in the context
            chunks = self.get_file_chunks()
            if chunks:
                context_messages.append({"role": "system", "content": "Here are the sections of the uploaded file:"})
                for chunk in chunks:
                    context_messages.append({"role": "system", "content": chunk})

            context_messages.append({"role": "user", "content": user_message})

            completion = self.client.chat.completions.create(
                model=self.MODEL_NAME,
                messages=context_messages,
                extra_headers={"HTTP-Referer": "your_site_url", "X-Title": "your_app_name"}
            )
            if completion.choices and completion.choices[0].message:
                response_message = completion.choices[0].message.content
                self.save_conversation(user_message, response_message)
                return response_message
            else:
                logging.error("Error processing query: No message found in API response.")
                return None
        except Exception as e:
            logging.error(f"Error processing query: {str(e)}")
            return None

    def save_conversation(self, message, summary=None, embedding=None, role=None):
        """
        Save the conversation to the database with all fields in a single row.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # If embedding is not provided, generate it using the memory handler
        if embedding is None:
            embedding = self.memory_handler.sentence_to_vec(message)
        
        # Convert embedding to a JSON-serializable format (if it's a numpy array)
        if isinstance(embedding, np.ndarray):
            embedding = json.dumps(embedding.tolist())
        
        # Insert the conversation into the database
        cursor.execute('''INSERT INTO conversations (timestamp, message, message_summary, embedding, role)
                          VALUES (?, ?, ?, ?, ?)''',
                       (timestamp, message, summary, embedding, role or self.agentic_role))
        conn.commit()
        conn.close()
        logging.info("Conversation saved to the database.")

    def notify(self, message):
        """
        Notify the user by logging the message and displaying it in the chatbot UI if available.
        """
        logging.info(message)
        if self.chatbot_ui and hasattr(self.chatbot_ui, 'display_response'):
            self.chatbot_ui.display_response({"type": "text", "content": message})
        else:
            logging.warning("Chatbot UI is not initialized. Cannot display response.")