import os
import datetime
import sqlite3
import logging
import json
import numpy as np
from openai import OpenAI as Client
from .memory_handler import MemoryHandler  # Import MemoryHandler
from .file_picker import FilePicker

# Set up logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

class ConversationManager:
    def __init__(self):
        """
        Initialize the ConversationManager with a memory directory, database, and Word2Vec model.
        """
        self.memory_dir = os.path.join(os.path.dirname(__file__), "Memory")
        os.makedirs(self.memory_dir, exist_ok=True)
        self.conv_folder = None
        self.db_path = None
        self.MODEL_NAME = self._load_model_name()
        self.OPEN_ROUTER_API_KEY = None
        self.client = None
        self.memory_handler = MemoryHandler(self.memory_dir)  # Initialize MemoryHandler
        self.file_picker = FilePicker(self)  # Initialize FilePicker
        self.chatbot_ui = None
        self.file_chunks = []  # Store file chunks in memory
        self.file_ids = {}
        self.agentic_role = None  # Agentic role for the current file
        self.init_conversation()

    def _load_model_name(self):
        """
        Load the model name from the config.json file.
        """
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
        # Create tables for conversations and files
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

    def get_previous_conversations(self):
        """
        Retrieve previous conversations from the database for context.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT message, role FROM conversations ORDER BY timestamp ASC")
        rows = cursor.fetchall()
        conn.close()
        return [{"message": row[0], "role": row[1]} for row in rows]  # Return a list of messages with roles

    def save_conversation(self, message, summary=None, embedding=None, role=None):
        """
        Save the conversation to the database with all fields in a single row.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Generate a summary if not provided
        if summary is None:
            summary = self.generate_summary(message)
        
        # Generate embedding if not provided
        if embedding is None:
            embedding = self.memory_handler.sentence_to_vec(message)
        
        # Convert embedding to a JSON-serializable format (if it's a numpy array)
        if isinstance(embedding, np.ndarray):
            embedding = json.dumps(embedding.tolist())
        
        # Ensure the role is set (default to agentic_role if not provided)
        role = role or self.agentic_role
        
        # Insert the conversation into the database
        cursor.execute('''INSERT INTO conversations (timestamp, message, message_summary, embedding, role)
                          VALUES (?, ?, ?, ?, ?)''',
                       (timestamp, message, summary, embedding, role))
        conn.commit()
        conn.close()
        logging.info("Conversation saved to the database.")

    def get_file_content(self, file_name):
        """
        Retrieve all chunks of a file from the database and combine them into a single string.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT message FROM conversations WHERE role = ? ORDER BY timestamp ASC", (f"file_chunk:{file_name}",))
        rows = cursor.fetchall()
        conn.close()
        return " ".join(row[0] for row in rows)  # Combine all chunks into a single string

    def set_openrouter_api_key(self, api_key):
        """
        Set the OpenRouter API key and initialize the OpenAI client.
        """
        self.OPEN_ROUTER_API_KEY = api_key
        self.client = Client(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.OPEN_ROUTER_API_KEY,
        )
        logging.info("OpenRouter API key set and client initialized.")

    def set_model_name(self, model_name):
        """
        Set the model name for the conversation.
        """
        self.MODEL_NAME = model_name
        logging.info(f"Model name set to: {self.MODEL_NAME}")

    def process_query(self, user_message):
        """
        Process a user query and generate a response using the AI model or file recall.
        """
        try:
            # Check if the user is asking about a file
            if "what is the file about" in user_message.lower():
                # Retrieve the file name from the query (e.g., "What is the file about example.txt?")
                file_name = user_message.split()[-1].strip("?")
                file_content = self.get_file_content(file_name)
                if file_content:
                    return f"The file '{file_name}' is about: {file_content}"
                else:
                    return f"No file named '{file_name}' found. Please upload the file first."
            else:
                # Retrieve previous conversations for context
                previous_conversations = self.get_previous_conversations()
                context_messages = [{"role": "system", "content": "You are an AI assistant."}]

                # Add previous conversations to the context
                for conv in previous_conversations:
                    if "\n" in conv["message"]:
                        user_part, ai_part = conv["message"].split("\n", 1)
                        user_part = user_part.replace("User: ", "").strip()
                        ai_part = ai_part.replace("AI: ", "").strip()
                        context_messages.append({"role": "user", "content": user_part})
                        context_messages.append({"role": "assistant", "content": ai_part})
                    else:
                        user_part = conv["message"].replace("User: ", "").strip()
                        context_messages.append({"role": "user", "content": user_part})

                # Add the current user message
                context_messages.append({"role": "user", "content": user_message})

                # Retrieve the last agentic role from the database
                agentic_role = self.get_last_agentic_role()
                if agentic_role:
                    context_messages.append({"role": "system", "content": f"You are an AI assistant with the role: {agentic_role}."})

                # Generate a response from the AI model
                completion = self.client.chat.completions.create(
                    model=self.MODEL_NAME,
                    messages=context_messages,
                    extra_headers={"HTTP-Referer": "your_site_url", "X-Title": "your_app_name"}
                )
                if completion.choices and completion.choices[0].message:
                    response_message = completion.choices[0].message.content

                    # Save the conversation to the database
                    self.save_conversation(
                        f"User: {user_message}\nAI: {response_message}",
                        self.generate_summary(response_message),
                        self.memory_handler.sentence_to_vec(response_message),
                        agentic_role
                    )

                    return response_message
                else:
                    logging.error("Error processing query: No message found in API response.")
                    return None
        except Exception as e:
            logging.error(f"Error processing query: {str(e)}")
            return None

    def get_last_agentic_role(self):
        """
        Retrieve the last agentic role from the database.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT role FROM conversations ORDER BY timestamp DESC LIMIT 1")
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else None

    def generate_summary(self, message):
        """
        Generate a concise bulleted list summary of the message in 100 words or less.
        """
        summary_prompt = f"Summarize the following message in a concise bulleted list (100 words or less):\n\n{message}"
        
        try:
            completion = self.client.chat.completions.create(
                model=self.MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are an AI assistant tasked with summarizing messages."},
                    {"role": "user", "content": summary_prompt}
                ],
                extra_headers={
                    "HTTP-Referer": "your_site_url",  # Replace with your site URL
                    "X-Title": "your_app_name",       # Replace with your app name
                }
            )
            if completion.choices and completion.choices[0].message:
                return completion.choices[0].message.content.strip()
            else:
                logging.error("Error summarizing message: No message found in API response.")
                return message
        except Exception as e:
            logging.error(f"Error summarizing message: {str(e)}")
            return message

    def notify(self, message):
        """
        Notify the user by logging the message and displaying it in the chatbot UI if available.
        """
        logging.info(message)
        if self.chatbot_ui and hasattr(self.chatbot_ui, 'display_response'):
            self.chatbot_ui.display_response({"type": "text", "content": message})
        else:
            logging.warning("Chatbot UI is not initialized. Cannot display response.")