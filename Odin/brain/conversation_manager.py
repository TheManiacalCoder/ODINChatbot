import os
import datetime
import sqlite3
import logging
import json
from openai import OpenAI as Client
from .memory_handler import MemoryHandler

# Set up logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

class ConversationManager:
    def __init__(self):
        """
        Initialize the ConversationManager.
        """
        self.memory_dir = os.path.join(os.path.dirname(__file__), "Memory")
        os.makedirs(self.memory_dir, exist_ok=True)
        self.conv_folder = None
        self.db_path = None
        self.MODEL_NAME = self._load_model_name()
        self.OPEN_ROUTER_API_KEY = None
        self.client = None
        self.memory_handler = MemoryHandler(self.memory_dir)
        self.init_conversation()

    def _load_model_name(self):
        """
        Load the MODEL_NAME from the config.json file.
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
        Initialize the conversation by creating a new database file and setting up the memory handler.
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.conv_folder = os.path.join(self.memory_dir, f"memory_{timestamp}")
        os.makedirs(self.conv_folder, exist_ok=True)
        self.db_path = os.path.join(self.conv_folder, "conversations.db")
        self.memory_handler = MemoryHandler(self.conv_folder)
        self.init_db()
        self.add_agentic_role_column()
        if not os.path.exists(self.memory_handler.model_path):
            logging.info("Initializing Word2Vec model with default data.")
            self.memory_handler.load_or_train_word2vec_model()

    def init_db(self):
        """
        Initialize the SQLite database with the necessary table.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                message TEXT,  -- Full message (user query + AI response)
                message_summary TEXT,  -- Bulleted list summary
                embedding TEXT,  -- Embedding based on the full message
                agentic_role TEXT  -- Dynamic role based on the message summary
            )''')
        conn.commit()
        conn.close()
        logging.info(f"Database initialized at {self.db_path}")

    def add_agentic_role_column(self):
        """
        Add the agentic_role column to the conversations table if it doesn't exist.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute("ALTER TABLE conversations ADD COLUMN agentic_role TEXT")
            conn.commit()
            logging.info("Added agentic_role column to the conversations table.")
        except sqlite3.OperationalError as e:
            logging.warning(f"Column agentic_role already exists: {str(e)}")
        finally:
            conn.close()

    def set_openrouter_api_key(self, api_key):
        """
        Set the OpenRouter API key and update the client.
        """
        self.OPEN_ROUTER_API_KEY = api_key
        self.update_client()

    def set_model_name(self, model_name):
        """
        Set the model name and update the client.
        """
        self.MODEL_NAME = model_name
        self.update_client()

    def update_client(self):
        """
        Update the OpenAI client with the current API key and model name.
        """
        if self.MODEL_NAME and self.OPEN_ROUTER_API_KEY:
            self.client = Client(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.OPEN_ROUTER_API_KEY,
            )
            logging.info("OpenAI client updated with new API key and model name.")

    def process_query(self, user_message):
        """
        Process a user query and generate a response using the AI model.
        """
        if not self.client:
            logging.error("OpenAI client is not initialized.")
            return "Error: OpenAI client is not initialized."

        # Retrieve previous conversations for context
        previous_conversations = self.get_previous_conversations()
        context_messages = [{"role": "system", "content": "You are an AI assistant."}]
        for conv in previous_conversations:
            context_messages.append({"role": "user", "content": conv["user_message"]})
            context_messages.append({"role": "assistant", "content": conv["ai_response"]})

        # Add the current user message
        context_messages.append({"role": "user", "content": user_message})

        # Generate a response from the AI model
        try:
            completion = self.client.chat.completions.create(
                model=self.MODEL_NAME,
                messages=context_messages,
                extra_headers={"HTTP-Referer": "your_site_url", "X-Title": "your_app_name"}
            )
            if completion.choices and completion.choices[0].message:
                response_message = completion.choices[0].message.content

                # Combine user query and AI response into a full message
                full_message = f"User: {user_message}\nAI: {response_message}"

                # Generate a bulleted list summary of the full message
                summary = self.generate_bulleted_summary(full_message)

                # Generate a dynamic role based on the message summary
                dynamic_role = self.generate_dynamic_role(summary)

                # Generate an embedding for the full message
                embedding = self.memory_handler.sentence_to_vec(full_message)

                # Save the conversation to the database
                self.save_conversation(full_message, summary, embedding, dynamic_role)

                return response_message
            else:
                logging.error("Error processing query: No message found in API response.")
                return None
        except Exception as e:
            logging.error(f"Error processing query: {str(e)}")
            return None

    def generate_bulleted_summary(self, message):
        """
        Generate a bulleted list summary of the message.
        """
        # Split the message into sentences
        sentences = message.split(". ")
        # Create a bulleted list from the sentences
        bulleted_summary = "\n".join([f"- {sentence.strip()}" for sentence in sentences if sentence.strip()])
        return bulleted_summary

    def generate_dynamic_role(self, summary):
        """
        Generate a dynamic role based on the message summary.
        If the summary is too short or general, assign a default role.
        """
        # Default role for general conversations
        default_role = "General Assistant - Provide helpful and accurate responses."

        # If the summary is too short or general, return the default role
        if len(summary.split()) <= 5:  # Adjust the threshold as needed
            logging.info("Summary is too short or general. Assigning default role.")
            return default_role

        # Use the AI model to generate a role based on the summary
        try:
            completion = self.client.chat.completions.create(
                model=self.MODEL_NAME,
                messages=[
                    {"role": "system", "content": "Generate a concise, professional role description based on the conversation summary. Focus solely on the role and its purpose. Example: 'Dating Coach - Provide actionable dating advice.'"},
                    {"role": "user", "content": summary}
                ],
                extra_headers={"HTTP-Referer": "your_site_url", "X-Title": "your_app_name"}
            )
            if completion.choices and completion.choices[0].message:
                dynamic_role = completion.choices[0].message.content
                return dynamic_role
            else:
                logging.error("Error generating dynamic role: No message found in API response.")
                return default_role
        except Exception as e:
            logging.error(f"Error generating dynamic role: {str(e)}")
            return default_role

    def save_conversation(self, full_message, summary, embedding, dynamic_role):
        """
        Save the conversation to the database.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute('''INSERT INTO conversations (timestamp, message, message_summary, embedding, agentic_role)
                          VALUES (?, ?, ?, ?, ?)''',
                       (timestamp, full_message, summary, str(embedding.tolist()) if embedding is not None else None, dynamic_role))
        conn.commit()
        conn.close()
        logging.info("Conversation saved to the database.")

    def get_previous_conversations(self):
        """
        Retrieve previous conversations from the database.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT message FROM conversations ORDER BY timestamp DESC LIMIT 5")
        rows = cursor.fetchall()
        conn.close()

        conversations = []
        for row in rows:
            message = row[0]
            if "\n" in message:
                user_message, ai_response = message.split("\n", 1)
                conversations.append({
                    "user_message": user_message.replace("User: ", "").strip(),
                    "ai_response": ai_response.replace("AI: ", "").strip()
                })
        return conversations