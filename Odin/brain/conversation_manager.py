import os
import datetime
import sqlite3
import logging
import json
import re  # For extracting content between ** **
from openai import OpenAI as Client
from .memory_handler import MemoryHandler

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class ConversationManager:
    def __init__(self):
        """
        Initialize the ConversationManager with default settings.
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
        self.memory_handler = MemoryHandler(self.conv_folder)  # Initialize MemoryHandler

        # Initialize the database
        self.init_db()

        # Add the agentic_role column if it doesn't exist
        self.add_agentic_role_column()

        # Initialize Word2Vec model only if it doesn't exist
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
                message TEXT,
                message_summary TEXT,
                embedding TEXT,
                agentic_role TEXT  -- New column for Agentic Reasoner role
            )''')
        conn.commit()
        conn.close()

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

    def get_previous_conversation_row(self):
        """
        Retrieve the previous row from the conversations table.
        """
        if self.db_path and os.path.exists(self.db_path):
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM conversations ORDER BY timestamp DESC LIMIT 1")
            row = cursor.fetchone()
            conn.close()
            if row:
                return {
                    "id": row[0],
                    "timestamp": row[1],
                    "message": row[2],
                    "message_summary": row[3],
                    "embedding": row[4],
                    "agentic_role": row[5]
                }
        return None

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

    def get_role_recommendation(self, user_message):
        """
        Get a role recommendation from the first AI based on the user's message.
        """
        if not self.client:
            logging.error("OpenAI client is not initialized.")
            return None

        try:
            # Summarize the user's message
            summary = self.summarize_message(user_message)

            # Ask the AI to recommend a role based on the summary
            role_prompt = f"Based on the following summary of the user's message, recommend a suitable role for an AI assistant to analyze and respond to it. Return the role in the format **Job Title - Job description.**:\n\nSummary: {summary}"
            completion = self.client.chat.completions.create(
                model=self.MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are an AI assistant that recommends roles for other AI assistants based on user prompts. Return the role in the format **Job Title - Job description.**."},
                    {"role": "user", "content": role_prompt}
                ],
                extra_headers={"HTTP-Referer": "your_site_url", "X-Title": "your_app_name"}
            )
            if completion.choices and completion.choices[0].message:
                return completion.choices[0].message.content.strip()
            else:
                logging.error("No role recommendation received from the AI.")
                return None
        except Exception as e:
            logging.error(f"Error getting role recommendation: {str(e)}")
            return None

    def extract_content_from_markdown(self, role_text):
        """
        Extract the content between ** ** from the markdown-wrapped text.
        """
        match = re.search(r"\*\*(.*?)\*\*", role_text)
        if match:
            return match.group(1)  # Return the content between ** **
        return role_text  # Fallback if no markdown is found

    def process_query(self, user_message):
        """
        Process a user query and generate a response using the AI model.
        """
        if not self.client:
            logging.error("OpenAI client is not initialized.")
            return "Error: OpenAI client is not initialized."

        # Load the conversation history from the database
        conversation_history = self.load_conversation_from_db()

        # Add the system message at the beginning of the conversation history
        conversation_history.insert(0, {"role": "system", "content": "You are an AI assistant. I will remember our conversation and provide relevant responses based on previous interactions."})

        # Add the user's current message
        conversation_history.append({"role": "user", "content": user_message})

        # Generate a response from the AI model
        try:
            completion = self.client.chat.completions.create(
                model=self.MODEL_NAME,
                messages=conversation_history,
                extra_headers={"HTTP-Referer": "your_site_url", "X-Title": "your_app_name"}
            )
            if completion.choices and completion.choices[0].message:
                response_message = completion.choices[0].message.content
                # Save the full message (User Query + AI Response) to the database
                self.append_to_conversation(user_message, response_message)
                return response_message
            else:
                logging.error("Error processing query: No message found in API response.")
                return None
        except Exception as e:
            logging.error(f"Error processing query: {str(e)}")
            return None

    def append_to_conversation(self, user_query, ai_response=None, agentic_role=None, bullet_points=None):
        """
        Append a new conversation entry to the database.
        Ensure all fields (message_summary, embedding, agentic_role) are populated.
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Combine the user query and AI response
        combined_message = f"User: {user_query}\nAI: {ai_response}" if ai_response else f"User: {user_query}"

        # Step 1: Generate a summary of the message (bullet points)
        message_summary = self.summarize_message(combined_message)

        # Step 2: Convert the message to a vector using Word2Vec
        embedding = self.memory_handler.sentence_to_vec(combined_message)
        embedding_str = ",".join(map(str, embedding)) if embedding is not None else ""

        # Step 3: Get the agentic role (if not provided)
        if agentic_role is None:
            agentic_role = self.get_role_recommendation(user_query)
            if agentic_role is None:
                agentic_role = "**Default Assistant - Provides general assistance and support**"

        # Extract the content between ** **
        extracted_role = self.extract_content_from_markdown(agentic_role)
        print(extracted_role)  # Print the extracted role and description

        # Step 4: Save the conversation to the database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO conversations (timestamp, message, message_summary, embedding, agentic_role) VALUES (?, ?, ?, ?, ?)",
                       (timestamp, combined_message, message_summary, embedding_str, extracted_role))
        conn.commit()
        conn.close()

        # Step 5: Update the Word2Vec model with the new message
        self.memory_handler.update_word2vec_model(combined_message)

    def summarize_message(self, message):
        """
        Summarize the message into bullet points.
        Ensure a summary is always generated, even for short or single-word messages.
        """
        try:
            # Split the message into sentences and format as bullet points
            sentences = message.split(". ")
            bullet_points = "\n".join([f"- {sentence.strip()}" for sentence in sentences if sentence.strip()])
            return bullet_points if bullet_points else f"- {message.strip()}"  # Fallback to the full message
        except Exception as e:
            logging.error(f"Error summarizing message: {str(e)}")
            return f"- {message.strip()}"  # Fallback to the full message

    def load_conversation_from_db(self):
        """
        Load the conversation history from the database.
        """
        conversation_history = []
        if self.db_path and os.path.exists(self.db_path):
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT message FROM conversations ORDER BY timestamp ASC")
            rows = cursor.fetchall()
            for row in rows:
                message = row[0]
                if "\n" in message:
                    user_part, ai_part = message.split("\n", 1)
                    user_part = user_part.replace("User: ", "").strip()
                    ai_part = ai_part.replace("AI: ", "").strip()
                    conversation_history.append({"role": "user", "content": user_part})
                    conversation_history.append({"role": "assistant", "content": ai_part})
                else:
                    user_part = message.replace("User: ", "").strip()
                    conversation_history.append({"role": "user", "content": user_part})
            conn.close()
        return conversation_history

    def clear_conversation(self, new_conversation=False):
        """
        Clear the current conversation and optionally start a new one.
        """
        if self.conv_folder:
            for root, dirs, files in os.walk(self.conv_folder, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(self.conv_folder)
        
        if new_conversation:
            self.init_conversation()