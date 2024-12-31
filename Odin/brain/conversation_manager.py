import os
import json
import datetime
import sqlite3
import logging
import numpy as np
from openai import OpenAI as Client
from .memory_handler import MemoryHandler

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class ConversationManager:
    def __init__(self):
        self.memory_dir = os.path.join(os.path.dirname(__file__), "Memory")
        os.makedirs(self.memory_dir, exist_ok=True)
        self.conv_folder = None
        self.db_path = None
        self.MODEL_NAME = "gpt-4"  # Default model name
        self.OPEN_ROUTER_API_KEY = None  # API key for OpenRouter
        self.client = None  # OpenAI client
        self.memory_handler = None
        self.init_conversation()

    def init_conversation(self):
        """
        Initialize the conversation by creating a new database file.
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.conv_folder = os.path.join(self.memory_dir, f"memory_{timestamp}")
        os.makedirs(self.conv_folder, exist_ok=True)
        self.db_path = os.path.join(self.conv_folder, "conversations.db")
        self.memory_handler = MemoryHandler(self.conv_folder)

        # Initialize the database
        self.init_db()

    def init_db(self):
        """
        Initialize the SQLite database with the necessary table.
        """
        self.memory_handler.init_db(self.db_path)

    def set_openrouter_api_key(self, api_key):
        """
        Set the OpenRouter API key and update the OpenAI client.
        """
        self.OPEN_ROUTER_API_KEY = api_key
        self.update_client()

    def set_model_name(self, model_name):
        """
        Set the model name and update the OpenAI client.
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

    def load_conversation_from_db(self):
        """
        Load the conversation from the database and return it as a list of messages.
        """
        if not os.path.exists(self.db_path):
            logging.warning("Database file does not exist.")
            return []

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Fetch all rows from the conversations table
        cursor.execute("SELECT timestamp, message, message_summary FROM conversations ORDER BY timestamp ASC")
        rows = cursor.fetchall()

        conversation_history = []
        for row in rows:
            timestamp, message, summary = row
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
                # Chain prompt to summarize the AI response
                summary = self.chain_prompt_summarize(response_message)
                logging.info(f"AI response summarized: {summary}")
                # Save the full message (User Query + AI Response) to the database
                self.append_to_conversation(user_message, response_message, summary)
                return response_message
            else:
                logging.error("Error processing query: No message found in API response.")
                return None
        except Exception as e:
            logging.error(f"Error processing query: {str(e)}")
            return None

    def chain_prompt_summarize(self, message):
        """
        Chain prompt the AI to summarize the message into concise bullet points.
        """
        try:
            completion = self.client.chat.completions.create(
                model=self.MODEL_NAME,
                messages=[
                    {"role": "system", "content": "Summarize the following message into concise bullet points, capturing all main points. Use bullet points (•) for each key point. Be concise and do not include the full story."},
                    {"role": "user", "content": message}
                ],
                extra_headers={"HTTP-Referer": "your_site_url", "X-Title": "your_app_name"}
            )
            if completion.choices and completion.choices[0].message:
                return completion.choices[0].message.content
            else:
                logging.error("Error summarizing message: No message found in API response.")
                return None
        except Exception as e:
            logging.error(f"Error summarizing message: {str(e)}")
            return None

    def append_to_conversation(self, user_query, ai_response=None, summary=None):
        """
        Append a new conversation entry to the database.
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        combined_message = f"User: {user_query}\nAI: {ai_response}" if ai_response else f"User: {user_query}"

        # Generate the summary if it's not provided
        if summary is None:
            summary = self.chain_prompt_summarize(combined_message)

        # Log the summary (or print it if needed)
        logging.info(f"Summary generated: {summary}")

        # Convert the summary to a vector for embedding
        embedding = self.memory_handler.sentence_to_vec(summary)  # Train on the summary, not the raw message
        embedding_str = ",".join(map(str, embedding)) if embedding is not None else ""

        # Save the conversation to the database
        self.memory_handler.save_conversation_to_db(self.db_path, combined_message, embedding_str, timestamp, summary)

        # Update the Word2Vec model with the summary
        self.memory_handler.update_word2vec_model(summary)

    def clear_conversation(self, new_conversation=False):
        """
        Clear the current conversation.
        """
        if self.conv_folder:
            for root, dirs, files in os.walk(self.conv_folder, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(self.conv_folder)
        
        self.init_conversation()

# Define the conversation manager instance
conversation_manager = ConversationManager()
conversation_manager.set_model_name("gpt-4")  # Set your desired model name
conversation_manager.set_openrouter_api_key("your_openrouter_api_key")  # Set your OpenRouter API key
conversation_manager.init_conversation()

# Example usage:
user_query = "Tell me a story about a brave knight who saved a kingdom from a dragon."
response = conversation_manager.process_query(user_query)
print(f"AI Response: {response}")