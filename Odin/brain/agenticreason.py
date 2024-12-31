import os
import json
import logging
from openai import OpenAI as Client
from .conversation_manager import ConversationManager
from .memory_handler import MemoryHandler

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class AgenticReasoner:
    def __init__(self, api_key, model_name="gpt-4"):
        """
        Initialize the Agentic Reasoner with an API key and model name.
        """
        self.OPEN_ROUTER_API_KEY = api_key
        self.MODEL_NAME = model_name
        self.client = Client(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.OPEN_ROUTER_API_KEY,
        )
        self.conversation_manager = ConversationManager()
        self.memory_handler = MemoryHandler(self.conversation_manager.memory_dir)
        self.role = None  # Initialize role as None, to be set dynamically
        logging.info("Agentic Reasoner initialized.")

    def analyze_conversation_and_set_role(self, user_message):
        """
        Get the role recommendation from the first AI assistant or use the role from memory.
        """
        # Check if there is a previous row in the memory table
        previous_row = self.conversation_manager.get_previous_conversation_row()
        if previous_row:
            # Use the agentic_role from the previous row
            self.role = previous_row.get("agentic_role")
            logging.info(f"Using agentic_role from previous row: {self.role}")
            print(f"Using agentic_role from previous row: {self.role}")
        else:
            # Get a new role recommendation from the first AI assistant
            role_recommendation = self.conversation_manager.get_role_recommendation(user_message)
            if role_recommendation:
                self.role = role_recommendation
                logging.info(f"Second AI Assistant Role: {self.role}")
                print(f"Second AI Assistant Role: {self.role}")
            else:
                logging.warning("No role recommendation received. Using default role.")
                self.role = "You are an AI assistant that helps refine and improve responses from another AI assistant. Analyze the response and make it more detailed, accurate, or user-friendly."
                print(f"Using default role: {self.role}")

    def process_query(self, user_message):
        """
        Process the user query in the following flow:
        1. Get role recommendation from the first AI assistant or use the role from memory.
        2. Original AI Assistant processes the query.
        3. Agentic Reasoner refines the response using the recommended role.
        4. Display the refined response.
        5. Generate bullet points for the refined response.
        6. Process the refined response with Word2Vec for memory and embeddings.
        7. Save the refined response, bullet points, and role to the database.
        """
        logging.info(f"Processing user query: {user_message}")

        # Step 1: Get role recommendation or use role from memory
        self.analyze_conversation_and_set_role(user_message)

        # Step 2: Original AI Assistant processes the query
        original_response = self.conversation_manager.process_query(user_message)
        if not original_response:
            logging.error("Original AI Assistant failed to generate a response.")
            raise ValueError("Original AI Assistant failed to generate a response.")
        logging.info(f"Original AI Assistant response: {original_response}")

        # Step 3: Agentic Reasoner refines the response using the recommended role
        try:
            completion = self.client.chat.completions.create(
                model=self.MODEL_NAME,
                messages=[
                    {"role": "system", "content": self.role},
                    {"role": "user", "content": original_response}
                ],
                extra_headers={"HTTP-Referer": "your_site_url", "X-Title": "your_app_name"}
            )
            if completion.choices and completion.choices[0].message:
                refined_response = completion.choices[0].message.content
                logging.info(f"Agentic Reasoner refined response: {refined_response}")
            else:
                refined_response = original_response
                logging.warning("Agentic Reasoner failed to refine the response. Using original response.")
        except Exception as e:
            refined_response = original_response
            logging.error(f"Error refining response: {str(e)}")

        # Step 4: Display the refined response
        logging.info(f"Displaying refined response: {refined_response}")

        # Step 5: Generate bullet points for the refined response
        bullet_points = self.generate_bullet_points(refined_response)
        logging.info(f"Generated bullet points: {bullet_points}")

        # Step 6: Process the refined response with Word2Vec for memory and embeddings
        embedding = self.process_response_with_word2vec(refined_response)
        if embedding:
            logging.info(f"Generated embedding: {embedding}")
        else:
            logging.warning("Failed to generate embedding for the response.")

        # Step 7: Save the refined response, bullet points, and role to the database
        self.conversation_manager.append_to_conversation(user_message, refined_response, self.role, bullet_points)
        logging.info("Refined response, bullet points, and role saved to the database.")

        return refined_response

    def generate_bullet_points(self, response):
        """
        Generate bullet points from the refined response.
        """
        sentences = response.split(". ")
        bullet_points = "\n".join([f"- {sentence.strip()}" for sentence in sentences if sentence.strip()])
        return bullet_points

    def process_response_with_word2vec(self, response):
        """
        Process the response with Word2Vec for memory and embeddings.
        Returns the embedding vector.
        """
        try:
            embedding = self.memory_handler.sentence_to_vec(response)
            if embedding:
                logging.info("Response processed with Word2Vec and embedded in memory.")
                return embedding
            else:
                logging.warning("Failed to generate Word2Vec embedding for the response.")
                return None
        except Exception as e:
            logging.error(f"Error processing response with Word2Vec: {str(e)}")
            return None