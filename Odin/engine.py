import os
import json
import customtkinter as ctk
from gui.app import run_gui_wrapper
from brain.ai_memory import conversation_manager
from brain.agenticreason import AgenticReasoner
from config import OPEN_ROUTER_API_KEY, MODEL_NAME

# Initialize the appearance and theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Set the OpenRouter API key and model name for the original assistant
conversation_manager.set_openrouter_api_key(OPEN_ROUTER_API_KEY)
conversation_manager.set_model_name(MODEL_NAME)

# Initialize the Agentic Reasoner
agentic_reasoner = AgenticReasoner(api_key=OPEN_ROUTER_API_KEY, model_name=MODEL_NAME)

def run_engine():
    """
    Main function to run the Odin engine and GUI.
    """
    root = ctk.CTk()
    root.title("Odin")

    # Set the window size and position
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    window_width = int(screen_width * 0.25)
    window_height = int(screen_height * 0.85)

    window_x = (screen_width - window_width) // 2
    window_y = (screen_height - window_height) // 2

    root.geometry(f"{window_width}x{window_height}+{window_x}+{window_y}")
    root.configure(fg_color="#000000")

    # Create a container for the GUI
    container = ctk.CTkFrame(root, fg_color="#000000", border_width=0)
    container.pack(padx=20, pady=20, fill=ctk.BOTH, expand=True)

    # Run the GUI wrapper
    run_gui_wrapper(container)

    # Start the main loop
    root.mainloop()

def send_message(user_message):
    """
    Process the user message using the following flow:
    1. Original AI Assistant processes the query.
    2. Agentic Reasoner refines the response.
    3. Display the refined response.
    4. Process the response with Word2Vec for memory and embeddings.
    """
    try:
        # Step 1: Original AI Assistant processes the query
        original_response = conversation_manager.process_query(user_message)
        if not original_response:
            raise ValueError("Original AI Assistant failed to generate a response.")

        # Step 2: Agentic Reasoner refines the response
        refined_response = agentic_reasoner.process_query(original_response)

        # Step 3: Display the refined response
        chatbot_ui.display_response({"type": "text", "content": refined_response})

        # Step 4: Process the refined response with Word2Vec
        agentic_reasoner.process_response_with_word2vec(refined_response)

    except Exception as e:
        print(f"Error processing user message: {str(e)}")
        chatbot_ui.display_response({"type": "text", "content": f"Error: {str(e)}"})

if __name__ == "__main__":
    run_engine()