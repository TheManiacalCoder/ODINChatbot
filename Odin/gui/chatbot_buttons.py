import customtkinter as ctk

class ChatbotButtons:
    def __init__(self, master, chatbot_ui):
        self.master = master
        self.chatbot_ui = chatbot_ui

        self.button_frame = ctk.CTkFrame(self.master)
        self.button_frame.pack(expand=True, fill=ctk.X, padx=10, pady=10)

        self.new_conversation_button = ctk.CTkButton(
            self.button_frame, 
            text="New Conversation", 
            command=self.new_conversation, 
            width=180,
            height=50,
            fg_color="#000000",  
            corner_radius=0,     
            font=("Segoe UI", 15)  
        )
        self.new_conversation_button.pack(side=ctk.LEFT, padx=10, pady=10)

        self.clear_chat_button = ctk.CTkButton(
            self.button_frame, 
            text="Clear Chat", 
            command=self.clear_chat, 
            width=180,
            height=50,
            fg_color="#000000",  
            corner_radius=0,     
            font=("Segoe UI", 15)  
        )
        self.clear_chat_button.pack(side=ctk.LEFT, padx=10, pady=10)

        self.update_model_button = ctk.CTkButton(
            self.button_frame, 
            text="Update Model", 
            command=self.update_model, 
            width=180,
            height=50,
            fg_color="#000000",  
            corner_radius=0,     
            font=("Segoe UI", 15)  
        )
        self.update_model_button.pack(side=ctk.LEFT, padx=10, pady=10)

        self.model_name_entry = ctk.CTkEntry(
            self.button_frame, 
            font=("Segoe UI", 18),
            placeholder_text=self.chatbot_ui.conversation_manager.MODEL_NAME,
            width=700 - 60
        )
        self.model_name_entry.pack(side=ctk.LEFT, padx=(10, 10), pady=(10, 10))

    def new_conversation(self):
        # Ensure we properly clear the chat and start new one
        self.chatbot_ui.clear_chat(new_conversation=True)

    def clear_chat(self):
        # Clear chat without starting a new conversation
        self.chatbot_ui.clear_chat(new_conversation=False)

    def update_model(self):
        new_model_name = self.model_name_entry.get().strip()
        if new_model_name:
            self.chatbot_ui.conversation_manager.set_model_name(new_model_name)
            self.chatbot_ui.widgets['text_box'].configure(state="normal")
            self.chatbot_ui.widgets['text_box'].insert('end', f"Model updated to {new_model_name}.\n", "assistant")
            self.chatbot_ui.widgets['text_box'].configure(state="disabled")
            self.chatbot_ui.widgets['text_box'].yview('end')