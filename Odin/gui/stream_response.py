# START OF FILE: C:\Users\Sean Craig\Desktop\AI Python Tools\Odin 2\gui\stream_response.py
from .message_parser import MessageParser

def stream_response(chatbot_ui, user_message):
    """
    Stream the AI response to the GUI, ensuring code blocks are displayed simply with a black background.
    """
    # Process the user message and get the AI response
    response = chatbot_ui.conversation_manager.process_query(user_message)
    if not response:
        chatbot_ui.response_queue.put({"type": "text", "content": "Error: No response from the AI model."})
        chatbot_ui.response_queue.put(None)  # Signal end of response
        return

    # Parse the response into messages
    parser = MessageParser()
    parsed_messages = parser.parse_response(response)

    # Send parsed messages to the queue
    for message in parsed_messages:
        chatbot_ui.response_queue.put(message)

    chatbot_ui.response_queue.put(None)  # Signal end of response

# END OF FILE: C:\Users\Sean Craig\Desktop\AI Python Tools\Odin 2\gui\stream_response.py