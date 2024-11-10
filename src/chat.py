import streamlit as st
import time


LANGUAGES = ["English", "Hungarian", "Danish", "German", "Italian", "Spanish", "French", "Portuguese", "Dutch", "Swedish", "Norwegian", "Finnish"]
LEVELS = ["Beginner", "Intermediate", "Advanced"]

def show_chat_history() -> None:
    """
    Displays the chat history stored in the Streamlit session state.
    Iterates through the messages stored in `st.session_state.messages` and 
    displays each message in the chat interface. Each message is expected to 
    be a tuple where the first element is the sender and the second element 
    is the message content.
    Returns:
        None
    """
    for message in st.session_state.messages:
        with st.chat_message(message[0]):
            st.markdown(message[1])

def stream_output(response):
    """
    Generator function that yields words from the given response one by one, 
    with a short delay between each word.
    Args:
        response (str): The input string to be split into words and streamed.
    Yields:
        str: The next word in the response followed by a space.
    """
    for word in response.split():
        yield word + " "
        time.sleep(0.05)