import streamlit as st
from transformers import pipeline

# Load a pre-trained conversational model (DialoGPT-medium)
chatbot = pipeline("conversational", model="microsoft/DialoGPT-medium")

# Streamlit UI
st.title("Simple Chatbot with Hugging Face")
st.write("Ask me anything!")

# Create a chat history (list to maintain conversation)
chat_history = []

# Function to get chatbot response
def get_bot_response(user_input):
    global chat_history
    # Get bot response
    bot_response = chatbot(user_input)
    # Append user and bot responses to the history
    chat_history.append({"role": "user", "content": user_input})
    chat_history.append({"role": "bot", "content": bot_response[0]['generated_text']})
    return bot_response[0]['generated_text']

# Display the conversation so far
if "conversation" not in st.session_state:
    st.session_state.conversation = []

# Add user input to the conversation history
user_input = st.text_input("You: ", "")

if user_input:
    # Get bot's response to user input
    bot_response = get_bot_response(user_input)
    # Store the conversation in the session state
    st.session_state.conversation.append(f"You: {user_input}")
    st.session_state.conversation.append(f"Bot: {bot_response}")

# Display chat history
for message in st.session_state.conversation:
    st.write(message)
