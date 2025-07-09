import streamlit as st
from transformers import pipeline, Conversation
import os

# Access Hugging Face token from secrets
hf_token = st.secrets["HUGGINGFACE_TOKEN"]["token"]

# Set the Hugging Face token for authentication (you can set it as an environment variable)
os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token

# Alternatively, you can log in with the token (if needed)
# from huggingface_hub import login
# login(token=hf_token)

# Load a pre-trained conversational model (DialoGPT-medium)
chatbot = pipeline("conversational", model="microsoft/DialoGPT-medium")

# Streamlit UI
st.title("Simple Chatbot with Hugging Face")
st.write("Ask me anything!")

# Initialize conversation history if it doesn't exist
if "conversation" not in st.session_state:
    st.session_state.conversation = []

# Function to get chatbot response
def get_bot_response(user_input):
    # Create a new Conversation instance
    conversation = Conversation(user_input)
    # Get bot response using the chatbot pipeline
    bot_response = chatbot(conversation)
    return bot_response[-1]['generated_text']

# Display user input and bot response in the conversation history
user_input = st.text_input("You: ", "")

if user_input:
    # Get bot's response to user input
    bot_response = get_bot_response(user_input)
    # Add both user input and bot response to the session state
    st.session_state.conversation.append(f"You: {user_input}")
    st.session_state.conversation.append(f"Bot: {bot_response}")

# Display chat history
for message in st.session_state.conversation:
    st.write(message)
