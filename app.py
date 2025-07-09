import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the pre-trained model and tokenizer from Hugging Face
model_name = "microsoft/DialoGPT-medium"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set up conversation history
conversation_history = []

# Streamlit UI
st.title("Chat with the AI")
st.write("This is a simple chatbot built with Hugging Face's DialoGPT and Streamlit!")

# Get user input
user_input = st.text_input("You:", "")

# Process and respond to user input
if user_input:
    # Add the user's message to the conversation history
    conversation_history.append(f"User: {user_input}")
    
    # Tokenize the conversation history and the current user input
    new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")

    # Append the new input to the conversation history
    bot_input_ids = torch.cat([torch.tensor(conversation_history, dtype=torch.long), new_user_input_ids], dim=-1)

    # Generate the chatbot's response
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # Decode the generated response and add it to conversation history
    bot_output = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    conversation_history.append(f"Bot: {bot_output}")

    # Display the conversation
    st.text_area("Conversation History", value="\n".join(conversation_history), height=300)

