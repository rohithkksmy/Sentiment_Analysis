import streamlit as st
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer

# Create and train the chatbot
chatbot = ChatBot(
    'SimpleBot',
    storage_adapter='chatterbot.storage.SQLStorageAdapter',
    logic_adapters=[
        'chatterbot.logic.BestMatch',
        'chatterbot.logic.MathematicalEvaluation'
    ],
    database_uri='sqlite:///database.db'
)

# Train the chatbot on the English corpus
trainer = ChatterBotCorpusTrainer(chatbot)
trainer.train('chatterbot.corpus.english')

# Streamlit interface
st.title("Free Chatbot with ChatterBot")

st.write("Ask anything to the chatbot:")

# Input box for the user
user_input = st.text_input("You: ", "")

# Chatbot response logic
if user_input:
    with st.spinner("Thinking..."):
        bot_response = chatbot.get_response(user_input)
    st.write(f"Chatbot: {bot_response}")
