import streamlit as st
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
from textblob import TextBlob

# Create a new chatbot instance
chatbot = ChatBot(
    'DoctorBot',
    storage_adapter='chatterbot.storage.SQLStorageAdapter',
    logic_adapters=[
        'chatterbot.logic.BestMatch',
        'chatterbot.logic.MathematicalEvaluation',
        'chatterbot.logic.TimeLogicAdapter'
    ],
    database_uri='sqlite:///database.db'  # Stores conversations locally
)

# Train the chatbot with the English corpus
trainer = ChatterBotCorpusTrainer(chatbot)
trainer.train('chatterbot.corpus.english')

# Function for sentiment analysis
def get_sentiment(text):
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity
    return sentiment_score

# Streamlit UI setup
st.title("DoctorBot - Your Personal Assistant")

# Initialize chat history in session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Display the chat history
if st.session_state.chat_history:
    for message in st.session_state.chat_history:
        st.write(message)

# Collect user input
user_input = st.text_input("Type your message:")

# When user submits a message
if user_input:
    # Add user message to the history
    st.session_state.chat_history.append(f"You: {user_input}")
    
    # Perform sentiment analysis on the input
    sentiment_score = get_sentiment(user_input)
    
    # Generate chatbot's response using ChatterBot
    bot_reply = chatbot.get_response(user_input)
    
    # Analyze mood based on sentiment
    if sentiment_score > 0:
        mood = "Positive"
    elif sentiment_score < 0:
        mood = "Negative"
    else:
        mood = "Neutral"
    
    # Add bot reply and mood analysis to chat history
    st.session_state.chat_history.append(f"Bot: {bot_reply}")
    st.session_state.chat_history.append(f"Bot: Your mood is {mood} based on the analysis.")
    
    # Display the chat history
    for message in st.session_state.chat_history:
        st.write(message)

# Add a "Reset Chat" button
if st.button("Reset Chat"):
    st.session_state.chat_history = []
