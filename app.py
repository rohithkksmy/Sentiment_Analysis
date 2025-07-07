import streamlit as st
import requests
from bs4 import BeautifulSoup
from transformers import pipeline

sentiment_analyzer = pipeline("sentiment-analysis", use_auth_token=st.secrets["HUGGINGFACE_TOKEN"])

def fetch_doctors_live(specialist, city="chennai"):
    # Your fetching code here (same as before)
    # For demo, returning dummy doctors:
    return ["Dr. A – Clinic X – ₹500", "Dr. B – Clinic Y – ₹600"]

def suggest_specialist(text, sentiment_label):
    text = text.lower()
    if any(word in text for word in ["sad", "depressed", "anxious", "stress", "tired", "suicidal"]):
        return "psychologist"
    elif sentiment_label == "NEGATIVE":
        return "mental-health"
    return "general-physician"

def predict_urgency(text):
    urgent_keywords = ["suicidal", "kill myself", "die", "death", "hopeless", "emergency", "urgent", "help me now"]
    text_lower = text.lower()
    if any(word in text_lower for word in urgent_keywords):
        return "HIGH"
    sentiment_result = sentiment_analyzer(text)[0]
    if sentiment_result['label'] == "NEGATIVE" and sentiment_result['score'] > 0.9:
        return "MEDIUM"
    return "LOW"

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("Chatbot with Instant Reply + Submit Analysis")

user_input = st.chat_input("You:")

if user_input:
    # Append user message
    st.session_state.messages.append({"role": "user", "message": user_input})
    # Append instant bot reply (can customize)
    st.session_state.messages.append({"role": "bot", "message": "Thanks for sharing! When ready, click Submit to analyze conversation."})

# Display messages
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["message"])
    else:
        st.chat_message("assistant").markdown(msg["message"])

# Submit button for full analysis
if st.button("Submit and Analyze Conversation"):
    user_text = " ".join(msg["message"] for msg in st.session_state.messages if msg["role"] == "user")
    sentiment = sentiment_analyzer(user_text)[0]
    urgency = predict_urgency(user_text)
    specialist = suggest_specialist(user_text, sentiment['label'])
    doctors = fetch_doctors_live(specialist)

    urgency_message = ""
    if urgency == "HIGH":
        urgency_message = "**⚠️ URGENT: Please seek immediate help or call emergency services!**\n\n"
    elif urgency == "MEDIUM":
        urgency_message = "**⚠️ Medium urgency detected. Consider reaching out to a specialist soon.**\n\n"

    bot_message = (
        urgency_message +
        f"**Sentiment:** {sentiment['label']} (Score: {sentiment['score']:.2f})\n\n"
        f"**Recommended Specialist:** {specialist}\n\n"
        f"**Nearby Doctors:**\n- " + "\n- ".join(doctors)
    )

    st.session_state.messages.append({"role": "bot", "message": bot_message})

    # Rerun to show new bot message immediately
    st.experimental_rerun()
