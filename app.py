import streamlit as st
import openai
import requests
from bs4 import BeautifulSoup
from transformers import pipeline

# Load keys
openai.api_key = st.secrets["OPEN_AI"]

# Cache Hugging Face sentiment model
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", use_auth_token=st.secrets["HUGGINGFACE_TOKEN"])

sentiment_analyzer = load_sentiment_model()

# Predict urgency
def predict_urgency(text):
    urgent_keywords = ["suicidal", "kill myself", "die", "hopeless", "urgent", "emergency", "end my life"]
    text_lower = text.lower()
    if any(word in text_lower for word in urgent_keywords):
        return "HIGH"
    sentiment = sentiment_analyzer(text)[0]
    if sentiment["label"] == "NEGATIVE" and sentiment["score"] > 0.9:
        return "MEDIUM"
    return "LOW"

# Suggest specialist
def suggest_specialist(text, sentiment_label):
    text = text.lower()
    if any(w in text for w in ["suicidal", "depressed", "anxious", "panic", "mental", "overwhelmed"]):
        return "psychologist"
    elif sentiment_label == "NEGATIVE":
        return "mental-health"
    return "general-physician"

# Fetch doctors from Practo (or fallback)
def fetch_doctors_live(specialist, city="chennai"):
    try:
        url = f"https://www.practo.com/{city}/{specialist}"
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")
        listings = soup.select(".listing-row")[:5]
        doctors = []
        for doc in listings:
            name = doc.select_one("h2")
            clinic = doc.select_one(".u-regular.u-color--grey-3")
            fee = doc.select_one(".fees")
            if name and clinic and fee:
                doctors.append(f"{name.get_text(strip=True)} – {clinic.get_text(strip=True)} – {fee.get_text(strip=True)}")
        return doctors if doctors else ["No doctors found."]
    except Exception as e:
        return [f"Could not fetch doctors: {str(e)}"]

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "system", "content": "You are a compassionate mental health assistant. Gently ask questions to help users express their feelings."}
    ]

st.title("🧠 Mental Health Chatbot + Doctor Suggestion")

# Chat input
user_input = st.chat_input("You:")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.spinner("Thinking..."):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # or "gpt-4" if available
            messages=st.session_state.chat_history
        )
        bot_reply = response["choices"][0]["message"]["content"]
        st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})

# Display chat
for msg in st.session_state.chat_history[1:]:  # Skip system prompt
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").write(msg["content"])

# Submit and analyze full conversation
if st.button("Submit and Analyze Conversation"):
    full_text = " ".join(m["content"] for m in st.session_state.chat_history if m["role"] == "user")
    sentiment = sentiment_analyzer(full_text)[0]
    urgency = predict_urgency(full_text)
    specialist = suggest_specialist(full_text, sentiment["label"])
    doctors = fetch_doctors_live(specialist)

    urgency_note = ""
    if urgency == "HIGH":
        urgency_note = "**⚠️ URGENT: Please seek help immediately or call a crisis line.**\n\n"
    elif urgency == "MEDIUM":
        urgency_note = "**⚠️ Medium urgency detected. You should talk to a specialist soon.**\n\n"

    analysis_response = (
        urgency_note +
        f"**Sentiment:** {sentiment['label']} (Confidence: {sentiment['score']:.2f})\n\n"
        f"**Recommended Specialist:** {specialist}\n\n"
        f"**Top Doctors Near You:**\n- " + "\n- ".join(doctors)
    )

    st.session_state.chat_history.append({"role": "assistant", "content": analysis_response})
    st.experimental_rerun()
