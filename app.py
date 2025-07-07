import streamlit as st
import requests
from bs4 import BeautifulSoup
from transformers import pipeline

# Initialize sentiment analyzer
sentiment_analyzer = pipeline("sentiment-analysis")

# Function to get doctors from Practo (top 5)
def fetch_doctors_live(specialist, city="chennai"):
    url = f"https://www.practo.com/{city}/{specialist}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code != 200:
            return ["Error fetching doctor data."]
        soup = BeautifulSoup(res.text, "html.parser")
        doctors = []
        listings = soup.select(".listing-row")[:5]
        for doc in listings:
            name = doc.select_one("h2")
            clinic = doc.select_one(".u-regular.u-color--grey-3")
            fee = doc.select_one(".fees")
            if name and clinic and fee:
                doctors.append(f"{name.get_text(strip=True)} – {clinic.get_text(strip=True)} – {fee.get_text(strip=True)}")
        return doctors if doctors else ["No doctors found."]
    except Exception as e:
        return [f"Error: {str(e)}"]

# Suggest specialist based on sentiment and text
def suggest_specialist(text, sentiment_label):
    text = text.lower()
    if any(word in text for word in ["sad", "depressed", "anxious", "stress", "tired", "suicidal"]):
        return "psychologist"
    elif sentiment_label == "NEGATIVE":
        return "mental-health"
    return "general-physician"

# Urgency detection function
def predict_urgency(text):
    urgent_keywords = ["suicidal", "kill myself", "die", "death", "hopeless", "emergency", "urgent", "help me now"]
    text_lower = text.lower()
    if any(word in text_lower for word in urgent_keywords):
        return "HIGH"
    sentiment_result = sentiment_analyzer(text)[0]
    sentiment_label = sentiment_result['label']
    sentiment_score = sentiment_result['score']
    if sentiment_label == "NEGATIVE" and sentiment_score > 0.9:
        return "MEDIUM"
    return "LOW"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

st.title("🧠 Chatbot with Submit Button for Sentiment, Urgency & Doctor Suggestions")

# Chat input
user_input = st.chat_input("You:")

if user_input:
    st.session_state.messages.append({"role": "user", "message": user_input})
    st.session_state.analysis_done = False  # reset analysis flag on new input

# Display chat messages
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["message"])
    else:
        st.chat_message("assistant").markdown(msg["message"])

# Submit button for final analysis
if st.button("Submit and Analyze Conversation"):
    with st.spinner("Analyzing conversation..."):
        user_text = " ".join([msg["message"] for msg in st.session_state.messages if msg["role"] == "user"])
        sentiment = sentiment_analyzer(user_text)[0]

        urgency = predict_urgency(user_text)

        specialist = suggest_specialist(user_text, sentiment['label'])
        doctors = fetch_doctors_live(specialist, city="chennai")

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
    st.session_state.analysis_done = True

    # Redisplay all chat including the bot analysis at the end
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.chat_message("user").write(msg["message"])
        else:
            st.chat_message("assistant").markdown(msg["message"])

# Optional: show a small note if analysis not done yet
if not st.session_state.analysis_done:
    st.info("Type your messages and click **Submit and Analyze Conversation** to see sentiment, urgency, and doctor suggestions.")
