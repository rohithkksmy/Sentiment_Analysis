import streamlit as st
from transformers import pipeline
from bs4 import BeautifulSoup
import requests

# Initialize Hugging Face DialoGPT model for chat using text-generation task
chatbot = pipeline("text-generation", model="microsoft/DialoGPT-medium")

# Initialize sentiment analysis model (for mood detection)
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

# Suggest specialist based on the detected sentiment or text
def suggest_specialist(sentiment):
    sentiment = sentiment.lower()
    if "negative" in sentiment:
        return "psychologist"
    elif "positive" in sentiment:
        return "general-physician"
    return "general-physician"

# Streamlit chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🧠 Real-Time Sentiment Chatbot with Live Doctor Search")

# User input
user_input = st.text_input("You:")

if user_input:
    st.session_state.messages.append({"role": "user", "message": user_input})

    # Use Hugging Face chatbot to generate a response
    bot_message = chatbot(user_input, max_length=50, num_return_sequences=1)[0]["generated_text"]
    
    st.session_state.messages.append({"role": "bot", "message": bot_message})

# Show chat messages
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["message"])
    else:
        st.chat_message("assistant").markdown(msg["message"])

# Add a Submit button to analyze sentiment and recommend doctor
if st.button("Submit"):
    # Concatenate user messages to analyze sentiment
    user_text = " ".join([msg["message"] for msg in st.session_state.messages if msg["role"] == "user"])

    # Analyze sentiment
    sentiment = sentiment_analyzer(user_text)[0]
    mood = sentiment["label"]
    sentiment_score = sentiment["score"]

    # Suggest specialist based on sentiment
    specialist = suggest_specialist(mood)
    
    # Fetch doctors based on suggested specialist
    doctors = fetch_doctors_live(specialist, city="chennai")
    
    # Display results
    st.write(f"**Mood:** {mood} (Sentiment Score: {sentiment_score:.2f})")
    st.write(f"**Recommended Specialist:** {specialist}")
    st.write(f"**Nearby Doctors:**")
    for doctor in doctors:
        st.write(f"- {doctor}")
