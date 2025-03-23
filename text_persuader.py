import streamlit as st
from transformers import pipeline

# Initialize sentiment and text generation models
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
text_generator = pipeline("text2text-generation", model="facebook/bart-large-cnn")

st.title("Text Persuasion Assistant")

# Input text area
original_text = st.text_area("Enter your text (up to 100 words):", max_chars=500)

# Context selection
context = st.selectbox(
    "Select the context/tone for adaptation:",
    ["Professional/Formal", "Friendly/Casual", "Persuasive/Sales", "Academic/Technical"]
)

if st.button("Analyze and Adapt") and original_text:
    # Analyze sentiment
    sentiment = sentiment_analyzer(original_text)[0]
    st.subheader("Sentiment Analysis:")
    st.write(f"Tone: {sentiment['label']} (Confidence: {sentiment['score']:.2%})")
    
    # Generate context prompt based on selection
    context_prompts = {
        "Professional/Formal": "Rewrite this professionally and formally: ",
        "Friendly/Casual": "Rewrite this in a friendly and casual tone: ",
        "Persuasive/Sales": "Rewrite this to be more persuasive and compelling: ",
        "Academic/Technical": "Rewrite this in an academic and technical style: "
    }
    
    # Generate adapted text
    prompt = context_prompts[context] + original_text
    adapted_text = text_generator(prompt, max_length=150, min_length=50)[0]['generated_text']
    
    st.subheader("Adapted Text:")
    st.write(adapted_text) 