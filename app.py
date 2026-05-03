import streamlit as st
from transformers import pipeline
from gtts import gTTS
import os

# --- 1. SETUP MODELS ---
# We use the pipeline API for simplicity and readability
@st.cache_resource
def load_models():
    captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    # GPT-2 needs a bit of help to stay "on track" for kids
    story_gen = pipeline("text-generation", model="gpt2") 
    return captioner, story_gen

caption_model, story_model = load_models()

# --- 2. THE APP INTERFACE ---
st.title("🪄 Magic Story Machine")
st.write("Upload a photo to turn it into a magical adventure!")

uploaded_file = st.file_uploader("Choose a picture", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Save and Display Image
    with open("temp_img.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.image(uploaded_file, caption="Your Picture", use_container_width=True)

    # STAGE 1: Image to Text
    with st.spinner("Looking at your picture..."):
        caption_result = caption_model("temp_img.jpg")
        caption = caption_result[0]['generated_text']
        st.info(f"**I see:** {caption}")

    # STAGE 2: Text to Story (Revised for Length)
    with st.spinner("Writing a long magical story..."):
        # We give GPT-2 a very specific starting point to encourage length
        prompt = (
            f"Once upon a time, there was {caption}. "
            "It was a magical day and something amazing happened. "
            "First,"
        )
        
        # Setting min_new_tokens ensures the story isn't too short
        story_output = story_model(
            prompt, 
            max_new_tokens=150,  # Allows for ~100-150 words
            min_new_tokens=80,   # Forces it to be at least ~60-80 words
            do_sample=True, 
            temperature=0.7,
            truncation=True
        )
        story_text = story_output[0]['generated_text']
        
        st.subheader("📖 Your Story")
        st.write(story_text)

    # STAGE 3: Story to Audio
    with st.spinner("Preparing the audio..."):
        tts = gTTS(text=story_text, lang='en')
        tts.save("story.mp3")
        st.audio("story.mp3")
        st.balloons()
