import streamlit as st
from transformers import pipeline
from gtts import gTTS
from PIL import Image
import os

# --- STEP 1: MODULAR FUNCTIONS ---

def generate_caption(image):
    """Processes image and returns a caption using Hugging Face BLIP model."""
    # Using the Salesforce/blip-image-captioning-base model suggested in guidelines
    captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    result = captioner(image)
    return result[0]['generated_text']

def generate_story(caption):
    """Expands the caption into a 50-100 word story for kids aged 3-10."""
    story_gen = pipeline("text-generation", model="gpt2")
    
    # Constructing a kid-friendly prompt
    prompt = f"Write a happy, short story for 5-year-old kids based on this: {caption}. The story should be about friendship and fun."
    
    # Generating the story (max_new_tokens helps hit the 50-100 word requirement)
    result = story_gen(prompt, max_new_tokens=100, min_new_tokens=60, temperature=0.7, do_sample=True)
    return result[0]['generated_text']

def text_to_speech(text):
    """Converts the story text into an MP3 audio file."""
    tts = gTTS(text=text, lang='en')
    audio_path = "story_audio.mp3"
    tts.save(audio_path)
    return audio_path

# --- STEP 2: STREAMLIT UI ---

def main():
    st.set_page_config(page_title="Magic Storyteller for Kids", page_icon="📖")
    st.title("📖 Magic Storyteller")
    st.write("Upload a picture, and I'll tell you a story! (For kids aged 3-10)")

    # File uploader for image input
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Your Uploaded Image", use_container_width=True)
        
        with st.spinner("Creating your story..."):
            try:
                # 1. Image Captioning
                caption = generate_caption(image)
                
                # 2. Story Generation
                story = generate_story(caption)
                
                # 3. Text-to-Speech
                audio_file_path = text_to_speech(story)

                # --- DISPLAY OUTPUTS ---
                st.subheader("The Story")
                st.write(story)
                
                st.subheader("Listen to the Story")
                st.audio(audio_file_path)
                
            except Exception as e:
                st.error(f"Oops! Something went wrong: {e}")

if __name__ == "__main__":
    main()
