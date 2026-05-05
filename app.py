import streamlit as st
from transformers import pipeline
from PIL import Image
from gtts import gTTS
import os

# ==========================================
# MODEL INITIALIZATION (Cached for performance)
# ==========================================
@st.cache_resource
def load_resource_models():
    """
    Loads the Hugging Face models into memory. 
    Cached to prevent reloading on every button click.
    """
    # Using 'image-to-text' as it is the official task name in modern transformers
    cap_mod = pipeline(
        "image-to-text", 
        model="Salesforce/blip-image-captioning-base"
    )
    
    # Using GPT-2 for text generation
    story_mod = pipeline(
        "text-generation", 
        model="gpt2"
    )
    return cap_mod, story_mod

# Load models at startup
caption_model, story_model = load_resource_models()

# ==========================================
# FUNCTION 1: Image Processing & Captioning
# ==========================================
def generate_caption(image_input):
    """
    Takes a PIL Image and returns a descriptive string using BLIP.
    """
    results = caption_model(image_input)
    # Extract the generated text from the pipeline output
    return results[0]['generated_text']

# ==========================================
# FUNCTION 2: Story Generation
# ==========================================
def generate_story(caption):
    """
    Takes a caption and expands it into a kid-friendly story (approx. 50-100 words).
    """
    # Kid-friendly prompt structure
    prompt = f"Once upon a time, in a magical land, there was {caption}. Suddenly, "
    
    # Generate story with parameters tuned for creativity but keeping it concise
    story_output = story_model(
        prompt, 
        max_length=100,      # Keeps it within the 50-100 word requirement
        do_sample=True,      # Enables creative generation
        temperature=0.8,     # Adds a bit of randomness
        truncation=True      # Ensures it doesn't exceed max length gracefully
    )
    return story_output[0]['generated_text']

# ==========================================
# FUNCTION 3: Text-to-Speech Conversion
# ==========================================
def text_to_speech(text):
    """
    Converts text to an MP3 audio file using Google TTS (gTTS).
    """
    tts = gTTS(text=text, lang='en', slow=False)
    file_path = "magical_story_audio.mp3"
    tts.save(file_path)
    return file_path

# ==========================================
# MAIN APP: UI and Execution Flow
# ==========================================
def main():
    # Page setup
    st.set_page_config(page_title="Magical Storyteller", page_icon="🪄", layout="centered")
    
    st.title("🪄 Magical AI Storyteller")
    st.write("Upload a picture and let the AI write and read a magical story for you!")

    # State management for the reset button
    if 'story_generated' not in st.session_state:
        st.session_state.story_generated = False

    # Image Uploader
    uploaded_file = st.file_uploader("Upload an Image (JPG/PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Your magical picture!", use_container_width=True)

        # Trigger button
        if st.button("✨ Create My Story ✨"):
            st.session_state.story_generated = True
            
            with st.spinner("The AI is thinking of a magical tale..."):
                
                # --- Step 1: Caption ---
                caption = generate_caption(image)
                st.info(f"**What the AI sees:** {caption}")

                # --- Step 2: Story ---
                story = generate_story(caption)
                st.subheader("📖 Your Story:")
                st.write(story)

                # --- Step 3: Audio ---
                audio_path = text_to_speech(story)
                st.subheader("🎧 Listen to the Story:")
                st.audio(audio_path, format="audio/mp3")

        # Reset flow (appears only after a story is made)
        if st.session_state.story_generated:
            st.divider()
            if st.button("🔄 Create Another Story"):
                st.session_state.story_generated = False
                st.rerun()

if __name__ == "__main__":
    main()
