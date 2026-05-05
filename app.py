import streamlit as st
from transformers import pipeline
from PIL import Image
from gtts import gTTS
import os

# --- PRE-LOAD MODELS (To keep functions fast) ---
@st.cache_resource
def load_resource_models():
    # Model for Function 1 (Fixed task name to avoid KeyError)
    cap_mod = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    # Model for Function 2
    story_mod = pipeline("text-generation", model="gpt2")
    return cap_mod, story_mod

caption_model, story_model = load_resource_models()

# ==========================================
# FUNCTION 1: Image Processing & Captioning
# ==========================================
def generate_caption(image_input):
    """
    Uses BLIP to turn pixels into a descriptive sentence.
    """
    results = caption_model(image_input)
    caption = results[0]['generated_text']
    return caption

# ==========================================
# FUNCTION 2: Story Generation
# ==========================================
def generate_story(caption):
    """
    Uses GPT-2 to expand the caption into a creative story.
    """
    prompt = f"Once upon a time, there was {caption}. It was a day that changed everything because"
    # Generate text
    story_output = story_model(prompt, max_length=150, do_sample=True, temperature=0.9, truncation=True)
    story_text = story_output[0]['generated_text']
    return story_text

# ==========================================
# FUNCTION 3: Text-to-Speech Conversion
# ==========================================
def text_to_speech(text):
    """
    Converts the story into an MP3 file using gTTS.
    """
    tts = gTTS(text=text, lang='en')
    file_path = "story_audio.mp3"
    tts.save(file_path)
    return file_path

# ==========================================
# MAIN APP: UI and Execution Flow
# ==========================================
def main():
    st.set_page_config(page_title="Deep Learning Storyteller", page_icon="🎨")
    st.title("Deep Learning Business App: Storyteller")
    st.markdown("### Assignment: Multimodal AI Pipeline")

    # --- NEW: Initialize Session States for Reset Functionality ---
    if 'processed' not in st.session_state:
        st.session_state.processed = False
    if 'uploader_key' not in st.session_state:
        st.session_state.uploader_key = 0

    # --- MODIFIED: Added dynamic key to file_uploader ---
    uploaded_file = st.file_uploader(
        "Upload an Image", 
        type=["jpg", "jpeg", "png"], 
        key=f"uploader_{st.session_state.uploader_key}"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("Generate Experience"):
            # Mark as processed so the reset button appears later
            st.session_state.processed = True
            
            with st.spinner("Processing through the AI pipeline..."):
                
                # Execution Flow:
                # 1. Image -> Caption
                caption = generate_caption(image)
                st.subheader("1. Generated Caption")
                st.success(caption)

                # 2. Caption -> Story
                story = generate_story(caption)
                st.subheader("2. Generated Story")
                st.write(story)

                # 3. Story -> Audio
                audio_file = text_to_speech(story)
                st.subheader("3. Audio Narration")
                st.audio(audio_file, format="audio/mp3")

    # --- NEW: Create Another Story Flow ---
    if st.session_state.processed:
        st.divider()
        if st.button("Create Another Story"):
            # Reset the processed state
            st.session_state.processed = False
            # Change the uploader key to force Streamlit to clear the file
            st.session_state.uploader_key += 1
            # Rerun the app from the top
            st.rerun()

if __name__ == "__main__":
    main()
