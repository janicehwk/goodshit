import streamlit as st
from transformers import pipeline
from PIL import Image
from gtts import gTTS
import os

# --- Model Loading (Backend Setup) ---
@st.cache_resource
def load_resource_models():
    # Model for Function 1: Image Captioning
    cap_mod = pipeline("image-captioning", model="Salesforce/blip-image-captioning-base")
    # Model for Function 2: Story Generation (using GPT-2)
    story_mod = pipeline("text-generation", model="gpt2")
    return cap_mod, story_mod

caption_model, story_model = load_resource_models()

# ==========================================
# FUNCTION 1: Image Processing & Captioning
# ==========================================
def generate_caption(image_input):
    """Uses BLIP to generate a descriptive caption."""
    results = caption_model(image_input)
    return results[0]['generated_text']

# ==========================================
# FUNCTION 2: Story Generation
# ==========================================
def generate_story(caption):
    """Uses GPT-2 to expand the caption into a narrative."""
    prompt = f"Once upon a time, there was {caption}. It was a strange day because"
    story_output = story_model(prompt, max_length=150, do_sample=True, temperature=0.8, truncation=True)
    return story_output[0]['generated_text']

# ==========================================
# FUNCTION 3: Text-to-Speech Conversion
# ==========================================
def text_to_speech(text):
    """Converts the story text into an MP3 file using gTTS."""
    tts = gTTS(text=text, lang='en')
    file_path = "story_audio.mp3"
    tts.save(file_path)
    return file_path

# ==========================================
# MAIN APP: UI and Execution Flow
# ==========================================
def main():
    st.set_page_config(page_title="Multimodal AI Storyteller", page_icon="📖")
    st.title("Deep Learning Business App")
    st.write("Complete the pipeline: Image → Caption → Story → Speech")

    # Initialize a 'processed' state to keep track of our flow
    if 'processed' not in st.session_state:
        st.session_state.processed = False

    # UI: Image Upload
    uploaded_file = st.file_uploader("Step 1: Upload an Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Your Image", use_container_width=True)

        # BUTTON: Run the Pipeline
        if st.button("Generate Story & Audio"):
            st.session_state.processed = True
            
            with st.spinner("Executing AI Models..."):
                # 1. Image -> Caption
                caption = generate_caption(image)
                st.subheader("Output 1: Image Caption")
                st.info(caption)

                # 2. Caption -> Story
                story = generate_story(caption)
                st.subheader("Output 2: Generated Story")
                st.write(story)

                # 3. Story -> Audio
                audio_path = text_to_speech(story)
                st.subheader("Output 3: Audio Narration")
                st.audio(audio_path, format="audio/mp3")

        # --- LAST FLOW: RESET BUTTON ---
        if st.session_state.processed:
            st.divider()
            if st.button("🔄 Reset and Create More Stories"):
                # This clears the uploader and the state, effectively restarting the app
                st.session_state.processed = False
                st.rerun() 

if __name__ == "__main__":
    main()
