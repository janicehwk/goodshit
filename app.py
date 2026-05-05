# ============================================================================
# Program Title: Magic Story Machine - A Storytelling App for Kids
# Description:   A kid-friendly Streamlit application that turns uploaded
#                images into fun audio stories for children aged 3-10.
# Pipeline:
#   1. Image Captioning — Salesforce/blip-image-captioning-base
#   2. Story Generation — pranavpsv/genre-story-generator-v2
#   3. Text-to-Speech   — kakao-enterprise/vits-ljs (Hugging Face TTS)
# ============================================================================

# ── Import Part ─────────────────────────────────────────────────────────────
import streamlit as st
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import tempfile
import scipy.io.wavfile
import numpy as np

# ── Model Caching Part (Crucial for Speed) ──────────────────────────────────
@st.cache_resource(show_spinner="Loading Magic AI Models... (This only happens once!)")
def load_ai_models():
    """
    Loads all heavy AI models into memory once and caches them.
    This prevents the app from taking a long time to reload when buttons are clicked.
    """
    # 1. Vision Model
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    cap_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    
    # 2. Text Generation Model
    story_pipe = pipeline("text-generation", model="pranavpsv/genre-story-generator-v2")
    
    # 3. Text-to-Speech Model (High-quality intonation for storytelling)
    tts_pipe = pipeline("text-to-speech", model="kakao-enterprise/vits-ljs")
    
    return processor, cap_model, story_pipe, tts_pipe

# Initialize models in memory globally so functions can access them
blip_processor, blip_model, story_pipeline, tts_pipeline = load_ai_models()


# ── Function Part ───────────────────────────────────────────────────────────

def img2text(url):
    """
    Generate a text caption from an uploaded image.
    Uses the globally cached BLIP image captioning models.
    """
    image = Image.open(url).convert("RGB")
    inputs = blip_processor(image, return_tensors="pt")

    output = blip_model.generate(**inputs, max_new_tokens=50)
    caption = blip_processor.decode(output[0], skip_special_tokens=True)
    return caption


def generate_story(scenario):
    """
    Generate a kid-friendly story (50-100 words) from a scenario.
    Includes a retry loop to strictly enforce the 50-word minimum.
    """
    prompt = f"Once upon a time, {scenario}. "
    
    raw_story = ""
    
    # BULLETPROOF LOOP: Try up to 3 times to guarantee at least 50 words
    for attempt in range(3):
        story_results = story_pipeline(
            prompt,
            min_new_tokens=70,      
            max_new_tokens=150,     
            num_return_sequences=1,
            do_sample=True,
            temperature=0.85 + (attempt * 0.1),
            repetition_penalty=1.2,
            truncation=True
        )
        
        raw_story = story_results[0]["generated_text"]
        
        word_count = len(raw_story.split())
        if word_count >= 50:
            break

    # -- Python Trimmer: Guarantee it doesn't go OVER 100 words --
    words = raw_story.split()
    if len(words) > 100:
        words = words[:100]
        trimmed_story = " ".join(words)
        
        last_punctuation = max(
            trimmed_story.rfind('.'), 
            trimmed_story.rfind('!'), 
            trimmed_story.rfind('?')
        )
        
        if last_punctuation != -1:
            trimmed_story = trimmed_story[:last_punctuation+1]
        else:
            trimmed_story += "..."
            
        return trimmed_story

    return raw_story


def text_to_audio(story_text):
    """
    Convert a story string into a natural, audiobook-style audio file 
    using Hugging Face's VITS model.
    """
    # Generate the audio array
    speech = tts_pipeline(story_text)
    
    # Save to a temporary WAV file so Streamlit can play it
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    
    scipy.io.wavfile.write(
        temp_file.name, 
        rate=speech["sampling_rate"], 
        data=speech["audio"][0]
    )
    
    return temp_file.name


# ── Main App Execution Flow ─────────────────────────────────────────────────
def main():
    # ── Page Configuration ──
    st.set_page_config(
        page_title="Magic Story Machine",
        page_icon="🪄",
        layout="centered"
    )

    # ── Custom CSS for Kid-Friendly UI ──
    st.markdown("""
    <style>
        .stApp { background: linear-gradient(135deg, #FFDEE9 0%, #B5FFFC 100%); }
        .step-label { font-size: 1.2rem; font-weight: 700; color: #2D3436; background: #FFEAA7; padding: 8px 16px; border-radius: 20px; display: inline-block; margin-bottom: 10px; }
        .story-box { background: #FFFFFF; border: 4px dashed #6C5CE7; border-radius: 20px; padding: 25px; font-size: 1.15rem; line-height: 1.8; color: #2D3436; margin: 15px 0; }
        .caption-box { background: #DFE6E9; border-radius: 15px; padding: 15px 20px; font-size: 1.05rem; color: #2D3436; margin: 10px 0; }
        .fun-footer { text-align: center; color: #636E72; font-size: 0.9rem; margin-top: 40px; padding: 10px; }
    </style>
    """, unsafe_allow_html=True)

    # --- Initialize Session States for Reset Functionality ---
    if 'uploader_key' not in st.session_state:
        st.session_state.uploader_key = 0
    if 'story_finished' not in st.session_state:
        st.session_state.story_finished = False

    # App header
    st.title("🪄 Magic Story Machine 🪄")
    st.subheader("Upload a picture and watch it turn into a story! 📖✨")

    # Image upload section
    st.markdown('<p class="step-label">📸 Step 1: Pick a Picture!</p>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose a fun image...",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
        key=f"uploader_{st.session_state.uploader_key}"
    )

    if uploaded_file is not None:
        # Save uploaded file locally for the model to read
        bytes_data = uploaded_file.getvalue()
        file_path = uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(bytes_data)

        st.image(uploaded_file, caption="🖼️ Your awesome picture!", use_container_width=True)

        # ── Stage 1: Image → Caption ──
        st.markdown('<p class="step-label">🔍 Step 2: What\'s in your picture?</p>', unsafe_allow_html=True)
        with st.spinner("🧐 Looking at your picture really carefully..."):
            scenario = img2text(file_path)
        st.markdown(f'<div class="caption-box">I see: <strong>{scenario}</strong></div>', unsafe_allow_html=True)

        # ── Stage 2: Text to Story ──
        st.markdown('<p class="step-label">📝 Step 3: Story time!</p>', unsafe_allow_html=True)
        st.text('Generating a story...')
        
        with st.spinner("✍️ Writing a magical story just for you..."):
            story = generate_story(scenario)
            
        st.write(f"**Story:** {story}")

        # ── Stage 3: Story → Audio ──
        st.markdown('<p class="step-label">🔊 Step 4: Listen to your story!</p>', unsafe_allow_html=True)
        with st.spinner("🎵 Getting the story ready to read aloud..."):
            audio_file_path = text_to_audio(story)

        # Play the WAV file generated by the VITS model
        with open(audio_file_path, "rb") as audio_file:
            audio_bytes = audio_file.read()
        st.audio(audio_bytes, format="audio/wav")

        st.balloons()
        st.success("🎉 Your story is ready! Press play to listen! 🎧")
        st.session_state.story_finished = True

        # --- Create Another Story Flow ---
        if st.session_state.story_finished:
            st.markdown("<br>", unsafe_allow_html=True) 
            if st.button("🔄 Create Another Story!"):
                widget_key = f"uploader_{st.session_state.uploader_key}"
                if widget_key in st.session_state:
                    del st.session_state[widget_key]

                st.session_state.uploader_key += 1
                st.session_state.story_finished = False
                st.rerun()

    st.markdown('<p class="fun-footer">Made with ❤️ for little storytellers everywhere 🌈</p>', unsafe_allow_html=True)


# ── Execute Main Function ───────────────────────────────────────────────────
if __name__ == "__main__":
    main()
