# ============================================================================
# Program Title: Magic Story Machine - A Storytelling App for Kids
# Description:   A kid-friendly Streamlit application that turns uploaded
#                images into fun audio stories for children aged 3-10.
# Models Used:
#   1. Image Captioning — Salesforce/blip-image-captioning-base
#   2. Story Generation — Prashant-karwasra/GPT2_text_generation_model
#   3. Text-to-Speech   — gTTS (Google Text-to-Speech)
# ============================================================================

# ── Import Part ─────────────────────────────────────────────────────────────
import streamlit as st
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import tempfile
from gtts import gTTS

# ── Model Caching Part (Crucial for Streamlit Cloud Stability) ──────────────
@st.cache_resource(show_spinner="Loading Magic AI Models into Memory...")
def load_models():
    """
    Loads all heavy AI models into memory once and caches them.
    This prevents Out-of-Memory (OOM) crashes on Streamlit Cloud.
    """
    # 1. Vision Model
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    
    # 2. Story Generation Model
    story_pipe = pipeline("text-generation", model="Prashant-karwasra/GPT2_text_generation_model")
    
    return blip_processor, blip_model, story_pipe

# Initialize models globally
processor, img_model, story_generator = load_models()

# ── Function Part ───────────────────────────────────────────────────────────

def img2text(image_path):
    """
    Convert an uploaded image into a text caption using BLIP.
    """
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt")
    
    output = img_model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption


def text2story(caption):
    """
    Generate a kid-friendly story (50-100 words) from an image caption.
    """
    prompt = (
        f"Write a short, happy story for a 5-year-old child about {caption}. "
        f"Use simple words and a happy ending. "
        f"Once upon a time, {caption}. "
    )
    
    result = story_generator(
        prompt,
        max_length=150,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.7
    )
    story = result[0]["generated_text"]

    # Keep only the story part starting from "Once upon a time"
    story_start = story.find("Once upon a time")
    if story_start != -1:
        story = story[story_start:]

    # Trim the story to strictly stay within 50-100 words
    words = story.split()
    if len(words) > 100:
        trimmed = " ".join(words[:100])
        # End cleanly at the last complete sentence
        for punct in [".", "!", "?"]:
            last_pos = trimmed.rfind(punct)
            if last_pos != -1:
                trimmed = trimmed[: last_pos + 1]
                break
        story = trimmed
        
    return story


def text2audio(story_text):
    """
    Convert a story into an audio file using Google TTS (gTTS).
    Ultra-lightweight, ensuring Streamlit Cloud doesn't crash.
    """
    tts = gTTS(text=story_text, lang='en', slow=False)
    
    # Save to a temporary MP3 file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_file.name)
    
    return temp_file.name


# ── Main Application Flow ───────────────────────────────────────────────────
def main():
    # ── Page Setup ──
    st.set_page_config(
        page_title="Magic Story Machine",
        page_icon="🪄",
        layout="centered"
    )

    # ── Kid-Friendly CSS Styling ──
    st.markdown("""
    <style>
        .stApp { background: linear-gradient(135deg, #FFDEE9 0%, #B5FFFC 100%); }
        .kid-title { text-align: center; font-size: 3rem; font-weight: 800; color: #FF6B6B; text-shadow: 3px 3px 0px #FFE66D; margin-bottom: 0; }
        .kid-subtitle { text-align: center; font-size: 1.3rem; color: #6C5CE7; margin-top: 0; margin-bottom: 30px; }
        .step-label { font-size: 1.2rem; font-weight: 700; color: #2D3436; background: #FFEAA7; padding: 8px 16px; border-radius: 20px; display: inline-block; margin-bottom: 10px; }
        .story-box { background: #FFFFFF; border: 4px dashed #6C5CE7; border-radius: 20px; padding: 25px; font-size: 1.15rem; line-height: 1.8; color: #2D3436; margin: 15px 0; }
        .caption-box { background: #DFE6E9; border-radius: 15px; padding: 15px 20px; font-size: 1.05rem; color: #2D3436; margin: 10px 0; }
        .fun-footer { text-align: center; color: #636E72; font-size: 0.9rem; margin-top: 40px; padding: 10px; }
    </style>
    """, unsafe_allow_html=True)

    # --- Initialize Session States ---
    if 'uploader_key' not in st.session_state:
        st.session_state.uploader_key = 0
    if 'story_finished' not in st.session_state:
        st.session_state.story_finished = False

    # ── App Header ──
    st.markdown('<p class="kid-title">🪄 Magic Story Machine 🪄</p>', unsafe_allow_html=True)
    st.markdown('<p class="kid-subtitle">Upload a picture and watch it turn into a story! 📖✨</p>', unsafe_allow_html=True)

    # ── Step 1: Image Upload ──
    st.markdown('<p class="step-label">📸 Step 1: Pick a Picture!</p>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a fun image...",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
        key=f"uploader_{st.session_state.uploader_key}"
    )

    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        file_path = uploaded_file.name
        with open(file_path, "wb") as file:
            file.write(bytes_data)

        st.image(uploaded_file, caption="🖼️ Your awesome picture!", use_container_width=True)

        # ── Step 2: Image → Caption ──
        st.markdown('<p class="step-label">🔍 Step 2: What\'s in your picture?</p>', unsafe_allow_html=True)
        with st.spinner("🧐 Looking at your picture really carefully..."):
            scenario = img2text(file_path)
        st.markdown(f'<div class="caption-box">I see: <strong>{scenario}</strong></div>', unsafe_allow_html=True)

        # ── Step 3: Caption → Story ──
        st.markdown('<p class="step-label">📝 Step 3: Story time!</p>', unsafe_allow_html=True)
        with st.spinner("✍️ Writing a magical story just for you..."):
            story = text2story(scenario)
        st.markdown(f'<div class="story-box">📖 {story}</div>', unsafe_allow_html=True)

        # ── Step 4: Story → Audio ──
        st.markdown('<p class="step-label">🔊 Step 4: Listen to your story!</p>', unsafe_allow_html=True)
        with st.spinner("🎵 Getting the story ready to read aloud..."):
            audio_file_path = text2audio(story)

        # Play the generated audio (MP3 format for gTTS)
        with open(audio_file_path, "rb") as audio_file:
            audio_bytes = audio_file.read()
        st.audio(audio_bytes, format="audio/mp3")

        st.balloons()
        st.success("🎉 Your story is ready! Press play to listen! 🎧")
        st.session_state.story_finished = True

        # ── Reset App Flow ──
        if st.session_state.story_finished:
            st.markdown("<br>", unsafe_allow_html=True) 
            if st.button("🔄 Create Another Story!"):
                # Clear the uploader cache
                widget_key = f"uploader_{st.session_state.uploader_key}"
                if widget_key in st.session_state:
                    del st.session_state[widget_key]

                st.session_state.uploader_key += 1
                st.session_state.story_finished = False
                st.rerun()

    # Footer
    st.markdown('<p class="fun-footer">Made with ❤️ for little storytellers everywhere 🌈</p>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
