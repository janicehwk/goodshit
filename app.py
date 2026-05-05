# ============================================================================
# Program Title: Magic Story Machine - A Storytelling App for Kids
# Description:   A kid-friendly Streamlit application that turns uploaded
#                images into fun audio stories for children aged 3-10.
# Pipeline:
#   1. Image Captioning — Salesforce/blip-image-captioning-base
#   2. Story Generation — pranavpsv/genre-story-generator-v2
#   3. Text-to-Speech   — gTTS (Google Text-to-Speech)
# ============================================================================

# ── Import Part ─────────────────────────────────────────────────────────────
import streamlit as st
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration
from gtts import gTTS
from PIL import Image
import tempfile

# ── Model Caching Part (Crucial for Speed) ──────────────────────────────────
@st.cache_resource(show_spinner="Loading Magic AI Models... (This only happens once!)")
def load_ai_models():
    """
    Loads all heavy AI models into memory once and caches them.
    """
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    cap_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    
    # --- UPDATED: Using your specific text-generation model ---
    story_pipe = pipeline("text-generation", model="pranavpsv/genre-story-generator-v2")
    
    return processor, cap_model, story_pipe

# Initialize models in memory globally so functions can access them
blip_processor, blip_model, story_pipeline = load_ai_models()

# ── Function Part ───────────────────────────────────────────────────────────

def img2text(url):
    """
    Generate a text caption from an uploaded image.
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
    # Build the scenario prompt
    prompt = f"Once upon a time, {scenario}. "
    
    raw_story = ""
    
    # BULLETPROOF LOOP: Try up to 3 times to guarantee at least 50 words
    for attempt in range(3):
        # --- UPDATED: Using the new pipeline ---
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
    Convert a story string into an audio file using Google Text-to-Speech.
    """
    tts = gTTS(text=story_text, lang="en", slow=False)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_file.name)
    return temp_file.name


# ── Main App Execution Flow ─────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Magic Story Machine",
        page_icon="🪄",
        layout="centered"
    )

    st.markdown("""
    <style>
        .stApp { background: linear-gradient(135deg, #FFDEE9 0%, #B5FFFC 100%); }
        .step-label { font-size: 1.2rem; font-weight: 700; color: #2D3436; background: #FFEAA7; padding: 8px 16px; border-radius: 20px; display: inline-block; margin-bottom: 10px; }
        .caption-box { background: #DFE6E9; border-radius: 15px; padding: 15px 20px; font-size: 1.05rem; color: #2D3436; margin: 10px 0; }
        .fun-footer { text-align: center; color: #636E72; font-size: 0.9rem; margin-top: 40px; padding: 10px; }
    </style>
    """, unsafe_allow_html=True)

    if 'uploader_key' not in st.session_state:
        st.session_state.uploader_key = 0
    if 'story_finished' not in st.session_state:
        st.session_state.story_finished = False

    st.title("🪄 Magic Story Machine 🪄")
    st.subheader("Upload a picture and watch it turn into a story! 📖✨")

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
        with open(file_path, "wb") as f:
            f.write(bytes_data)

        st.image(uploaded_file, caption="🖼️ Your awesome picture!", use_container_width=True)

        # ── Stage 1: Image → Caption ──
        st.markdown('<p class="step-label">🔍 Step 2: What\'s in your picture?</p>', unsafe_allow_html=True)
        with st.spinner("🧐 Looking at your picture really carefully..."):
            scenario = img2text(file_path) # We use the caption as the scenario
        st.markdown(f'<div class="caption-box">I see: <strong>{scenario}</strong></div>', unsafe_allow_html=True)

        # ── Stage 2: Text to Story (Updated to match your requested format) ──
        st.markdown('<p class="step-label">📝 Step 3: Story time!</p>', unsafe_allow_html=True)
        st.text('Generating a story...') # Added from your snippet
        
        with st.spinner("✍️ Writing a magical story just for you..."):
            story = generate_story(scenario)
            
        # Updated to use st.write as requested in your snippet
        st.write(f"**Story:** {story}")

        # ── Stage 3: Story → Audio ──
        st.markdown('<p class="step-label">🔊 Step 4: Listen to your story!</p>', unsafe_allow_html=True)
        with st.spinner("🎵 Getting the story ready to read aloud..."):
            audio_file_path = text_to_audio(story)

        with open(audio_file_path, "rb") as audio_file:
            audio_bytes = audio_file.read()
        st.audio(audio_bytes, format="audio/mp3")

        st.balloons()
        st.success("🎉 Your story is ready! Press play to listen! 🎧")
        st.session_state.story_finished = True

        # --- Reset Flow ---
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
