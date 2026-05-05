# ============================================================================
# Program Title: Magic Story Machine - A Storytelling App for Kids
# Description:   A kid-friendly Streamlit application that turns uploaded
#                images into fun audio stories for children aged 3-10.
# Pipeline:
#   1. Image Captioning — Salesforce/blip-image-captioning-base
#   2. Story Generation — Prashant-karwasra/GPT2_text_generation_model
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
    This prevents the app from taking a long time to reload when buttons are clicked.
    """
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    cap_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    story_gen = pipeline("text-generation", model="Prashant-karwasra/GPT2_text_generation_model")
    
    return processor, cap_model, story_gen

# Initialize models in memory globally
blip_processor, blip_model, story_pipeline = load_ai_models()

# ── Function Part ───────────────────────────────────────────────────────────

def img2text(url):
    """
    Generate a text caption from an uploaded image.
    Uses the globally cached BLIP image captioning models.
    
    Parameters:
        url (str): File path or URL of the uploaded image.
    Returns:
        str: A short caption describing the image content.
    """
    # Open and process the image
    image = Image.open(url).convert("RGB")
    inputs = blip_processor(image, return_tensors="pt")

    # Generate the caption
    output = blip_model.generate(**inputs, max_new_tokens=50)
    caption = blip_processor.decode(output[0], skip_special_tokens=True)
    return caption


def generate_story(caption):
    """
    Generate a kid-friendly story (50-100 words) from an image caption.
    Uses the globally cached GPT2 text generation pipeline.
    """
    # Build a kid-friendly prompt from the caption that encourages a story
    prompt = (
        f"Once upon a time, {caption}. "
        "This was the beginning of an incredible and unexpected adventure. "
        "Suddenly, "
    )

    # Generate text enforcing a longer length, preventing repetitive sentences
    result = story_pipeline(
        prompt,
        min_length=80,          # Forces the model to generate at least ~60 words
        max_length=160,         # Gives room to finish thoughts
        num_return_sequences=1,
        do_sample=True,
        temperature=0.85,
        repetition_penalty=1.2, # Stops GPT2 from repeating itself
        truncation=True
    )
    
    raw_story = result[0]["generated_text"]

    # Trim story to exactly 50-100 words cleanly
    words = raw_story.split()
    if len(words) > 100:
        words = words[:100]
        trimmed_story = " ".join(words)
        
        # Try to end at a natural sentence boundary (. ! ?)
        last_punctuation = max(
            trimmed_story.rfind('.'), 
            trimmed_story.rfind('!'), 
            trimmed_story.rfind('?')
        )
        
        if last_punctuation != -1:
            trimmed_story = trimmed_story[:last_punctuation+1]
        else:
            trimmed_story += "..." # Fallback
            
        return trimmed_story

    return raw_story


def text_to_audio(story_text):
    """
    Convert a story string into an audio file using Google Text-to-Speech.
    """
    tts = gTTS(text=story_text, lang="en", slow=False)
    # Save to a temporary file so Streamlit can play it
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_file.name)
    return temp_file.name


# ── Page Configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Magic Story Machine",
    page_icon="🪄",
    layout="centered"
)

# ── Custom CSS for Kid-Friendly UI ──────────────────────────────────────────
st.markdown("""
<style>
    /* Fun background gradient */
    .stApp {
        background: linear-gradient(135deg, #FFDEE9 0%, #B5FFFC 100%);
    }

    /* Subtitle styling */
    .kid-subtitle {
        text-align: center;
        font-size: 1.3rem;
        color: #6C5CE7;
        margin-top: 0;
        margin-bottom: 30px;
    }

    /* Step labels */
    .step-label {
        font-size: 1.2rem;
        font-weight: 700;
        color: #2D3436;
        background: #FFEAA7;
        padding: 8px 16px;
        border-radius: 20px;
        display: inline-block;
        margin-bottom: 10px;
    }

    /* Story box */
    .story-box {
        background: #FFFFFF;
        border: 4px dashed #6C5CE7;
        border-radius: 20px;
        padding: 25px;
        font-size: 1.15rem;
        line-height: 1.8;
        color: #2D3436;
        margin: 15px 0;
    }

    /* Caption box */
    .caption-box {
        background: #DFE6E9;
        border-radius: 15px;
        padding: 15px 20px;
        font-size: 1.05rem;
        color: #2D3436;
        margin: 10px 0;
    }

    /* Fun footer */
    .fun-footer {
        text-align: center;
        color: #636E72;
        font-size: 0.9rem;
        margin-top: 40px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ── Main Part ───────────────────────────────────────────────────────────────

# --- Initialize Session States for Reset Functionality ---
if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0
if 'story_finished' not in st.session_state:
    st.session_state.story_finished = False

# App header using standard st.title
st.title("🪄 Magic Story Machine 🪄")
st.subtitle("Upload a picture and watch it turn into a story! 📖✨")

# Image upload section
st.markdown('<p class="step-label">📸 Step 1: Pick a Picture!</p>', unsafe_allow_html=True)

# Added dynamic key to file_uploader to allow clearing
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

    # Display the uploaded image
    st.image(uploaded_file, caption="🖼️ Your awesome picture!", use_container_width=True)

    # ── Stage 1: Image → Caption ────────────────────────────────────────
    st.markdown(
        '<p class="step-label">🔍 Step 2: What\'s in your picture?</p>',
        unsafe_allow_html=True,
    )
    with st.spinner("🧐 Looking at your picture really carefully..."):
        # Updated function call to img2text
        caption = img2text(file_path)
    st.markdown(
        f'<div class="caption-box">I see: <strong>{caption}</strong></div>',
        unsafe_allow_html=True,
    )

    # ── Stage 2: Caption → Story ────────────────────────────────────────
    st.markdown(
        '<p class="step-label">📝 Step 3: Story time!</p>',
        unsafe_allow_html=True,
    )
    with st.spinner("✍️ Writing a magical story just for you..."):
        story = generate_story(caption)
    st.markdown(
        f'<div class="story-box">📖 {story}</div>',
        unsafe_allow_html=True,
    )

    # ── Stage 3: Story → Audio ──────────────────────────────────────────
    st.markdown(
        '<p class="step-label">🔊 Step 4: Listen to your story!</p>',
        unsafe_allow_html=True,
    )
    with st.spinner("🎵 Getting the story ready to read aloud..."):
        audio_file_path = text_to_audio(story)

    # Audio player
    with open(audio_file_path, "rb") as audio_file:
        audio_bytes = audio_file.read()
    st.audio(audio_bytes, format="audio/mp3")

    # Celebration message
    st.balloons()
    st.success("🎉 Your story is ready! Press play to listen! 🎧")
    
    # Flag that the story is done
    st.session_state.story_finished = True

    # --- Create Another Story Flow ---
    if st.session_state.story_finished:
        st.markdown("<br>", unsafe_allow_html=True) # Adds a little spacing
        if st.button("🔄 Create Another Story!"):
            # Delete the file from Streamlit's internal cache immediately
            widget_key = f"uploader_{st.session_state.uploader_key}"
            if widget_key in st.session_state:
                del st.session_state[widget_key]

            # Change the uploader key to force Streamlit to clear the file UI
            st.session_state.uploader_key += 1
            # Reset the finished flag
            st.session_state.story_finished = False
            # Rerun the app instantly
            st.rerun()

# Footer
st.markdown(
    '<p class="fun-footer">Made with ❤️ for little storytellers everywhere 🌈</p>',
    unsafe_allow_html=True,
)
