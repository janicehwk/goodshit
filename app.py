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

# ── Function Part ───────────────────────────────────────────────────────────

def generate_caption(image_path):
    """
    Generate a text caption from an uploaded image.
    Uses the BLIP image captioning model from Hugging Face.
    Loads the model directly via BlipProcessor and BlipForConditionalGeneration
    for maximum compatibility.
    
    Parameters:
        image_path (str): File path of the uploaded image.
    Returns:
        str: A short caption describing the image content.
    """
    # Load the processor and model directly
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    # Open and process the image
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt")

    # Generate the caption
    output = model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption


def generate_story(caption):
    """
    Generate a kid-friendly story (50-100 words) from an image caption.
    Uses a GPT2-based text generation model from Hugging Face.
    
    Parameters:
        caption (str): A short image caption to base the story on.
    Returns:
        str: A short, fun story suitable for children aged 3-10.
    """
    # Build a kid-friendly prompt from the caption
    prompt = (
        f"Once upon a time, {caption}. "
        "This is a fun and magical story for little kids: "
    )

    story_generator = pipeline(
        "text-generation",
        model="Prashant-karwasra/GPT2_text_generation_model"
    )

    # Generate text with controlled length for 50-100 words
    result = story_generator(
        prompt,
        max_length=120,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.8
    )
    story = result[0]["generated_text"]

    # Trim story to roughly 50-100 words
    words = story.split()
    if len(words) > 100:
        # Cut at the last full sentence within 100 words
        trimmed = " ".join(words[:100])
        # Try to end at a sentence boundary
        for punctuation in [".", "!", "?"]:
            last_pos = trimmed.rfind(punctuation)
            if last_pos != -1:
                trimmed = trimmed[: last_pos + 1]
                break
        story = trimmed

    return story


def text_to_audio(story_text):
    """
    Convert a story string into an audio file using Google Text-to-Speech.
    
    Parameters:
        story_text (str): The story text to convert to speech.
    Returns:
        str: File path of the generated audio (.mp3) file.
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

    /* Title styling */
    .kid-title {
        text-align: center;
        font-size: 3rem;
        font-weight: 800;
        color: #FF6B6B;
        text-shadow: 3px 3px 0px #FFE66D;
        margin-bottom: 0;
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

# --- NEW: Initialize Session States for Reset Functionality ---
if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0
if 'story_finished' not in st.session_state:
    st.session_state.story_finished = False

# App header with playful styling
st.markdown('<p class="kid-title">🪄 Magic Story Machine 🪄</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="kid-subtitle">Upload a picture and watch it turn into a story! 📖✨</p>',
    unsafe_allow_html=True,
)

# Image upload section
st.markdown('<p class="step-label">📸 Step 1: Pick a Picture!</p>', unsafe_allow_html=True)

# --- MODIFIED: Added dynamic key to file_uploader to allow clearing ---
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
        caption = generate_caption(file_path)
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
    
    # --- NEW: Flag that the story is done ---
    st.session_state.story_finished = True

    # --- NEW: Create Another Story Flow ---
    if st.session_state.story_finished:
        st.markdown("<br>", unsafe_allow_html=True) # Adds a little spacing
        if st.button("🔄 Create Another Story!"):
            # Change the uploader key to force Streamlit to clear the file
            st.session_state.uploader_key += 1
            # Reset the finished flag
            st.session_state.story_finished = False
            # Rerun the app from the top
            st.rerun()

# Footer
st.markdown(
    '<p class="fun-footer">Made with ❤️ for little storytellers everywhere 🌈</p>',
    unsafe_allow_html=True,
)
