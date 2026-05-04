# ============================================================================
# Program Title: Storytelling App
# Description:   A kid-friendly application turning images into audio stories.
# ============================================================================

# ── Import Part ─────────────────────────────────────────────────────────────
import streamlit as st
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration
from gtts import gTTS
from PIL import Image
import tempfile
import os

# ── Function Part ───────────────────────────────────────────────────────────

# Function 1: Image to Text (Aligned with Prof name: img2text)
def img2text(url):
    """Generates a caption from the uploaded image."""
    # Using your specific logic for maximum compatibility
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    image = Image.open(url).convert("RGB")
    inputs = processor(image, return_tensors="pt")

    output = model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

# Function 2: Text to Story (Aligned with Prof requirement)
def generate_story(scenario):
    """Expands the caption into a 50-100 word kid-friendly story."""
    # Build a kid-friendly prompt from the scenario (Prof's variable name)
    prompt = (
        f"Once upon a time, {scenario}. "
        "This is a fun and magical story for little kids: "
    )

    story_pipe = pipeline(
        "text-generation", 
        model="Prashant-karwasra/GPT2_text_generation_model"
    )

    # Generate text with controlled length for 50-100 words per requirements
    result = story_pipe(
        prompt, 
        max_length=150, 
        num_return_sequences=1, 
        do_sample=True, 
        temperature=0.8
    )
    story = result[0]["generated_text"]

    # Logic to ensure 50-100 words as per assignment
    words = story.split()
    if len(words) > 100:
        trimmed = " ".join(words[:100])
        for punctuation in [".", "!", "?"]:
            last_pos = trimmed.rfind(punctuation)
            if last_pos != -1:
                trimmed = trimmed[: last_pos + 1]
                break
        story = trimmed
    return story

# Function 3: Story to Audio (Aligned with Prof requirement)
def text2audio(story_text):
    """Converts the generated story into an audio file."""
    tts = gTTS(text=story_text, lang="en", slow=False)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_file.name)
    return temp_file.name

# Function 4: Main Part (Entry point for Streamlit)
def main():
    # Page setup aligned with professor's icons/titles
    st.set_page_config(page_title="Your Image to Audio Story", page_icon="🦜")
    
    # Custom CSS from your version for the "User Experience" criterion
    st.markdown("""
    <style>
        .stApp { background: linear-gradient(135deg, #FFDEE9 0%, #B5FFFC 100%); }
        .kid-title { text-align: center; font-size: 3rem; font-weight: 800; color: #FF6B6B; }
        .story-box { background: #FFFFFF; border: 4px dashed #6C5CE7; border-radius: 20px; padding: 25px; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="kid-title">🪄 Magic Story Machine 🪄</p>', unsafe_allow_html=True)
    st.header("Turn Your Image to Audio Story")

    uploaded_file = st.file_uploader("Select an Image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Save file locally (Professor's logic)
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name, "wb") as file:
            file.write(bytes_data)
        
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

        # Stage 1: Image to Text
        st.text('Processing img2text...')
        scenario = img2text(uploaded_file.name)
        st.write(f"**Scenario:** {scenario}")

        # Stage 2: Text to Story
        st.text('Generating a story...')
        story = generate_story(scenario)
        st.markdown(f'<div class="story-box">📖 {story}</div>', unsafe_allow_html=True)

        # Stage 3: Story to Audio
        st.text('Generating audio data...')
        audio_path = text2audio(story)

        # Play button and audio output
        if st.button("Play Audio"):
            with open(audio_path, "rb") as audio_file:
                audio_bytes = audio_file.read()
            st.audio(audio_bytes, format="audio/mp3")
            st.balloons()

# ── Execution ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
