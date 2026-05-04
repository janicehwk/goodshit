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
import os

# ── Function Part ───────────────────────────────────────────────────────────

def img2text(image_path):
    """
    Function 1: Generate a text caption from an uploaded image.
    Aligned with professor naming convention.[cite: 1]
    """
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt")

    output = model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption


def generate_story(caption):
    """
    Function 2: Generate a kid-friendly story (50-100 words) from a caption.
    Uses min_new_tokens to force the model to meet assignment length requirements.[cite: 1]
    """
    # Enhanced prompt to encourage descriptive storytelling for children[cite: 1]
    prompt = (
        f"Instruction: Write a magical, long adventure story for a child about {caption}. "
        f"The story should be wonderous and fun. Once upon a time, "
    )

    story_generator = pipeline(
        "text-generation",
        model="Prashant-karwasra/GPT2_text_generation_model"
    )

    # Parameters set to ensure word count stays between 50-100 words[cite: 1]
    result = story_generator(
        prompt,
        max_new_tokens=150,
        min_new_tokens=85,       # Forces the model to generate a longer narrative
        num_return_sequences=1,
        do_sample=True,
        temperature=0.85,
        repetition_penalty=1.2   # Prevents loops during longer generation
    )
    
    raw_story = result[0]["generated_text"]
    
    # Clean the story by removing the instruction prefix
    story = raw_story.replace(prompt, "Once upon a time, ")

    # Trim to exactly 100 words and end at a clean sentence[cite: 1]
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


def text2audio(story_text):
    """
    Function 3: Convert a story string into an audio file using gTTS.
    Aligned with professor naming convention.[cite: 1]
    """
    tts = gTTS(text=story_text, lang="en", slow=False)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_file.name)
    return temp_file.name


# ── Main Part ───────────────────────────────────────────────────────────────

def main():
    """
    Function 4: The Main Application logic and Streamlit UI.[cite: 1]
    """
    st.set_page_config(
        page_title="Magic Story Machine",
        page_icon="🪄",
        layout="centered"
    )

    # Custom CSS for Kid-Friendly UI
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

    st.markdown('<p class="kid-title">🪄 Magic Story Machine 🪄</p>', unsafe_allow_html=True)
    st.markdown('<p class="kid-subtitle">Upload a picture and watch it turn into a story! 📖✨</p>', unsafe_allow_html=True)

    st.markdown('<p class="step-label">📸 Step 1: Pick a Picture!</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a fun image...", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if uploaded_file is not None:
        # Save file locally as per professor's reference pattern[cite: 1]
        bytes_data = uploaded_file.getvalue()
        file_path = uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(bytes_data)

        st.image(uploaded_file, caption="🖼️ Your awesome picture!", use_container_width=True)

        # Stage 1: Image → Caption
        st.markdown('<p class="step-label">🔍 Step 2: What\'s in your picture?</p>', unsafe_allow_html=True)
        with st.spinner("🧐 Looking at your picture really carefully..."):
            caption = img2text(file_path)
        st.markdown(f'<div class="caption-box">I see: <strong>{caption}</strong></div>', unsafe_allow_html=True)

        # Stage 2: Caption → Story
        st.markdown('<p class="step-label">📝 Step 3: Story time!</p>', unsafe_allow_html=True)
        with st.spinner("✍️ Writing a magical story just for you..."):
            story = generate_story(caption)
        st.markdown(f'<div class="story-box">📖 {story}</div>', unsafe_allow_html=True)

        # Stage 3: Story → Audio
        st.markdown('<p class="step-label">🔊 Step 4: Listen to your story!</p>', unsafe_allow_html=True)
        with st.spinner("🎵 Getting the story ready to read aloud..."):
            audio_file_path = text2audio(story)

        # Audio player
        with open(audio_file_path, "rb") as audio_file:
            audio_bytes = audio_file.read()
        st.audio(audio_bytes, format="audio/mp3")

        # Success celebration[cite: 1]
        st.balloons()
        st.success("🎉 Your story is ready! Press play to listen! 🎧")

    st.markdown('<p class="fun-footer">Made with ❤️ for little storytellers everywhere 🌈</p>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
