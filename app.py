# ============================================================================
# Program Title: Magic Story Machine - A Storytelling App for Kids
# Description:   A deep learning Streamlit application turning images into 
#                coherent 50-100 word stories for children aged 3-10.
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
    Function 1: Image Captioning.
    Uses BLIP to generate a base description of the uploaded image.[cite: 1]
    """
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt")

    output = model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption


def generate_story(scenario):
    """
    Function 2: Story Generation.
    Uses a specialized story-generation model to expand the caption into a 
    narrative (50-100 words) for children.[cite: 1]
    """
    # Using the professor's recommended storytelling model for better logic[cite: 1]
    story_pipe = pipeline("text-generation", model="pranavpsv/genre-story-generator-v2")
    
    # We prefix the scenario with a 'kids' genre tag to guide the model[cite: 1]
    prompt = f"kids story: {scenario}"

    # min_new_tokens ensures we meet the minimum word count requirement[cite: 1]
    story_results = story_pipe(
        prompt, 
        max_length=150, 
        min_new_tokens=80, 
        do_sample=True, 
        temperature=0.7,
        repetition_penalty=1.2
    )
    
    story = story_results[0]['generated_text']
    
    # Clean up: Remove the prompt prefix from the final output
    story = story.replace(prompt, "").strip()

    # Final word count enforcement (50-100 words)[cite: 1]
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
    Function 3: Text-to-Speech Conversion.
    Utilizes gTTS to create an engaging audio experience for kids.[cite: 1]
    """
    tts = gTTS(text=story_text, lang="en", slow=False)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_file.name)
    return temp_file.name


# ── Main Part (Function 4) ──────────────────────────────────────────────────

def main():
    """
    Function 4: The Application Framework.
    Manages the Streamlit UI and ensures modular execution of tasks.[cite: 1]
    """
    st.set_page_config(
        page_title="Magic Story Machine",
        page_icon="🪄",
        layout="centered"
    )

    # Custom CSS for Kid-Friendly User Experience[cite: 1]
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
        # Save file locally for model processing[cite: 1]
        bytes_data = uploaded_file.getvalue()
        file_path = uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(bytes_data)

        st.image(uploaded_file, caption="🖼️ Your awesome picture!", use_container_width=True)

        # Stage 1: Image to Text
        st.markdown('<p class="step-label">🔍 Step 2: What\'s in your picture?</p>', unsafe_allow_html=True)
        with st.spinner("🧐 Looking at your picture really carefully..."):
            scenario = img2text(file_path)
        st.markdown(f'<div class="caption-box">I see: <strong>{scenario}</strong></div>', unsafe_allow_html=True)

        # Stage 2: Text to Story
        st.markdown('<p class="step-label">📝 Step 3: Story time!</p>', unsafe_allow_html=True)
        with st.spinner("✍️ Writing a magical story just for you..."):
            story = generate_story(scenario)
        st.markdown(f'<div class="story-box">📖 {story}</div>', unsafe_allow_html=True)

        # Stage 3: Story to Audio
        st.markdown('<p class="step-label">🔊 Step 4: Listen to your story!</p>', unsafe_allow_html=True)
        with st.spinner("🎵 Getting the story ready to read aloud..."):
            audio_file_path = text2audio(story)

        # Audio Output
        with open(audio_file_path, "rb") as audio_file:
            audio_bytes = audio_file.read()
        st.audio(audio_bytes, format="audio/mp3")

        # Visual Feedback[cite: 1]
        st.balloons()
        st.success("🎉 Your story is ready! Press play to listen! 🎧")

    st.markdown('<p class="fun-footer">Made with ❤️ for little storytellers everywhere 🌈</p>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
