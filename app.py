# ============================================================================
# Program Title: Magic Story Machine - A Storytelling App for Kids
# Description:   A kid-friendly Streamlit application that turns uploaded
#                images into fun audio stories for children aged 3-10.
# Models Used:
#   1. Image Captioning — Salesforce/blip-image-captioning-base
#   2. Story Generation — Prashant-karwasra/GPT2_text_generation_model
#   3. Text-to-Speech   — parler-tts/parler-tts-tiny-v1
# ============================================================================

# ── Import Part ─────────────────────────────────────────────────────────────
import streamlit as st
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration, AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration
from PIL import Image
import tempfile
import soundfile as sf
import numpy as np


# ── Function Part ───────────────────────────────────────────────────────────

def img2text(image_path):
    """
    Convert an uploaded image into a text caption.
    Uses the BLIP image captioning model, loaded directly via
    BlipProcessor and BlipForConditionalGeneration for compatibility
    with the latest version of the transformers library.

    Parameters:
        image_path (str): File path of the uploaded image.
    Returns:
        str: A short caption describing what is in the image.
    """
    # Load the BLIP processor and model
    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )

    # Open the image and convert to RGB format
    image = Image.open(image_path).convert("RGB")

    # Process the image into model-ready format
    inputs = processor(image, return_tensors="pt")

    # Generate the caption from the image
    output = model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption


def text2story(caption):
    """
    Generate a short kid-friendly story (50-100 words) from an image caption.
    Uses a GPT2-based text generation model. The prompt is written to produce
    simple, cheerful stories suitable for children aged 3-10.

    Parameters:
        caption (str): A short image caption to build the story around.
    Returns:
        str: A fun, age-appropriate story for young children.
    """
    # Build a kid-friendly story prompt
    prompt = (
        f"Write a short, happy story for a 5-year-old child about {caption}. "
        f"Use simple words and a happy ending. "
        f"Once upon a time, {caption}. "
    )

    # Load the text generation model
    story_generator = pipeline(
        "text-generation",
        model="Prashant-karwasra/GPT2_text_generation_model"
    )

    # Generate the story with controlled length
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

    # Trim the story to stay within 50-100 words
    words = story.split()
    if len(words) > 100:
        trimmed = " ".join(words[:100])
        # End at the last complete sentence
        for punct in [".", "!", "?"]:
            last_pos = trimmed.rfind(punct)
            if last_pos != -1:
                trimmed = trimmed[: last_pos + 1]
                break
        story = trimmed

    return story


def text2audio(story_text):
    """
    Convert a story into an audio file using Parler-TTS Tiny v1.
    This model produces natural, expressive speech. A voice description
    prompt is used to create a warm, friendly tone suitable for
    reading children's stories aloud.

    Parameters:
        story_text (str): The story text to convert to speech.
    Returns:
        str: File path of the generated audio (.wav) file.
    """
    # Load the Parler TTS model and tokenizer
    model = ParlerTTSForConditionalGeneration.from_pretrained(
        "parler-tts/parler-tts-tiny-v1"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "parler-tts/parler-tts-tiny-v1"
    )

    # Describe the voice style for a warm, kid-friendly reading
    voice_description = (
        "A female speaker reads a children's story with a warm, "
        "friendly, and expressive tone at a moderate speed. "
        "The recording is clear and close up."
    )

    # Tokenize the voice description and the story text
    input_ids = tokenizer(voice_description, return_tensors="pt").input_ids
    prompt_input_ids = tokenizer(story_text, return_tensors="pt").input_ids

    # Generate the audio
    generation = model.generate(
        input_ids=input_ids,
        prompt_input_ids=prompt_input_ids
    )

    # Convert to numpy array
    audio_array = generation.cpu().numpy().squeeze()

    # Save to a temporary WAV file so Streamlit can play it
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(temp_file.name, audio_array, model.config.sampling_rate)
    return temp_file.name


def main():
    """
    Main function that runs the Streamlit application.
    Sets up the kid-friendly UI and runs the three-stage pipeline:
    image upload → caption → story → audio playback.
    """

    # ── Page Setup ──────────────────────────────────────────────────────
    st.set_page_config(
        page_title="Magic Story Machine",
        page_icon="🪄",
        layout="centered"
    )

    # ── Kid-Friendly CSS Styling ────────────────────────────────────────
    st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(135deg, #FFDEE9 0%, #B5FFFC 100%);
        }
        .kid-title {
            text-align: center;
            font-size: 3rem;
            font-weight: 800;
            color: #FF6B6B;
            text-shadow: 3px 3px 0px #FFE66D;
            margin-bottom: 0;
        }
        .kid-subtitle {
            text-align: center;
            font-size: 1.3rem;
            color: #6C5CE7;
            margin-top: 0;
            margin-bottom: 30px;
        }
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
        .caption-box {
            background: #DFE6E9;
            border-radius: 15px;
            padding: 15px 20px;
            font-size: 1.05rem;
            color: #2D3436;
            margin: 10px 0;
        }
        .fun-footer {
            text-align: center;
            color: #636E72;
            font-size: 0.9rem;
            margin-top: 40px;
            padding: 10px;
        }
    </style>
    """, unsafe_allow_html=True)

    # ── App Header ──────────────────────────────────────────────────────
    st.markdown(
        '<p class="kid-title">🪄 Magic Story Machine 🪄</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="kid-subtitle">'
        'Upload a picture and watch it turn into a story! 📖✨'
        '</p>',
        unsafe_allow_html=True,
    )

    # ── Step 1: Image Upload ────────────────────────────────────────────
    st.markdown(
        '<p class="step-label">📸 Step 1: Pick a Picture!</p>',
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(
        "Choose a fun image...",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        # Save the uploaded file locally so the model can read it
        bytes_data = uploaded_file.getvalue()
        file_path = uploaded_file.name
        with open(file_path, "wb") as file:
            file.write(bytes_data)

        # Display the uploaded image
        st.image(
            uploaded_file,
            caption="🖼️ Your awesome picture!",
            use_container_width=True,
        )

        # ── Step 2: Image → Caption (using img2text function) ──────────
        st.markdown(
            '<p class="step-label">🔍 Step 2: What\'s in your picture?</p>',
            unsafe_allow_html=True,
        )
        with st.spinner("🧐 Looking at your picture really carefully..."):
            scenario = img2text(file_path)
        st.markdown(
            f'<div class="caption-box">'
            f'I see: <strong>{scenario}</strong>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ── Step 3: Caption → Story (using text2story function) ────────
        st.markdown(
            '<p class="step-label">📝 Step 3: Story time!</p>',
            unsafe_allow_html=True,
        )
        with st.spinner("✍️ Writing a magical story just for you..."):
            story = text2story(scenario)
        st.markdown(
            f'<div class="story-box">📖 {story}</div>',
            unsafe_allow_html=True,
        )

        # ── Step 4: Story → Audio (using text2audio function) ──────────
        st.markdown(
            '<p class="step-label">🔊 Step 4: Listen to your story!</p>',
            unsafe_allow_html=True,
        )
        with st.spinner("🎵 Getting the story ready to read aloud..."):
            audio_file_path = text2audio(story)

        # Play the generated audio
        with open(audio_file_path, "rb") as audio_file:
            audio_bytes = audio_file.read()
        st.audio(audio_bytes, format="audio/wav")

        # Celebration
        st.balloons()
        st.success("🎉 Your story is ready! Press play to listen! 🎧")

    # Footer
    st.markdown(
        '<p class="fun-footer">'
        'Made with ❤️ for little storytellers everywhere 🌈'
        '</p>',
        unsafe_allow_html=True,
    )


# ── Run the App ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
