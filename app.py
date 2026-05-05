import streamlit as st
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration, AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration
from PIL import Image
import tempfile
import soundfile as sf
import numpy as np

# ── 🚨 CRITICAL FIX: CACHE YOUR MODELS ──────────────────────────────────────
@st.cache_resource(show_spinner="Loading Heavy AI Models into Memory...")
def load_models():
    # 1. Vision
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    
    # 2. Story
    story_pipe = pipeline("text-generation", model="Prashant-karwasra/GPT2_text_generation_model")
    
    # 3. Audio
    tts_model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-tiny-v1")
    tts_tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-tiny-v1")
    
    return blip_processor, blip_model, story_pipe, tts_model, tts_tokenizer

# Initialize them globally
processor, img_model, story_generator, audio_model, audio_tokenizer = load_models()

# ── Function Part ───────────────────────────────────────────────────────────

def img2text(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt")
    output = img_model.generate(**inputs, max_new_tokens=50)
    return processor.decode(output[0], skip_special_tokens=True)

def text2story(caption):
    prompt = f"Write a short, happy story for a 5-year-old child about {caption}. Use simple words and a happy ending. Once upon a time, {caption}. "
    
    result = story_generator(
        prompt,
        max_length=150,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.7
    )
    story = result[0]["generated_text"]

    story_start = story.find("Once upon a time")
    if story_start != -1:
        story = story[story_start:]

    words = story.split()
    if len(words) > 100:
        trimmed = " ".join(words[:100])
        for punct in [".", "!", "?"]:
            last_pos = trimmed.rfind(punct)
            if last_pos != -1:
                trimmed = trimmed[: last_pos + 1]
                break
        story = trimmed
    return story

def text2audio(story_text):
    voice_description = (
        "A female speaker reads a children's story with a warm, "
        "friendly, and expressive tone at a moderate speed. "
        "The recording is clear and close up."
    )

    input_ids = audio_tokenizer(voice_description, return_tensors="pt").input_ids
    prompt_input_ids = audio_tokenizer(story_text, return_tensors="pt").input_ids

    generation = audio_model.generate(
        input_ids=input_ids,
        prompt_input_ids=prompt_input_ids
    )

    audio_array = generation.cpu().numpy().squeeze()
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    
    # Use audio_model.config.sampling_rate
    sf.write(temp_file.name, audio_array, audio_model.config.sampling_rate)
    return temp_file.name

# ... (Keep your main() function exactly as you wrote it) ...
