import streamlit as st
from transformers import pipeline

# Setting up the page config
st.set_page_config(page_title="Image Story Teller", page_icon="📖")

@st.cache_resource
def load_models():
    """
    Loads the ML models once and caches them to memory.
    """
    # FIX: Changed "image-to-text" to "image-captioning"
    captioner = pipeline(
        "image-captioning", 
        model="Salesforce/blip-image-captioning-base"
    )
    
    # Standard text generation pipeline for the story
    story_gen = pipeline(
        "text-generation", 
        model="gpt2"
    )
    
    return captioner, story_gen

# Initialize models
caption_model, story_model = load_models()

def main():
    st.title("Image to Story Generator")
    st.write("Upload an image and let AI spin a tale!")

    uploaded_file = st.file_冷静_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        
        if st.button("Generate Story"):
            with st.spinner("Analyzing image and writing story..."):
                # 1. Generate Caption
                # We pass the file directly to the pipeline
                caption_results = caption_model(uploaded_file)
                caption_text = caption_results[0]['generated_text']
                
                st.subheader("Image Description:")
                st.info(caption_text)
                
                # 2. Generate Story based on caption
                prompt = f"Once upon a time, {caption_text}. "
                story_results = story_model(
                    prompt, 
                    max_length=150, 
                    num_return_sequences=1,
                    truncation=True
                )
                story_text = story_results[0]['generated_text']
                
                st.subheader("The Story:")
                st.write(story_text)

if __name__ == "__main__":
    main()        # We give GPT-2 a very specific starting point to encourage length
        prompt = (
            f"Once upon a time, there was {caption}. "
            "It was a magical day and something amazing happened. "
            "First,"
        )
        
        # Setting min_new_tokens ensures the story isn't too short
        story_output = story_model(
            prompt, 
            max_new_tokens=150,  # Allows for ~100-150 words
            min_new_tokens=80,   # Forces it to be at least ~60-80 words
            do_sample=True, 
            temperature=0.7,
            truncation=True
        )
        story_text = story_output[0]['generated_text']
        
        st.subheader("📖 Your Story")
        st.write(story_text)

    # STAGE 3: Story to Audio
    with st.spinner("Preparing the audio..."):
        tts = gTTS(text=story_text, lang='en')
        tts.save("story.mp3")
        st.audio("story.mp3")
        st.balloons()
