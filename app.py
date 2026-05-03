import streamlit as st
from transformers import pipeline
from PIL import Image

# 1. Model Loading (Cached so it only runs once)
@st.cache_resource
def load_models():
    captioner = pipeline(
        "image-captioning", 
        model="Salesforce/blip-image-captioning-base"
    )
    story_gen = pipeline(
        "text-generation", 
        model="gpt2"
    )
    return captioner, story_gen

caption_model, story_model = load_models()

def main():
    st.set_page_config(page_title="Image Story Generator", page_icon="🖼️")
    st.title("🖼️ Image to Story Generator")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Your Uploaded Image", use_container_width=True)

        if st.button("Generate Story"):
            with st.spinner("Generating magic..."):
                try:
                    # Step 1: Captioning
                    captions = caption_model(image)
                    base_caption = captions[0]['generated_text']
                    
                    st.subheader("📝 Description")
                    st.info(base_caption)

                    # Step 2: Story Generation
                    prompt = f"Once upon a time, there was {base_caption}. "
                    
                    stories = story_model(
                        prompt, 
                        max_length=150, 
                        do_sample=True, 
                        temperature=0.8, 
                        truncation=True
                    )
                    
                    full_story = stories[0]['generated_text']

                    st.subheader("📖 The Story")
                    st.write(full_story)
                
                except Exception as e:
                    st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
                    # Step 1: Captioning
                    # BLIP expects a PIL Image or path
                    captions = caption_model(image)
                    base_caption = captions[0]['generated_text']
                    
                    st.subheader("Description")
                    st.write(base_caption)

                    # Step 2: Story Generation
                    # Constructing a clean prompt
                    prompt = f"Once upon a time, there was {base_caption}. "
                    
                    stories = story_model(
                        prompt, 
                        max_length=150, 
                        do_sample=True, 
                        temperature=0.7,
                        truncation=True
                    )
                    
                    full_story = stories[0]['generated_text']

                    st.subheader("The Story")
                    st.write(full_story)
                
                except Exception as e:
                    st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        
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
