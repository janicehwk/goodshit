def generate_story(caption):
    """
    優化版：生成約 50-100 字的兒童故事。
    """
    # 建立一個更有引導性的 Prompt，告訴模型這是一個給孩子的故事
    prompt = (
        f"Once upon a time, {caption}. "
        "It was a sunny day and something magical was about to happen. "
        "Then, "
    )

    # 這裡建議加上 @st.cache_resource 放在全域，但為了符合你原代碼結構，維持在函式內
    story_generator = pipeline(
        "text-generation",
        model="Prashant-karwasra/GPT2_text_generation_model"
    )

    # 參數說明：
    # max_length: Token 數，120 太短，100個單字大約需要 150-200 token
    # min_length: 強制模型不要太早結束
    # repetition_penalty: 設為 1.2 防止 GPT-2 反覆說同樣的話
    result = story_generator(
        prompt,
        max_length=200, 
        min_length=80,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.2
    )
    
    story = result[0]["generated_text"]

    # 移除 Prompt 部分，只留下生成的故事內容（可選）
    # story = story.replace(prompt, "")

    # 修剪邏輯：確保故事在 50-120 字之間，並在完整句子結束
    words = story.split()
    if len(words) > 120:
        trimmed = " ".join(words[:120])
        # 尋找最後一個句號，讓結尾自然
        for punctuation in [".", "!", "?"]:
            last_pos = trimmed.rfind(punctuation)
            if last_pos != -1:
                trimmed = trimmed[: last_pos + 1]
                break
        story = trimmed

    return story
