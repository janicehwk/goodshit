import re

def extract_keywords_from_jd(jd_text):
    # Simple keyword extraction - splits on common separators,
    # filters out short filler words
    tokens = re.split(r'[,\n\-•/()]+', jd_text)
    keywords = [t.strip() for t in tokens if 3 < len(t.strip()) < 40]
    return list(set(keywords))[:30]  # cap at 30 labels

# Then use these as labels instead of SKILL_LABELS
required_labels = extract_keywords_from_jd(job_description)
jd_result = pipe23(resume_text[:1000], required_labels, multi_label=True)
