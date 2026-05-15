import streamlit as st
from transformers import pipeline
import PyPDF2
from docx import Document
import re

st.set_page_config(page_title="HRVibeCheck", page_icon="👔", layout="wide")
st.markdown("""
    <style>
    .main-header { font-size: 2.8rem; font-weight: 700; color: #1e3a8a; margin-bottom: 0; }
    .sub-header { font-size: 1.1rem; color: #64748b; }
    .stMetric { background-color: white; padding: 25px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); }
    .skill-pill { background-color: #3b82f6; color: white; padding: 10px 20px; border-radius: 30px; margin: 5px; display: inline-block; font-weight: 500; }
    .skill-pill-match { background-color: #22c55e; color: white; padding: 10px 20px; border-radius: 30px; margin: 5px; display: inline-block; font-weight: 500; }
    .skill-pill-miss { background-color: #ef4444; color: white; padding: 10px 20px; border-radius: 30px; margin: 5px; display: inline-block; font-weight: 500; }
    .seniority-box { background: linear-gradient(135deg, #6366f1, #4f46e5); color: white; padding: 20px; border-radius: 16px; font-size: 1.15em; font-weight: 600; }
    .match-box-high { background: linear-gradient(135deg, #16a34a, #22c55e); color: white; padding: 20px; border-radius: 16px; font-size: 1.15em; font-weight: 600; }
    .match-box-mid  { background: linear-gradient(135deg, #d97706, #f59e0b); color: white; padding: 20px; border-radius: 16px; font-size: 1.15em; font-weight: 600; }
    .match-box-low  { background: linear-gradient(135deg, #dc2626, #ef4444); color: white; padding: 20px; border-radius: 16px; font-size: 1.15em; font-weight: 600; }
    .resume-box { background-color: #0f172a; color: #e2e8f0; padding: 28px; border-radius: 16px; line-height: 1.85; border-left: 5px solid #64748b; }
    </style>
    """, unsafe_allow_html=True)

# ── Fallback labels (only used when no job description is provided) ──
FALLBACK_SKILL_LABELS = [
    "Python", "Java", "JavaScript", "SQL", "Machine Learning", "Deep Learning",
    "AWS", "Azure", "Docker", "Kubernetes", "React", "Node.js", "Django",
    "TensorFlow", "PyTorch", "Data Analysis", "Leadership", "Project Management",
    "Communication", "Team Management", "Cloud Computing", "DevOps", "API",
    "Excel", "Tableau", "Power BI", "C++", "R", "Agile", "Scrum"
]

STOPWORDS = {
    "and", "or", "the", "with", "for", "from", "that", "this", "will", "have",
    "has", "are", "our", "you", "your", "their", "we", "be", "able", "must",
    "good", "strong", "experience", "knowledge", "understanding", "skills",
    "skill", "ability", "work", "working", "team", "role", "minimum", "least",
    "years", "year", "plus", "candidate", "candidates", "required", "preferred",
    "including", "such", "use", "using", "used", "well", "also", "etc"
}

# ── Load models once ──
@st.cache_resource(show_spinner="Loading AI Models...")
def load_pipelines():
    pipe1 = pipeline("text-classification",
                     model="Cheykong/HRVibeCheck-Retention-Predictor",
                     device=-1)
    pipe23 = pipeline("zero-shot-classification",
                      model="MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
                      device=-1)
    return pipe1, pipe23

pipe1, pipe23 = load_pipelines()

# ── Helper: extract text from uploaded file ──
def extract_text_from_file(uploaded_file):
    try:
        if uploaded_file.type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            return "".join([page.extract_text() or "" for page in pdf_reader.pages])
        elif "wordprocessingml" in uploaded_file.type:
            from docx import Document
            doc = Document(uploaded_file)
            return "\n".join([para.text for para in doc.paragraphs])
        return None
    except Exception as e:
        st.error(f"Could not read file: {e}")
        return None

# ── Helper: extract keywords from job description text ──
def extract_keywords_from_jd(jd_text):
    tokens = re.split(r'[,\n\-•/():;]+', jd_text)
    keywords = []
    for t in tokens:
        t = t.strip()
        if 2 < len(t) < 40 and t.lower() not in STOPWORDS:
            keywords.append(t)
    seen = set()
    unique_keywords = []
    for k in keywords:
        if k.lower() not in seen:
            seen.add(k.lower())
            unique_keywords.append(k)
    return unique_keywords[:30]

# ── Main app ──
def main():
    st.markdown("<h1 class='main-header'>👔 HRVibeCheck</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>AI-Powered Resume Screening • Smart Hiring Assistant</p>", unsafe_allow_html=True)

    with st.expander("📘 What is Hire Recommendation Score?", expanded=False):
        st.markdown("""
        **Hire Recommendation Score** (0–100%) represents our AI's confidence in recommending a candidate for hire.
        **How it is calculated**: The model was fine-tuned on real historical hiring decisions (`Hire` vs `Reject`).
        **Score Interpretation**:
        | Score Range     | Recommendation       | Meaning |
        |-----------------|----------------------|--------|
        | **75% – 100%**  | **Strong Hire**      | Excellent fit |
        | **60% – 74%**   | **Good Hire**        | Solid candidate |
        | **45% – 59%**   | **Moderate Fit**     | Needs more evaluation |
        | **Below 45%**   | **Further Review**   | High risk |
        """)

    # Sidebar
    with st.sidebar:
        st.header("📋 Job Role Details")
        job_role = st.text_input("Job Title", placeholder="e.g. Data Scientist, Software Engineer")
        job_description = st.text_area("Job Requirements / Description",
                                       height=160,
                                       placeholder="Paste the job description or list required skills here...")
        st.divider()

        st.header("📄 Candidate Resume")
        uploaded_file = st.file_uploader("Upload PDF or Word File", type=["pdf", "docx"])
        manual_text = st.text_area("Or paste resume text here", height=200,
                                   placeholder="Paste candidate resume text...")
        st.divider()
        analyze_btn = st.button("🚀 Run Full Analysis", type="primary", use_container_width=True)

    # Get resume text
    if uploaded_file:
        resume_text = extract_text_from_file(uploaded_file)
    else:
        resume_text = manual_text

    # Run analysis
    if analyze_btn and resume_text:
        with st.spinner("🤖 Analyzing resume with AI pipelines..."):

            # Pipeline 1: Hire Recommendation
            p1 = pipe1(resume_text[:512])[0]
            score = p1['score']
            is_strong = score > 0.55

            # Pipeline 2: Job Match or general skill detection
            job_match_score = None
            matched_skills = []
            missing_skills = []
            required_skills = []
            candidate_skills = []

            if job_description.strip():
                jd_keywords = extract_keywords_from_jd(job_description)
                if jd_keywords:
                    jd_result = pipe23(job_description[:1000], jd_keywords, multi_label=True)
                    required_skills = [l for l, sc in zip(jd_result['labels'], jd_result['scores']) if sc > 0.4]

                    resume_result = pipe23(resume_text[:1000], jd_keywords, multi_label=True)
                    candidate_skills = [l for l, sc in zip(resume_result['labels'], resume_result['scores']) if sc > 0.4]

                    if required_skills:
                        matched_skills = list(set(candidate_skills) & set(required_skills))
                        missing_skills = list(set(required_skills) - set(candidate_skills))
                        job_match_score = len(matched_skills) / len(required_skills)
            else:
                resume_result = pipe23(resume_text[:1000], FALLBACK_SKILL_LABELS, multi_label=True)
                candidate_skills = [l for l, sc in zip(resume_result['labels'], resume_result['scores']) if sc > 0.35][:10]

            # Pipeline 3: Seniority
            seniority_labels = ["Senior Level (5+ years)", "Mid Level (2-5 years)",
                                "Junior Level (0-2 years)", "Entry Level / Fresh Graduate"]
            p3 = pipe23(resume_text[:1500], seniority_labels, multi_label=False)
            predicted_level = p3['labels'][0]
            confidence = p3['scores'][0]

        st.success("✅ Analysis Complete!")

        tab1, tab2, tab3, tab4 = st.tabs(["📊 Assessment", "🎯 Job Match", "🔑 Skills & Seniority", "📄 Resume"])

        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("**Hire Recommendation Score**", f"{score:.1%}",
                          delta="Strong Hire" if is_strong else "Further Review",
                          delta_color="normal" if is_strong else "inverse")
            if job_match_score is not None:
                with col2:
                    match_label = "Strong Match" if job_match_score >= 0.7 else ("Partial Match" if job_match_score >= 0.4 else "Low Match")
                    st.metric("**Job Suitability Score**", f"{job_match_score:.1%}",
                              delta=match_label,
                              delta_color="normal" if job_match_score >= 0.7 else "inverse")

        with tab2:
            if not job_description.strip():
                st.info("💡 Fill in the **Job Requirements** in the sidebar to see job match results.")
            else:
                role_display = f" — {job_role}" if job_role.strip() else ""
                st.subheader(f"🎯 Job Suitability{role_display}")

                if required_skills:
                    if job_match_score >= 0.7:
                        box_class, verdict = "match-box-high", "Strong Match ✅"
                    elif job_match_score >= 0.4:
                        box_class, verdict = "match-box-mid", "Partial Match ⚠️"
                    else:
                        box_class, verdict = "match-box-low", "Low Match ❌"

                    st.markdown(f"""
                    <div class='{box_class}'>
                        Suitability Score: {job_match_score:.1%} &nbsp;|&nbsp; {verdict}<br>
                        Matched {len(matched_skills)} of {len(required_skills)} required skills
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("---")
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown("**✅ Skills Matched**")
                        if matched_skills:
                            st.markdown(" ".join([f"<span class='skill-pill-match'>{s}</span>" for s in matched_skills]), unsafe_allow_html=True)
                        else:
                            st.info("No matching skills found.")
                    with col_b:
                        st.markdown("**❌ Skills Missing**")
                        if missing_skills:
                            st.markdown(" ".join([f"<span class='skill-pill-miss'>{s}</span>" for s in missing_skills]), unsafe_allow_html=True)
                        else:
                            st.success("Candidate meets all detected skill requirements!")
                else:
                    st.warning("Could not detect specific skills from the job description. Try adding more detail.")

        with tab3:
            st.subheader("🔑 Skills Detected from Resume")
            if candidate_skills:
                st.markdown(" ".join([f"<span class='skill-pill'>{s}</span>" for s in candidate_skills]), unsafe_allow_html=True)
            else:
                st.info("No strong skills detected.")

            st.divider()
            st.subheader("📊 Candidate Seniority / Experience Level")
            st.markdown(f"""
            <div class='seniority-box'>
                Predicted Level: {predicted_level}<br>
                Confidence: {confidence:.1%}
            </div>
            """, unsafe_allow_html=True)

        with tab4:
            st.subheader("📄 Original Resume")
            st.markdown(f"<div class='resume-box'>{resume_text}</div>", unsafe_allow_html=True)

    elif analyze_btn:
        st.warning("⚠️ Please upload a file or paste resume text before running analysis.")

if __name__ == "__main__":
    main()
