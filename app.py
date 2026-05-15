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
    .verdict-select { background: linear-gradient(135deg, #16a34a, #22c55e); color: white; padding: 24px 32px; border-radius: 16px; font-size: 1.4em; font-weight: 700; text-align: center; }
    .verdict-reject { background: linear-gradient(135deg, #dc2626, #ef4444); color: white; padding: 24px 32px; border-radius: 16px; font-size: 1.4em; font-weight: 700; text-align: center; }
    .seniority-box { background: linear-gradient(135deg, #6366f1, #4f46e5); color: white; padding: 20px; border-radius: 16px; font-size: 1.15em; font-weight: 600; }
    .hire-box-high { background: linear-gradient(135deg, #16a34a, #22c55e); color: white; padding: 20px; border-radius: 16px; font-size: 1.15em; font-weight: 600; }
    .hire-box-low  { background: linear-gradient(135deg, #dc2626, #ef4444); color: white; padding: 20px; border-radius: 16px; font-size: 1.15em; font-weight: 600; }
    .resume-box { background-color: #0f172a; color: #e2e8f0; padding: 28px; border-radius: 16px; line-height: 1.85; border-left: 5px solid #64748b; }
    .info-box { background-color: #f1f5f9; padding: 16px 20px; border-radius: 12px; border-left: 4px solid #3b82f6; font-size: 0.95em; color: #334155; }
    </style>
    """, unsafe_allow_html=True)

# ==================== LOAD PIPELINES ====================
@st.cache_resource(show_spinner="Loading AI Models...")
def load_pipelines():
    pipe1 = pipeline(
        "text-classification",
        model="Cheykong/HRVibeCheck-Retention-Predictor",
        device=-1
    )
    pipe23 = pipeline(
        "zero-shot-classification",
        model="MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
        device=-1
    )
    return pipe1, pipe23

pipe1, pipe23 = load_pipelines()


# ==================== HELPER FUNCTIONS ====================
def extract_text_from_file(uploaded_file):
    try:
        if uploaded_file.type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            return "".join([page.extract_text() or "" for page in pdf_reader.pages])
        elif "wordprocessingml" in uploaded_file.type:
            doc = Document(uploaded_file)
            return "\n".join([para.text for para in doc.paragraphs])
        return None
    except Exception as e:
        st.error(f"Could not read file: {e}")
        return None


def predict_job_fit(resume_text, job_description):
    """
    Pipeline 2: Zero-shot classify resume against job description.
    Combines both texts and asks the model: select or reject?
    Returns: ('Select' or 'Reject', confidence score)
    """
    combined = (
        f"Job requirements: {job_description[:400]}\n\n"
        f"Candidate resume: {resume_text[:400]}"
    )
    result = pipe23(combined, ["suitable candidate for this job", "unsuitable candidate for this job"])
    top_label = result['labels'][0]
    confidence = round(result['scores'][0], 4)
    verdict = "Select" if "suitable candidate for this job" in top_label else "Reject"
    return verdict, confidence


# ==================== MAIN APP ====================
def main():
    st.markdown("<h1 class='main-header'>👔 HRVibeCheck</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>AI-Powered Resume Screening • Smart Hiring Assistant</p>", unsafe_allow_html=True)

    with st.expander("📘 How does HRVibeCheck work?", expanded=False):
        st.markdown("""
        HRVibeCheck runs **3 AI pipelines** on every resume:

        | Pipeline | What it does | Output |
        |----------|-------------|--------|
        | **Pipeline 1** — Hire Score | Fine-tuned DistilBERT trained on real recruiter decisions | Select / Reject + confidence % |
        | **Pipeline 2** — Job Fit | Zero-shot classification: does this resume match the job description? | Select / Reject + confidence % |
        | **Pipeline 3** — Seniority | Zero-shot classification of experience level | Junior / Mid / Senior |

        **Score Interpretation (Pipeline 1)**:
        | Score | Recommendation |
        |-------|---------------|
        | 75–100% | Strong Hire |
        | 60–74%  | Good Hire |
        | 45–59%  | Moderate Fit |
        | Below 45% | Further Review |
        """)

    # ==================== SIDEBAR ====================
    with st.sidebar:
        st.header("📋 Job Role Details")
        job_role = st.text_input("Job Title", placeholder="e.g. Data Scientist, Software Engineer")
        job_description = st.text_area(
            "Job Requirements / Description",
            height=160,
            placeholder="Paste the job description or list required skills here..."
        )
        st.divider()

        st.header("📄 Candidate Resume")
        uploaded_file = st.file_uploader("Upload PDF or Word File", type=["pdf", "docx"])
        manual_text = st.text_area(
            "Or paste resume text here",
            height=200,
            placeholder="Paste candidate resume text..."
        )
        st.divider()
        analyze_btn = st.button("🚀 Run Full Analysis", type="primary", use_container_width=True)

    # ==================== RESUME TEXT ====================
    if uploaded_file:
        resume_text = extract_text_from_file(uploaded_file)
    else:
        resume_text = manual_text

    # ==================== ANALYSIS ====================
    if analyze_btn and resume_text:
        with st.spinner("🤖 Analyzing resume with AI pipelines..."):

            # --- Pipeline 1: Hire Recommendation Score ---
            p1         = pipe1(resume_text[:512])[0]
            hire_score = round(p1['score'], 4)
            p1_verdict = "Select" if hire_score > 0.55 else "Reject"

            # --- Pipeline 2: Job Fit (Select / Reject) ---
            p2_verdict    = None
            p2_confidence = None
            if job_description.strip():
                p2_verdict, p2_confidence = predict_job_fit(resume_text, job_description)

            # --- Pipeline 3: Seniority ---
            seniority_labels = [
                "Senior Level (5+ years)",
                "Mid Level (2-5 years)",
                "Junior Level (0-2 years)",
                "Entry Level / Fresh Graduate"
            ]
            p3              = pipe23(resume_text[:1500], seniority_labels, multi_label=False)
            predicted_level = p3['labels'][0]
            sen_confidence  = p3['scores'][0]

        st.success("✅ Analysis Complete!")

        # ==================== TABS ====================
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Assessment", "🎯 Job Fit", "🔑 Seniority", "📄 Resume"])

        # --- Tab 1: Assessment Summary ---
        with tab1:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Pipeline 1 — Hire Recommendation**")
                box  = "hire-box-high" if p1_verdict == "Select" else "hire-box-low"
                icon = "✅" if p1_verdict == "Select" else "❌"
                st.markdown(f"""
                <div class='{box}'>
                    {icon} {p1_verdict}<br>
                    <span style='font-size:0.85em; font-weight:400;'>Confidence: {hire_score:.1%}</span>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown("**Pipeline 2 — Job Fit**")
                if p2_verdict:
                    box2  = "verdict-select" if p2_verdict == "Select" else "verdict-reject"
                    icon2 = "✅" if p2_verdict == "Select" else "❌"
                    st.markdown(f"""
                    <div class='{box2}'>
                        {icon2} {p2_verdict}<br>
                        <span style='font-size:0.85em; font-weight:400;'>Confidence: {p2_confidence:.1%}</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class='info-box'>
                        💡 Add a <b>Job Description</b> in the sidebar to get a Job Fit verdict.
                    </div>
                    """, unsafe_allow_html=True)

        # --- Tab 2: Job Fit Detail ---
        with tab2:
            role_display = f" for {job_role}" if job_role.strip() else ""
            st.subheader(f"🎯 Job Fit Verdict{role_display}")

            if not job_description.strip():
                st.info("💡 Fill in the **Job Requirements** in the sidebar to see job fit results.")
            else:
                box2  = "verdict-select" if p2_verdict == "Select" else "verdict-reject"
                icon2 = "✅" if p2_verdict == "Select" else "❌"
                st.markdown(f"""
                <div class='{box2}'>
                    {icon2} &nbsp; Verdict: {p2_verdict} &nbsp;|&nbsp; Confidence: {p2_confidence:.1%}
                </div>
                """, unsafe_allow_html=True)

                st.markdown("---")
                st.markdown("**Job Description provided:**")
                st.markdown(f"<div class='info-box'>{job_description}</div>", unsafe_allow_html=True)

        # --- Tab 3: Seniority ---
        with tab3:
            st.subheader("📊 Candidate Seniority / Experience Level")
            st.markdown(f"""
            <div class='seniority-box'>
                Predicted Level: {predicted_level}<br>
                Confidence: {sen_confidence:.1%}
            </div>
            """, unsafe_allow_html=True)

        # --- Tab 4: Resume ---
        with tab4:
            st.subheader("📄 Original Resume")
            st.markdown(f"<div class='resume-box'>{resume_text}</div>", unsafe_allow_html=True)

    elif analyze_btn:
        st.warning("⚠️ Please upload a file or paste resume text before running analysis.")


if __name__ == "__main__":
    main()
