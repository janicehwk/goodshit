import streamlit as st
from transformers import pipeline
import PyPDF2
from docx import Document
import pandas as pd
import torch

st.set_page_config(page_title="HRVibeCheck", page_icon="👔", layout="wide")

st.markdown("""
    <style>
    .main-header { font-size: 2.8rem; font-weight: 700; color: #1e3a8a; margin-bottom: 0; }
    .sub-header { font-size: 1.1rem; color: #64748b; margin-bottom: 1.5rem; }
    .skill-pill { background-color: #3b82f6; color: white; padding: 6px 14px; border-radius: 30px;
                  margin: 4px; display: inline-block; font-weight: 500; font-size: 0.85rem; }
    .rank-badge { background: linear-gradient(135deg, #1e3a8a, #3b82f6); color: white;
                  padding: 4px 12px; border-radius: 20px; font-weight: 700; }
    .score-box { background-color: white; padding: 20px; border-radius: 16px;
                 box-shadow: 0 4px 12px rgba(0,0,0,0.08); text-align: center; }
    </style>
    """, unsafe_allow_html=True)


# ==================== LOAD PIPELINES ====================
@st.cache_resource(show_spinner="Loading AI Models...")
def load_pipelines():
    # Pipeline 1: Hire Recommendation Score
    # TODO: Replace with your fine-tuned model after pushing to HuggingFace
    # e.g. model="your-hf-username/HRVibeCheck-Retention-Predictor"
    pipe1 = pipeline(
        "text-classification",
        model="Cheykong/HRVibeCheck-Retention-Predictor",
        device=0 if torch.cuda.is_available() else -1
    )

    # Pipeline 2: Skill Extraction (NER - no predefined list needed)
    pipe2 = pipeline(
        "token-classification",
        model="jjzha/jobbert-base-cased",
        aggregation_strategy="simple",
        device=0 if torch.cuda.is_available() else -1
    )

    return pipe1, pipe2

pipe1, pipe2 = load_pipelines()


# ==================== HELPER FUNCTIONS ====================
def extract_text_from_file(uploaded_file):
    """Extract raw text from PDF or DOCX file."""
    try:
        if uploaded_file.type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            return "".join([page.extract_text() or "" for page in pdf_reader.pages])
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = Document(uploaded_file)
            return "\n".join([para.text for para in doc.paragraphs])
        return None
    except:
        return None


def get_hire_score(resume_text, jd_text):
    """Pipeline 1: Get hire recommendation score from resume + JD."""
    combined = f"JOB DESCRIPTION: {jd_text} [SEP] RESUME: {resume_text}"
    result = pipe1(combined[:512])[0]
    label = result['label']
    score = result['score']
    if label in ['LABEL_1', 'POSITIVE', 'Hire', '1', 'HIRE']:
        return score
    else:
        return 1 - score


def extract_skills(resume_text):
    """Pipeline 2: Extract key skills from resume using NER (no predefined list)."""
    try:
        entities = pipe2(resume_text[:1500])
        skills = list(set([
            e['word'].strip()
            for e in entities
            if e['score'] > 0.7 and len(e['word'].strip()) > 1
        ]))
        return skills[:10]
    except:
        return []


def get_recommendation(score):
    """Return recommendation label based on score."""
    if score >= 0.75:
        return "✅ Strong Hire"
    elif score >= 0.60:
        return "👍 Good Hire"
    elif score >= 0.45:
        return "⚠️ Moderate Fit"
    else:
        return "❌ Further Review"


# ==================== MAIN APP ====================
def main():
    st.markdown("<h1 class='main-header'>👔 HRVibeCheck</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>AI-Powered Resume Screening • Smart Hiring Assistant</p>",
                unsafe_allow_html=True)

    with st.expander("📘 How does HRVibeCheck work?", expanded=False):
        st.markdown("""
        **HRVibeCheck** uses two AI pipelines to screen candidates:

        - **Pipeline 1 — Hire Recommendation Score**: A fine-tuned model compares each resume
          against the job description and outputs a hire confidence score (0–100%).
        - **Pipeline 2 — Key Skills Extraction**: An NER model reads each resume and automatically
          extracts the candidate's key skills — no predefined skill list needed.

        **Score Interpretation:**
        | Score Range    | Recommendation   |
        |----------------|-----------------|
        | 75% – 100%     | ✅ Strong Hire   |
        | 60% – 74%      | 👍 Good Hire     |
        | 45% – 59%      | ⚠️ Moderate Fit  |
        | Below 45%      | ❌ Further Review |
        """)

    # ── Sidebar ──────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("📋 Job Description")
        jd_text = st.text_area(
            "Paste the Job Description here",
            height=220,
            placeholder="e.g. We are looking for a Python developer with AWS and ML experience..."
        )

        st.divider()

        st.header("📄 Upload Resumes")
        uploaded_files = st.file_uploader(
            "Upload PDF or Word files",
            type=["pdf", "docx"],
            accept_multiple_files=True,
            help="Upload one or more candidate resumes to compare"
        )

        st.divider()
        analyze_btn = st.button("🚀 Analyze Candidates", type="primary", use_container_width=True)

    # ── Analysis ──────────────────────────────────────────────────────────────────
    if analyze_btn:
        if not jd_text.strip():
            st.warning("⚠️ Please enter a Job Description before analyzing.")
            return
        if not uploaded_files:
            st.warning("⚠️ Please upload at least one resume file.")
            return

        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, uploaded_file in enumerate(uploaded_files):
            candidate_name = uploaded_file.name.replace(".pdf", "").replace(".docx", "")
            status_text.text(f"🔍 Analyzing {candidate_name}... ({i+1}/{len(uploaded_files)})")

            resume_text = extract_text_from_file(uploaded_file)

            if resume_text and resume_text.strip():
                hire_score = get_hire_score(resume_text, jd_text)
                skills = extract_skills(resume_text)

                results.append({
                    "Candidate": candidate_name,
                    "Hire Score": hire_score,
                    "Score %": f"{hire_score:.1%}",
                    "Recommendation": get_recommendation(hire_score),
                    "Key Skills": skills,
                    "Resume Text": resume_text
                })
            else:
                st.warning(f"Could not extract text from {uploaded_file.name}. Skipping.")

            progress_bar.progress((i + 1) / len(uploaded_files))

        status_text.empty()
        progress_bar.empty()

        if not results:
            st.error("No resumes could be processed. Please check your files.")
            return

        results.sort(key=lambda x: x["Hire Score"], reverse=True)
        st.success(f"✅ Analysis Complete! {len(results)} candidate(s) analyzed.")

        # ── Rankings Table ────────────────────────────────────────────────────────
        st.subheader("🏆 Candidate Rankings")

        table_data = []
        for rank, r in enumerate(results, 1):
            table_data.append({
                "Rank": f"#{rank}",
                "Candidate": r["Candidate"],
                "Hire Score": r["Score %"],
                "Recommendation": r["Recommendation"],
                "Key Skills": ", ".join(r["Key Skills"]) if r["Key Skills"] else "—"
            })

        st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

        # ── Detailed Cards ────────────────────────────────────────────────────────
        st.subheader("📋 Candidate Details")

        for rank, r in enumerate(results, 1):
            with st.expander(f"#{rank} {r['Candidate']}  —  {r['Score %']}  {r['Recommendation']}"):
                col1, col2 = st.columns([1, 2])

                with col1:
                    st.metric("Hire Score", r["Score %"])
                    st.write(f"**Recommendation:** {r['Recommendation']}")

                with col2:
                    st.write("**Key Skills (Selling Points):**")
                    if r["Key Skills"]:
                        skill_html = " ".join([
                            f"<span class='skill-pill'>{s}</span>"
                            for s in r["Key Skills"]
                        ])
                        st.markdown(skill_html, unsafe_allow_html=True)
                    else:
                        st.info("No skills detected.")

                st.divider()
                st.write("**Resume Preview:**")
                preview = r["Resume Text"][:600] + "..." if len(r["Resume Text"]) > 600 else r["Resume Text"]
                st.text(preview)


if __name__ == "__main__":
    main()
