import streamlit as st
import spacy
import subprocess
import sys
import pandas as pd
import fitz  # PyMuPDF for PDF extraction
import docx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
# Ensure SpaCy model is installed
# ---------------------------
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# ---------------------------
# Helper functions
# ---------------------------

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file."""
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text("text")
    return text

def extract_text_from_docx(docx_file):
    """Extract text from a DOCX file."""
    doc = docx.Document(docx_file)
    return "\n".join([para.text for para in doc.paragraphs])

def preprocess_text(text):
    """Preprocess text: Tokenization, Stopword Removal, Lemmatization."""
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and not token.like_num]
    return " ".join(tokens)

def rank_resumes(job_description, resumes):
    """Compute similarity between job description and resumes."""
    processed_jd = preprocess_text(job_description)
    processed_resumes = [preprocess_text(resume) for resume in resumes]

    documents = [processed_jd] + processed_resumes
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    ranked_resumes = sorted(zip(resumes, similarities), key=lambda x: x[1], reverse=True)
    return ranked_resumes

# ---------------------------
# Streamlit UI
# ---------------------------

st.set_page_config(page_title="Resume Ranker", page_icon="📄", layout="wide")

st.title("📄 Resume Ranker 🚀")
st.write("Upload resumes and enter job details to find the best match!")

# Job Description Input
job_description = st.text_area("📝 Job Description", height=200)

# Additional Inputs
technical_skills = st.text_input("🛠️ Required Technical Skills (comma-separated)")
soft_skills = st.text_input("🤝 Required Soft Skills (comma-separated)")
qualifications = st.text_input("🎓 Required Qualifications (comma-separated)")

# File Upload
uploaded_files = st.file_uploader("📂 Upload Resumes (PDF/DOCX):", type=["pdf", "docx"], accept_multiple_files=True)

if st.button("🔍 Rank Resumes"):
    if not job_description:
        st.error("⚠️ Please enter a job description.")
    elif not uploaded_files:
        st.error("⚠️ Please upload at least one resume.")
    else:
        resume_texts = []
        for file in uploaded_files:
            if file.name.endswith(".pdf"):
                resume_texts.append(extract_text_from_pdf(file))
            elif file.name.endswith(".docx"):
                resume_texts.append(extract_text_from_docx(file))

        ranked = rank_resumes(job_description, resume_texts)

        st.subheader("🏆 Ranked Resumes:")
        results = []
        for idx, (resume, score) in enumerate(ranked, 1):
            st.markdown(f"**Rank {idx}** (Score: {score:.2f})")
            st.text_area("Resume Extract", resume[:500] + "...", height=150)  # Show first 500 chars
            results.append({"Rank": idx, "Score": round(score, 2), "Resume": resume[:500]})

        # Download results as CSV
        df = pd.DataFrame(results)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(label="📥 Download Results", data=csv, file_name="resume_ranking.csv", mime="text/csv")
