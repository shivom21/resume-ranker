import streamlit as st
import spacy
import subprocess
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
# Ensure spaCy model is installed
# ---------------------------
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# ---------------------------
# Helper functions
# ---------------------------

def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and not token.like_num]
    return " ".join(tokens)

def rank_resumes(job_description, resumes):
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

st.title("Resume Ranker 🚀")
st.write("Upload resumes and a job description to rank the best matches.")

job_description = st.text_area("Job Description", height=200)
uploaded_files = st.file_uploader("Upload Resumes (TXT format):", type=["txt"], accept_multiple_files=True)

if st.button("Rank Resumes"):
    if not job_description:
        st.error("Please add a job description.")
    elif not uploaded_files:
        st.error("Please upload at least one resume.")
    else:
        resumes = [file.read().decode("utf-8") for file in uploaded_files]
        ranked = rank_resumes(job_description, resumes)

        st.subheader("Top Matches:")
        for idx, (resume, score) in enumerate(ranked, 1):
            st.markdown(f"**Rank {idx}** (Score: {score:.2f})")
            st.text_area("Resume", resume, height=150)
