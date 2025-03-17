import streamlit as st
import spacy
import subprocess
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# ---------------------------
# Ensure spaCy model is installed
# ---------------------------
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# ---------------------------
# Helper functions
# ---------------------------

def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and not token.like_num]
    return " ".join(tokens)

def rank_resumes(job_description, resumes):
    # Preprocess job description and resumes
    processed_jd = preprocess_text(job_description)
    processed_resumes = [preprocess_text(resume) for resume in resumes]

    # Combine JD and resumes for TF-IDF vectorization
    documents = [processed_jd] + processed_resumes

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)

    # Cosine similarity between JD and each resume
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    # Rank resumes based on similarity
    ranked_resumes = sorted(zip(resumes, similarities), key=lambda x: x[1], reverse=True)

    return ranked_resumes

# ---------------------------
# Streamlit App UI
# ---------------------------

st.title("AI Resume Ranker 🚀")
st.write("Upload resumes and a job description, and get the top matching resumes!")

# Upload job description
job_description = st.text_area("Paste the Job Description here:", height=200)

# Upload multiple resumes as text files
uploaded_files = st.file_uploader("Upload Resumes (TXT format, multiple allowed):", type=["txt"], accept_multiple_files=True)

if st.button("Rank Resumes"):
    if not job_description:
        st.error("Please provide a Job Description.")
    elif not uploaded_files:
        st.error("Please upload at least one resume in TXT format.")
    else:
        # Read content from uploaded files
        resumes = [file.read().decode("utf-8") for file in uploaded_files]

        # Rank resumes
        ranked_results = rank_resumes(job_description, resumes)

        # Display results
        st.subheader("Top Matching Resumes:")
        for idx, (resume, score) in enumerate(ranked_results, 1):
            st.markdown(f"### Rank {idx}: (Similarity Score: {score:.2f})")
            st.text_area("Resume Content", resume, height=200)

st.markdown("---")
st.markdown("💡 Make sure resumes are in plain text format for better matching accuracy.")
