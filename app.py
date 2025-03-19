import streamlit as st
import spacy
import fitz  # PyMuPDF for PDF processing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.error("SpaCy model not found. Please install with `python -m spacy download en_core_web_sm`.")

# Helper function to extract text from PDF
def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Preprocess text
def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and not token.like_num]
    return " ".join(tokens)

# Rank resumes based on job description
def rank_resumes(job_description, resumes):
    processed_jd = preprocess_text(job_description)
    processed_resumes = [preprocess_text(resume) for resume in resumes]

    documents = [processed_jd] + processed_resumes
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    ranked_resumes = sorted(zip(resumes, similarities), key=lambda x: x[1], reverse=True)
    return ranked_resumes

# Streamlit UI
st.title("AI Resume Screening")
st.write("Upload resumes and provide job description to rank candidates.")

job_description = st.text_area("Job Description", height=150)
uploaded_files = st.file_uploader("Upload Resumes (PDF format):", type=["pdf"], accept_multiple_files=True)

if st.button("Rank Resumes"):
    if not job_description:
        st.error("Please add a job description.")
    elif not uploaded_files:
        st.error("Please upload at least one resume.")
    else:
        resumes = [extract_text_from_pdf(file) for file in uploaded_files]
        ranked = rank_resumes(job_description, resumes)

        st.subheader("Top Matches:")
        for idx, (resume, score) in enumerate(ranked, 1):
            st.markdown(f"**Rank {idx}** (Score: {score:.2f})")
