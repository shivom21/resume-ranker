import streamlit as st
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the SpaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    st.error("SpaCy model 'en_core_web_sm' not found. Please ensure it is installed.")

# Title of the app
st.title("AI-Powered Resume Ranking App 🚀")
st.write("Upload resumes and enter a job description to rank the best candidates!")

# Job description input
job_description = st.text_area("Enter Job Description:", height=200)

# Upload resumes
uploaded_files = st.file_uploader("Upload Resumes (txt files)", type=["txt"], accept_multiple_files=True)

# Function to clean and preprocess text
def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and not token.like_num]
    return " ".join(tokens)

# Button to trigger ranking
if st.button("Rank Resumes") and job_description and uploaded_files:
    resumes = []
    resume_names = []

    # Read and preprocess resumes
    for uploaded_file in uploaded_files:
        text = uploaded_file.read().decode("utf-8")
        cleaned_text = preprocess_text(text)
        resumes.append(cleaned_text)
        resume_names.append(uploaded_file.name)

    # Preprocess job description
    cleaned_jd = preprocess_text(job_description)

    # Combine job description with resumes for vectorization
    documents = [cleaned_jd] + resumes

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)

    # Compute similarity
    similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    # Rank resumes
    ranked_indices = similarity_scores.argsort()[::-1]

    # Display ranked resumes
    st.subheader("Ranked Resumes (Best Match First):")
    for idx in ranked_indices:
        st.write(f"**{resume_names[idx]}** — Similarity Score: {similarity_scores[idx]:.4f}")

else:
    st.info("Please enter a job description and upload at least one resume to start ranking.")
=======
import streamlit as st
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the SpaCy model
nlp = spacy.load("en_core_web_sm")

# Title of the app
st.title("AI-Powered Resume Ranking App 🚀")
st.write("Upload resumes and enter a job description to rank the best candidates!")

# Job description input
job_description = st.text_area("Enter Job Description:", height=200)

# Upload resumes
uploaded_files = st.file_uploader("Upload Resumes (txt files)", type=["txt"], accept_multiple_files=True)

# Function to clean and preprocess text
def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and not token.like_num]
    return " ".join(tokens)

# Button to trigger ranking
if st.button("Rank Resumes") and job_description and uploaded_files:
    resumes = []
    resume_names = []

    # Read and preprocess resumes
    for uploaded_file in uploaded_files:
        text = uploaded_file.read().decode("utf-8")
        cleaned_text = preprocess_text(text)
        resumes.append(cleaned_text)
        resume_names.append(uploaded_file.name)

    # Preprocess job description
    cleaned_jd = preprocess_text(job_description)

    # Combine job description with resumes for vectorization
    documents = [cleaned_jd] + resumes

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)

    # Compute similarity
    similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    # Rank resumes
    ranked_indices = similarity_scores.argsort()[::-1]

    # Display ranked resumes
    st.subheader("Ranked Resumes (Best Match First):")
    for idx in ranked_indices:
        st.write(f"**{resume_names[idx]}** — Similarity Score: {similarity_scores[idx]:.4f}")
else:
    st.info("Please enter a job description and upload at least one resume to start ranking.")
