{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "5d44f32c-0aaa-4699-83a0-9e5ca24c9537",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "SpaCy loaded successfully!\n"
     ]
    }
   ],
   "source": [
    "import spacy\n",
    "nlp = spacy.load(\"en_core_web_sm\")\n",
    "print(\"SpaCy loaded successfully!\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "cf910829-7fa6-4d7f-b3f6-e722f0d9caa8",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-03-17 00:03:13.809 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-17 00:03:14.759 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\hp\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n",
      "2025-03-17 00:03:14.761 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-17 00:03:14.761 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-17 00:03:14.762 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-17 00:03:14.764 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-17 00:03:14.766 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-17 00:03:14.767 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-17 00:03:14.768 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-17 00:03:14.769 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-17 00:03:14.772 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-17 00:03:14.773 Session state does not function when running a script without `streamlit run`\n",
      "2025-03-17 00:03:14.775 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-17 00:03:14.777 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-17 00:03:14.779 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-17 00:03:14.781 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-17 00:03:14.783 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-17 00:03:14.785 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-17 00:03:14.787 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-17 00:03:14.790 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-17 00:03:14.791 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-17 00:03:14.792 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-17 00:03:14.793 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-17 00:03:14.794 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-17 00:03:14.795 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-17 00:03:14.796 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    }
   ],
   "source": [
    "import streamlit as st\n",
    "import spacy\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.metrics.pairwise import cosine_similarity\n",
    "\n",
    "# Load the SpaCy model\n",
    "nlp = spacy.load(\"en_core_web_sm\")\n",
    "\n",
    "# Title of the app\n",
    "st.title(\"AI-Powered Resume Ranking App 🚀\")\n",
    "st.write(\"Upload resumes and enter a job description to rank the best candidates!\")\n",
    "\n",
    "# Job description input\n",
    "job_description = st.text_area(\"Enter Job Description:\", height=200)\n",
    "\n",
    "# Upload resumes\n",
    "uploaded_files = st.file_uploader(\"Upload Resumes (txt files)\", type=[\"txt\"], accept_multiple_files=True)\n",
    "\n",
    "# Function to clean and preprocess text\n",
    "def preprocess_text(text):\n",
    "    doc = nlp(text.lower())\n",
    "    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and not token.like_num]\n",
    "    return \" \".join(tokens)\n",
    "\n",
    "# Button to trigger ranking\n",
    "if st.button(\"Rank Resumes\") and job_description and uploaded_files:\n",
    "    resumes = []\n",
    "    resume_names = []\n",
    "\n",
    "    # Read and preprocess resumes\n",
    "    for uploaded_file in uploaded_files:\n",
    "        text = uploaded_file.read().decode(\"utf-8\")\n",
    "        cleaned_text = preprocess_text(text)\n",
    "        resumes.append(cleaned_text)\n",
    "        resume_names.append(uploaded_file.name)\n",
    "\n",
    "    # Preprocess job description\n",
    "    cleaned_jd = preprocess_text(job_description)\n",
    "\n",
    "    # Combine job description with resumes for vectorization\n",
    "    documents = [cleaned_jd] + resumes\n",
    "\n",
    "    # TF-IDF Vectorization\n",
    "    vectorizer = TfidfVectorizer()\n",
    "    tfidf_matrix = vectorizer.fit_transform(documents)\n",
    "\n",
    "    # Compute similarity\n",
    "    similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()\n",
    "\n",
    "    # Rank resumes\n",
    "    ranked_indices = similarity_scores.argsort()[::-1]\n",
    "\n",
    "    # Display ranked resumes\n",
    "    st.subheader(\"Ranked Resumes (Best Match First):\")\n",
    "    for idx in ranked_indices:\n",
    "        st.write(f\"**{resume_names[idx]}** — Similarity Score: {similarity_scores[idx]:.4f}\")\n",
    "else:\n",
    "    st.info(\"Please enter a job description and upload at least one resume to start ranking.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "944cbfc2-25d7-4e34-a0fc-0d142adab6fd",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
