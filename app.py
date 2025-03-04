import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader  
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to extract text from an uploaded PDF file
def extract_text_from_pdf(uploaded_file):
    text = ''
    pdf = PdfReader(uploaded_file)  # Read directly from the uploaded file
    for page in pdf.pages:
        text += page.extract_text() if page.extract_text() else ''  # Handle None case
    return text

# Function to rank resumes based on job description
def rank_resumes(job_description, resumes):
    # Combine job description and resumes
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()

    # Calculate cosine similarity
    job_description_vector = vectors[0]
    resume_vectors = vectors[1:]
    cosine_similarities = cosine_similarity([job_description_vector], resume_vectors).flatten()

    return cosine_similarities

# Streamlit app
st.title('Resume Ranking App')

# Job description input
job_description = st.text_area('Enter job description')

# Upload resumes
st.header('Upload Resumes')
uploaded_files = st.file_uploader("Upload your resumes", type=['pdf'], accept_multiple_files=True)

if uploaded_files and job_description:
    st.header('Ranking Resumes')

    resumes = []
    for file in uploaded_files:
        text = extract_text_from_pdf(file)
        resumes.append(text)

    # Rank resumes
    scores = rank_resumes(job_description, resumes)
    results = pd.DataFrame({'Resume': [file.name for file in uploaded_files], 'Score': scores})
    results = results.sort_values(by='Score', ascending=False)

    st.write(results)
