import os
from flask import Flask, render_template, request
import fitz  # PyMuPDF
from docx import Document
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai

app = Flask(__name__)

# Ensure your API key is set correctly
genai.configure(api_key='AIzaSyB_1OPgHXgBcZ1pTFytOqmxQ-nCrcdfQyM')

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    text = ""
    for page_num in range(document.page_count):
        page = document[page_num]
        text += page.get_text()
    return text

# Function to extract text from DOCX
def extract_text_from_docx(docx_path):
    document = Document(docx_path)
    text = "\n".join([paragraph.text for paragraph in document.paragraphs])
    return text

# Ensure 'uploads' directory exists
if not os.path.exists('uploads'):
    os.makedirs('uploads')

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze_resume():
    resume_file = request.files["resume"]
    if resume_file:
        resume_filename = resume_file.filename
        resume_extension = os.path.splitext(resume_filename)[1].lower()
        if resume_extension in [".pdf", ".docx"]:
            resume_path = os.path.join("uploads", resume_filename)
            resume_file.save(resume_path)
            
            # Extract text from resume
            if resume_extension == ".pdf":
                resume_text = extract_text_from_pdf(resume_path)
            else:
                resume_text = extract_text_from_docx(resume_path)
            
            # Extract skills section using generative AI model
            skills_text = extract_skills_section(resume_text)

            # Process skills section and find matching job descriptions
            result = find_matching_jobs(skills_text)

            return render_template("result.html", skills_text=skills_text, result=result)
        else:
            return "Unsupported file format. Please upload a PDF or DOCX file."
    else:
        return "No file uploaded."

def extract_skills_section(resume_text):
    prompt = f"Extract the skills section from the following resume text:\n\n{resume_text}"
    response = genai.generate_text(prompt=prompt, temperature=0.5)
    
    # Assuming the response has a 'generations' attribute which is a list of generated texts
    if hasattr(response, 'generations') and len(response.generations) > 0:
        skills_text = response.generations[0].text  # Adjust based on the actual response structure
    else:
        skills_text = "No skills section found in the resume text."
    
    return skills_text

def find_matching_jobs(skills_text):
    # Load job descriptions CSV (job_des)
    job_des = pd.read_csv('resume.csv')
    
    # Process skills_text and find matching job descriptions
    cv = CountVectorizer()

    result = {}
    for i in range(len(job_des)):
        job_title1 = str(job_des['JobTitle'].iloc[i])
        job_title2 = str(job_des['Summary'].iloc[i])
        job_title = job_title1 + " " + job_title2
        content = [job_title, skills_text]
        count_matrix = cv.fit_transform(content)
        cosine_sim = cosine_similarity(count_matrix)

        if cosine_sim[0][1] > 0.05:
            result[i] = {
                "Role": job_title,
                "Company": job_des['Company'].iloc[i],
                "URL": job_des['JobUrl'].iloc[i]
            }

    return result

if __name__ == "__main__":
    app.run(debug=True)
