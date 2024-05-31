import os
import textwrap
import google.generativeai as genai
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
job_des=pd.read_csv('resume.csv')

# Set your Google API key
os.environ['GOOGLE_API_KEY'] = 'AIzaSyB_1OPgHXgBcZ1pTFytOqmxQ-nCrcdfQyM'

# Configure the generative AI module
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])  

# List available models
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(m.name) 

# Initialize the generative model (assuming 'gemini-pro' is one of the available models)
model = genai.GenerativeModel('gemini-pro')

# Import libraries
import fitz  # PyMuPDF
from docx import Document

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

# Path to PDF and DOCX files (change as per your file paths)
pdf_file_path = "software-developer-1562751287.pdf"
docx_file_path = ""
data=""
if pdf_file_path!="":
   pdf_text = extract_text_from_pdf(pdf_file_path)
   print("Text from PDF:")
   data+=pdf_text

# Extract text from DOCX
if docx_file_path!="":
    docx_text = extract_text_from_docx(docx_file_path)
    print("Text from DOCX:")
    data+=docx_text

# Combine text from PDF and DOCX

def extract_skills_section(data):
    prompt = f"Extract the skills section from the following text and provide it in a list format: {data}"
    response = model.generate_content(prompt)
    # Assuming the response object has an attribute `text` or a similar method to access the generated content
    return response.text if hasattr(response, 'text') else response.generate()  # Adjust according to actual method


# Extract skills section
skills_text = extract_skills_section(data)
print("Extracted Skills Section Text:", skills_text)

# Assuming the model generates a list format string, process it into a Python list
skills_list = skills_text.strip().split('\n')  # Adjust parsing logic as needed
print("Skills List:", skills_list)
s=""
for i in skills_list:
  s+=" "+i
  num_rows = len(job_des)
cv = CountVectorizer()
for i in range(num_rows):
    job_title1 = str(job_des['JobTitle'].iloc[i])
    job_title2 = str(job_des['Summary'].iloc[i])
    job_title = ""+job_title1 + " " + job_title2
    content = [job_title, s]
    count_matrix = cv.fit_transform(content)
    cosine_sim = cosine_similarity(count_matrix)
    #print(cosine_sim[0][1])

    if cosine_sim[0][1] > 0.05:
        print("Role:", job_title, "\nCompany:", job_des['Company'].iloc[i], "\nURL:", job_des['JobUrl'].iloc[i], "\n")