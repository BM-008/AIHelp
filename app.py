import streamlit as st
from transformers import pipeline
import PyPDF2
import docx
import textwrap

# Streamlit Page Config
st.set_page_config(
    page_title="📄 Text Summarization",
    page_icon="📑",
    layout="wide"
)

@st.cache_resource
def load_summarization_model():
    try:
        summarization_model = pipeline("summarization", model="facebook/bart-large-cnn")
        return summarization_model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

summarization_model = load_summarization_model()

# Function to Extract Text from PDF
def extract_text_from_pdf(uploaded_pdf):
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_pdf)
        pdf_text = ""
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                pdf_text += text + "\n"
        if not pdf_text.strip():
            st.error("No text found in the PDF.")
            return None
        return pdf_text
    except Exception as e:
        st.error(f"Error reading the PDF: {e}")
        return None

# Function to Extract Text from TXT
def extract_text_from_txt(uploaded_txt):
    try:
        return uploaded_txt.read().decode("utf-8").strip()
    except Exception as e:
        st.error(f"Error reading the TXT file: {e}")
        return None

# Function to Extract Text from DOCX
def extract_text_from_docx(uploaded_docx):
    try:
        doc = docx.Document(uploaded_docx)
        return "\n".join([para.text for para in doc.paragraphs]).strip()
    except Exception as e:
        st.error(f"Error reading the DOCX file: {e}")
        return None

# Function to Split Text into 1024-Token Chunks
def chunk_text(text, max_tokens=1024):
    return textwrap.wrap(text, width=max_tokens)

# Streamlit UI for Summarization
st.title("📄 Text Summarization")
uploaded_file = st.file_uploader("Upload a document (PDF, TXT, DOCX)", type=["pdf", "txt", "docx"])

text_to_summarize = ""

if uploaded_file:
    file_type = uploaded_file.name.split(".")[-1].lower()

    if file_type == "pdf":
        text_to_summarize = extract_text_from_pdf(uploaded_file)
    elif file_type == "txt":
        text_to_summarize = extract_text_from_txt(uploaded_file)
    elif file_type == "docx":
        text_to_summarize = extract_text_from_docx(uploaded_file)
    else:
        st.error("Unsupported file format.")

if st.button("Summarize"):
    with st.spinner('Summarizing...'):
        try:
            if text_to_summarize:
                chunks = chunk_text(text_to_summarize, max_tokens=1024)
                summaries = [summarization_model(chunk, max_length=300, min_length=100, do_sample=False)[0]['summary_text'] for chunk in chunks]

                final_summary = " ".join(summaries)  # Combine all chunk summaries

                st.write("### Summary:")
                st.write(final_summary)
            else:
                st.error("Please upload a document first.")
        except Exception as e:
            st.error(f"Error: {e}")

