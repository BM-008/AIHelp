import streamlit as st
from transformers import pipeline
import PyPDF2
import docx
from io import BytesIO

# Streamlit Page Config
st.set_page_config(
    page_title="Text Summarization",
    page_icon="📄",
    layout="wide"
)

@st.cache_resource
def load_summarization_model():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarization_model = load_summarization_model()

def extract_text_from_pdf(uploaded_pdf):
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_pdf)
        pdf_text = "".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
        return pdf_text.strip() if pdf_text else None
    except Exception as e:
        st.error(f"Error reading the PDF: {e}")
        return None

def extract_text_from_txt(uploaded_txt):
    try:
        return uploaded_txt.read().decode("utf-8").strip()
    except Exception as e:
        st.error(f"Error reading the TXT file: {e}")
        return None

def extract_text_from_docx(uploaded_docx):
    try:
        doc = docx.Document(uploaded_docx)
        return "\n".join([para.text for para in doc.paragraphs]).strip()
    except Exception as e:
        st.error(f"Error reading the DOCX file: {e}")
        return None

st.title("Text Summarization")
st.markdown("Upload a document (PDF, TXT, DOCX) to generate a summary.")

uploaded_file = st.file_uploader("Upload File", type=["pdf", "txt", "docx"])
text_to_summarize = ""

if uploaded_file:
    file_extension = uploaded_file.name.split(".")[-1].lower()
    if file_extension == "pdf":
        text_to_summarize = extract_text_from_pdf(uploaded_file)
    elif file_extension == "txt":
        text_to_summarize = extract_text_from_txt(uploaded_file)
    elif file_extension == "docx":
        text_to_summarize = extract_text_from_docx(uploaded_file)
    else:
        st.error("Unsupported file format. Please upload a PDF, TXT, or DOCX file.")

if st.button("Summarize"):
    if text_to_summarize:
        if len(text_to_summarize.split()) > 1024:
            text_to_summarize = " ".join(text_to_summarize.split()[:1024])  # Trim text
        with st.spinner("Summarizing text..."):
            try:
                summary = summarization_model(text_to_summarize, max_length=130, min_length=30, do_sample=False)
                st.write("### Summary:")
                st.write(summary[0]['summary_text'])
                st.balloons()
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.error("No valid text found. Please enter text or upload a document.")
