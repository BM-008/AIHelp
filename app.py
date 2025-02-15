import streamlit as st
from transformers import pipeline
import PyPDF2
import docx
import textwrap

# Streamlit Page Config
st.set_page_config(
    page_title="TextSphere",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Footer
st.markdown("""
    <style>
        .footer {
            position: fixed;
            bottom: 0;
            right: 0;
            padding: 10px;
            font-size: 16px;
            color: #333;
            background-color: #f1f1f1;
        }
    </style>
    <div class="footer">
        Made with ❤️ by Baibhav Malviya
    </div>
""", unsafe_allow_html=True)

# Load Model
@st.cache_resource
def load_models():
    try:
        summarization_model = pipeline("summarization", model="facebook/bart-large-cnn")
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")
    return summarization_model

summarization_model = load_models()

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

# Sidebar for Task Selection (Default: Text Summarization)
st.sidebar.title("AI Solutions")
option = st.sidebar.selectbox(
    "Choose a task",
    ["Text Summarization", "Question Answering", "Text Classification", "Language Translation"],
    index=0  # Default to "Text Summarization"
)

# Text Summarization Task
if option == "Text Summarization":
    st.title("📄 Text Summarization")
    st.markdown("<h4 style='font-size: 20px;'>- because who needs to read the whole document? 🥵</h4>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload a document (PDF, TXT, DOCX) - *Note: Processes only 1024 tokens per chunk*", 
        type=["pdf", "txt", "docx"]
    )

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
                    summaries = []
                    
                    for chunk in chunks:
                        input_length = len(chunk.split())  # Count words in the chunk
                        max_summary_length = max(50, input_length // 2)  # Dynamically adjust max_length
                        
                        summary = summarization_model(chunk, max_length=max_summary_length, min_length=50, do_sample=False)
                        summaries.append(summary[0]['summary_text'])
                    
                    final_summary = " ".join(summaries)  # Combine all chunk summaries

                    st.write("### Summary:")
                    st.write(final_summary)
                else:
                    st.error("Please upload a document first.")
            except Exception as e:
                st.error(f"Error: {e}")




