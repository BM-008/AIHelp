import streamlit as st
from transformers import pipeline
import PyPDF2
import docx
from io import BytesIO

st.set_page_config(
    page_title="TextSphere",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

@st.cache_resource
def load_models():
    try:
        text_classification_model = pipeline(
            "text-classification",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )

        question_answering_model = pipeline(
            "question-answering",
            model="distilbert-base-uncased-distilled-squad"
        )

        translation_model = pipeline(
            "translation",
            model="Helsinki-NLP/opus-mt-en-fr"
        )

        summarization_model = pipeline(
            "summarization",
            model="facebook/bart-large-cnn"
        )

    except Exception as e:
        raise RuntimeError(f"Failed to load models: {str(e)}")

    return text_classification_model, question_answering_model, translation_model, summarization_model

def extract_text_from_pdf(uploaded_file):
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text.strip()
    except Exception as e:
        st.error(f"Error reading the PDF: {e}")
        return None

def extract_text_from_docx(uploaded_file):
    try:
        doc = docx.Document(uploaded_file)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        st.error(f"Error reading the DOCX: {e}")
        return None

def extract_text_from_txt(uploaded_file):
    try:
        return uploaded_file.read().decode("utf-8")
    except Exception as e:
        st.error(f"Error reading the TXT file: {e}")
        return None

def extract_text_from_file(uploaded_file, file_type):
    if file_type == "pdf":
        return extract_text_from_pdf(uploaded_file)
    elif file_type == "docx":
        return extract_text_from_docx(uploaded_file)
    elif file_type == "txt":
        return extract_text_from_txt(uploaded_file)
    return None

try:
    classification_model, qa_model, translation_model, summarization_model = load_models()
except Exception as e:
    st.error(f"An error occurred while loading models: {e}")

st.sidebar.title("AI Solutions")
option = st.sidebar.selectbox(
    "Choose a task",
    ["Text Summarization", "Question Answering", "Text Classification", "Language Translation"],
    index=0
)

if option == "Text Summarization":
    st.title("Text Summarization")
    st.markdown("<h4 style='font-size: 20px;'>- because who needs to read the whole document, anyway? 🥵</h4>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload a document (PDF, DOCX, TXT) [Limit: 1024 Tokens]", type=["pdf", "docx", "txt"])
    
    text_to_summarize = st.text_area("Enter text to summarize (or leave empty if uploading a file):")

    if uploaded_file:
        file_type = uploaded_file.name.split(".")[-1].lower()
        text_to_summarize = extract_text_from_file(uploaded_file, file_type)

    if st.button("Summarize"):
        with st.spinner('Summarizing text...'):
            try:
                if text_to_summarize:
                    summary = summarization_model(text_to_summarize[:1024], max_length=300, min_length=50, do_sample=False)
                    st.write("Summary:", summary[0]['summary_text'])
                    st.balloons()
                else:
                    st.error("Please enter text or upload a document for summarization.")
            except Exception as e:
                st.error(f"An error occurred: {e}")

elif option == "Question Answering":
    st.title("Question Answering")
    st.markdown("<h4 style='font-size: 20px;'>- because Google wasn't enough 😉</h4>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload a document (PDF, DOCX, TXT) for context (optional)", type=["pdf", "docx", "txt"])
    
    context_input = st.text_area("Enter context (or leave empty if uploading a file):")
    question = st.text_input("Enter your question:")

    if uploaded_file:
        file_type = uploaded_file.name.split(".")[-1].lower()
        context_input = extract_text_from_file(uploaded_file, file_type)

    if st.button("Get Answer"):
        with st.spinner('Finding answer...'):
            try:
                if context_input and question:
                    answer = qa_model(question=question, context=context_input)
                    st.write("Answer:", answer['answer'])
                    st.balloons()
                else:
                    st.error("Please enter both context and a question.")
            except Exception as e:
                st.error(f"An error occurred: {e}")

elif option == "Text Classification":
    st.title("Text Classification")
    st.markdown("<h4 style='font-size: 20px;'>- where machines learn to hate spam as much as we do 😅</h4>", unsafe_allow_html=True)
    
    text = st.text_area("Enter text for classification:")
    
    if st.button("Classify Text"):
        with st.spinner('Classifying text...'):
            try:
                classification = classification_model(text)
                st.json(classification)
                st.balloons()
            except Exception as e:
                st.error(f"An error occurred: {e}")

elif option == "Language Translation":
    st.title("Language Translation (English to Multiple Languages)")
    st.markdown("<h4 style='font-size: 20px;'>- when 'translate' is the only button you know 😁</h4>", unsafe_allow_html=True)
    
    target_language = st.selectbox("Choose target language", ["French", "Spanish", "German", "Italian", "Portuguese", "Hindi"])
    
    language_models = {
        "French": "Helsinki-NLP/opus-mt-en-fr",
        "Spanish": "Helsinki-NLP/opus-mt-en-es",
        "German": "Helsinki-NLP/opus-mt-en-de",
        "Italian": "Helsinki-NLP/opus-mt-en-it",
        "Portuguese": "Helsinki-NLP/opus-mt-en-pt",
        "Hindi": "Helsinki-NLP/opus-mt-en-hi"
    }

    selected_model = language_models.get(target_language)
    translation_pipeline = pipeline("translation", model=selected_model)

    text_to_translate = st.text_area(f"Enter text to translate from English to {target_language}:")
    
    if st.button("Translate"):
        with st.spinner('Translating...'):
            try:
                if text_to_translate:
                    translated_text = translation_pipeline(text_to_translate)
                    st.write(f"Translated Text ({target_language}):", translated_text[0]['translation_text'])
                    st.balloons()
                else:
                    st.error("Please enter text to translate.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
