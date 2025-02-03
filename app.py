import streamlit as st
import Update_tracking
import legal_document_analysis
from rag_pipeline import extract_text_from_pdf, create_vector_store, create_qa_pipeline

# Streamlit App Navigation
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Choose a page", ["Update Tracking", "Legal Document Analysis"])

    if page == "Update Tracking":
        Update_tracking.display_tracking_status()  # Ensure the correct function name
    elif page == "Legal Document Analysis":
        legal_document_analysis.display_legal_analysis_page()

if __name__ == "__main__":
    main()
