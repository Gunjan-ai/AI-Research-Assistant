import os
import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import chromadb

load_dotenv()


def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    if chunk_size <= 0:
        return []

    safe_overlap = min(max(chunk_overlap, 0), max(chunk_size - 1, 0))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=safe_overlap,
    )
    return splitter.split_text(text)


def rank_chunks(question, chunks):
    if not chunks:
        return []

    question_terms = [term for term in question.lower().split() if term.isalnum()]
    scored_chunks = []

    for chunk in chunks:
        chunk_text = chunk.lower()
        score = sum(1 for term in question_terms if term in chunk_text)
        scored_chunks.append((score, chunk))

    scored_chunks.sort(key=lambda item: (-item[0], item[1]))
    return [chunk for _, chunk in scored_chunks]


def configure_gemini():
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if google_api_key:
        genai.configure(api_key=google_api_key)
    return google_api_key


def render_app():
    configure_gemini()

    st.set_page_config(
        page_title="AI Research Assistant",
        page_icon="📄",
        layout="wide",
    )

    with st.sidebar:
        st.title("📄 AI Research Assistant")
        st.write("Upload any PDF and ask questions.")
        st.markdown("---")
        st.write("Powered by")
        st.write("- Gemini")
        st.write("- ChromaDB")
        st.write("- Sentence Transformers")

    st.title("📄 AI Research Assistant")
    st.write("Upload a PDF and ask questions about it.")

    uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])
    question = st.text_input("Ask a question about the document")

    if st.button("Ask"):
        if uploaded_file is None:
            st.warning("Please upload a PDF first.")
        elif question == "":
            st.warning("Please enter a question.")
        else:
            with st.spinner("Processing PDF and generating embeddings..."):
                reader = PdfReader(uploaded_file)
                text = ""

                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text

                st.success("PDF successfully loaded!")
                st.subheader("Extracted Text")

                chunks = chunk_text(text)
                st.success(f"Successfully created {len(chunks)} chunks!")


                @st.cache_resource
                def load_embedding_model():
                    return SentenceTransformer("all-MiniLM-L6-v2")

                #embedding_model = load_embedding_model()
                embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
                embeddings = embedding_model.encode(chunks).tolist()

                client = chromadb.Client()
                try:
                    client.delete_collection("research_papers")
                except Exception:
                    pass

                collection = client.create_collection(name="research_papers")
                collection.upsert(
                    documents=chunks,
                    embeddings=embeddings,
                    ids=[str(i) for i in range(len(chunks))],
                )

            st.success(f"✅ Stored {len(chunks)} chunks in ChromaDB.")
            st.success("✅ PDF processed successfully!")

            ranked_chunks = rank_chunks(question, chunks)
            context = "\n\n".join(ranked_chunks[:3])
            prompt = f"""
            You are an AI Research Assistant.

            Your job is to answer questions using ONLY the provided context.

            Rules:
            1. Answer only from the context.
            2. Do not use your own knowledge.
            3. Keep the answer concise (1-3 sentences).
            4. If the answer is not present in the context, reply exactly:
            "I couldn't find that information in the document."
            5. Do not summarize the entire document unless the user specifically asks for a summary.

            Context:
            {context}

            Question:
            {question}

            Answer:
            """

            if not os.getenv("GOOGLE_API_KEY"):
                st.warning("Set GOOGLE_API_KEY in your environment to enable Gemini answers.")
                return

            model = genai.GenerativeModel("gemini-2.5-flash")
            with st.spinner("Searching the document..."):
                response = model.generate_content(prompt)

            st.subheader("🤖 Answer")
            st.write(response.text)
            #response = model.generate_content(prompt)
           # st.subheader("🤖 Answer")
            #st.write(response.text)

    st.markdown("---")
    st.caption("Built by Gunjan Gupta ❤️")


if __name__ == "__main__":
    render_app()