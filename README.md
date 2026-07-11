# AI Research Assistant

A Streamlit app that lets you upload a PDF, extract its content, and ask questions about the document using Gemini and vector search.

## Features
- Upload PDF documents
- Extract and chunk text from the file
- Store document chunks in ChromaDB
- Ask questions grounded in the uploaded content

## Setup
1. Create and activate a virtual environment
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your Google API key:
   ```bash
   GOOGLE_API_KEY=your_google_api_key_here
   ```
4. Run the app:
   ```bash
   streamlit run app.py
   ```

## GitHub deployment
1. Initialize a Git repository if needed:
   ```bash
   git init
   ```
2. Add and commit your files:
   ```bash
   git add .
   git commit -m "Initial commit"
   ```
3. Create a repository on GitHub and push:
   ```bash
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo>.git
   git push -u origin main
   ```
