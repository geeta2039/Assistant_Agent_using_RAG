\# Assistant Agent using RAG (Retrieval-Augmented Generation)



This project is an AI assistant that can answer questions based on uploaded employee guidelines or other documents using \*\*RAG (Retrieval-Augmented Generation)\*\*.



It uses:

\- \*\*LangChain\*\* for document processing and embeddings.

\- \*\*HuggingFace models\*\* for embeddings and text generation.

\- \*\*Python\*\* with FastAPI for serving the assistant.



---



\## Features



\- Upload documents (`.txt`, `.pdf`, `.docx`) locally.

\- Automatically split documents into chunks and generate embeddings.

\- Retrieve relevant chunks using semantic similarity.

\- Generate answers using HuggingFace text-generation models.

\- Completely local, no need for uploading data via web browser.



---



\## Getting Started



1\. \*\*Clone the repository\*\*



```bash

git clone https://github.com/username/Assistant\_Agent\_using\_RAG.git

cd Assistant\_Agent\_using\_RAG



