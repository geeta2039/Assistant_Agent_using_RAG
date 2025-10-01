# import os
# import numpy as np
# from langchain.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings
# from transformers import pipeline
# from config import DEFAULT_EMBEDDING, GENERATION_MODELS


# # In-memory storage
# chunks = []
# chunk_embeddings = []

# # Embedding model
# embedder = HuggingFaceEmbeddings(model_name=DEFAULT_EMBEDDING)

# # -------------------------
# # Load and chunk documents
# # -------------------------
# def add_document_to_chunks(file_path):
#     """Load a document, split into chunks, and store embeddings."""
#     # Pick loader based on file type
#     if file_path.endswith(".txt"):
#         loader = TextLoader(file_path)
#     elif file_path.endswith(".pdf"):
#         loader = PyPDFLoader(file_path)
#     elif file_path.endswith(".docx"):
#         loader = Docx2txtLoader(file_path)
#     else:
#         raise ValueError(f"Unsupported file format: {file_path}")

#     docs = loader.load()
#     splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#     new_chunks = splitter.split_documents(docs)

#     # Store chunks
#     chunks.extend(new_chunks)

#     # Compute embeddings
#     new_embeddings = embedder.embed_documents([c.page_content for c in new_chunks])
#     chunk_embeddings.extend(new_embeddings)

# # -------------------------
# # Cosine similarity
# # -------------------------
# def cosine_similarity(vec1, vec2):
#     return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# # -------------------------
# # Retrieve top-k relevant chunks
# # -------------------------
# def retrieve_relevant_chunks(query, top_k=3):
#     """Retrieve most relevant document chunks for a query."""
#     if not chunks:
#         return []

#     query_emb = embedder.embed_query(query)
#     similarities = [cosine_similarity(query_emb, emb) for emb in chunk_embeddings]
#     sorted_indices = np.argsort(similarities)[::-1]
#     return [chunks[i] for i in sorted_indices[:top_k]]

# # -------------------------
# # Answer Question
# # -------------------------
# generator_cache = {}

# def answer_question(query, relevant_chunks, model="flan-base"):
#     """Generate an answer from retrieved chunks using chosen model."""
#     if model not in GENERATION_MODELS:
#         raise ValueError(f"Model '{model}' not supported. Choose from: {list(GENERATION_MODELS.keys())}")

#     model_name = GENERATION_MODELS[model]

#     # Initialize generator once
#     if model not in generator_cache:
#         task = "text2text-generation" if "flan" in model else "text-generation"
#         generator_cache[model] = pipeline(task, model=model_name, device=-1)

#     generator = generator_cache[model]

#     context = "\n".join([doc.page_content for doc in relevant_chunks])
#     prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

#     response = generator(prompt, max_new_tokens=150, do_sample=False, truncation=True)
#     return response[0]['generated_text']
import os
import numpy as np
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import pipeline

from config import DEFAULT_EMBEDDING, GENERATION_MODELS

# -------------------------
# In-memory storage
# -------------------------
chunks = []
chunk_embeddings = []

# Embedding model
embedder = HuggingFaceEmbeddings(model_name=DEFAULT_EMBEDDING)

# -------------------------
# Load all documents from folder
# -------------------------
def load_documents_from_folder(folder_path="docs"):
    if not os.path.exists(folder_path):
        print(f"❌ Folder '{folder_path}' not found. Create it and add files.")
        return

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Pick loader based on file extension
        if filename.endswith(".txt"):
            loader = TextLoader(file_path)
        elif filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif filename.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        else:
            print(f"⚠️ Skipping unsupported file: {filename}")
            continue

        # Load and split docs
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        new_chunks = splitter.split_documents(docs)

        # Store chunks + embeddings
        chunks.extend(new_chunks)
        new_embeddings = embedder.embed_documents([doc.page_content for doc in new_chunks])
        chunk_embeddings.extend(new_embeddings)

        print(f"✅ Loaded {len(new_chunks)} chunks from {filename}")

# -------------------------
# Cosine similarity
# -------------------------
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# -------------------------
# Retrieve top-k relevant chunks
# -------------------------
def retrieve_relevant_chunks(query, top_k=3):
    if not chunks:
        print("⚠️ No documents loaded yet.")
        return []

    query_emb = embedder.embed_query(query)
    similarities = [cosine_similarity(query_emb, emb) for emb in chunk_embeddings]
    sorted_indices = np.argsort(similarities)[::-1]

    return [chunks[i] for i in sorted_indices[:top_k]]

# -------------------------
# Answer Question
# -------------------------
generator_cache = {}

def answer_question(query, relevant_chunks, model="flan-base"):
    if model not in GENERATION_MODELS:
        raise ValueError(f"❌ Model '{model}' not supported. Choose from: {list(GENERATION_MODELS.keys())}")

    model_name = GENERATION_MODELS[model]

    # Initialize model only once
    if model not in generator_cache:
        task = "text2text-generation" if "flan" in model else "text-generation"
        generator_cache[model] = pipeline(task, model=model_name, device=-1)

    generator = generator_cache[model]

    # Build prompt with context
    context = "\n".join([doc.page_content for doc in relevant_chunks])
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

    response = generator(prompt, max_new_tokens=150, do_sample=False, truncation=True)
    return response[0]['generated_text']

# -------------------------
# Load documents at startup
# -------------------------
load_documents_from_folder("docs")
