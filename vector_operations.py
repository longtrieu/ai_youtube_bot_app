from langchain_community.vectorstores import FAISS

def create_faiss_index(chunks, embedding_model):
  """Create a FAISS index from text chunks using the specified embedding model."""
  return FAISS.from_texts(chunks, embedding_model)

def perform_similarity_search(faiss_index, query, k=3):
  """Search for specific queries within the embedded transcript using the FAISS index."""
  results = faiss_index.similarity_search(query, k=k)
  return results

def retrieve(query, faiss_index, k=7):
  """Retrieve relevant context from the FAISS index based on the user's query."""
  relevant_context = faiss_index.similarity_search(query, k=k)
  return relevant_context
