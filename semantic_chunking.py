import streamlit as st
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import faiss
import nltk
from nltk.tokenize import sent_tokenize
import os
import pickle

# Download NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class SemanticChunkManager:
    """Manages semantic chunking and vector database operations"""
    
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """Initialize with a sentence transformer model"""
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []
        self.chunk_size = 2000
        self.overlap = 200
        self.embedding_dim = 384  # Default embedding dimension for the model
    
    def semantic_chunking(self, text, chunk_size=None, overlap=None):
        """
        Split text into semantically meaningful chunks
        
        Args:
            text: Text to chunk
            chunk_size: Target size of each chunk in characters
            overlap: Character overlap between chunks
            
        Returns:
            List of text chunks
        """
        if chunk_size is not None:
            self.chunk_size = chunk_size
        if overlap is not None:
            self.overlap = overlap
            
        # If text is shorter than chunk_size, return it as a single chunk
        if len(text) <= self.chunk_size:
            return [text]
        
        # First split text into sentences
        sentences = sent_tokenize(text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_len = len(sentence)
            
            # If adding this sentence would exceed chunk_size
            if current_size + sentence_len > self.chunk_size and current_chunk:
                # Add the current chunk to our list of chunks
                chunks.append(" ".join(current_chunk))
                
                # Start a new chunk with overlap
                # Keep sentences from the end of the previous chunk for context
                overlap_size = 0
                overlap_chunk = []
                
                for s in reversed(current_chunk):
                    if overlap_size + len(s) <= self.overlap:
                        overlap_chunk.insert(0, s)
                        overlap_size += len(s)
                    else:
                        break
                        
                current_chunk = overlap_chunk
                current_size = overlap_size
            
            # Add the current sentence to the chunk
            current_chunk.append(sentence)
            current_size += sentence_len
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks
    
    def build_vector_db(self, chunks):
        """
        Build a FAISS vector database from text chunks
        
        Args:
            chunks: List of text chunks
            
        Returns:
            FAISS index
        """
        self.chunks = chunks
        
        # Generate embeddings for each chunk
        embeddings = self.model.encode(chunks)
        
        # Convert to float32 (required by FAISS)
        embeddings = np.array(embeddings).astype('float32')
        
        # Create FAISS index
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.index.add(embeddings)
        
        return self.index
    
    def search(self, query, top_k=3):
        """
        Search for the most semantically similar chunks to the query
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of most relevant text chunks
        """
        if not self.index or not self.chunks:
            return []
        
        # Get query embedding
        query_embedding = self.model.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        
        # Search the index
        distances, indices = self.index.search(query_embedding, min(top_k, len(self.chunks)))
        
        # Return the corresponding chunks
        results = [self.chunks[idx] for idx in indices[0]]
        
        return results
    
    def save(self, directory="./vector_db"):
        """Save the index and chunks to disk"""
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        # Save the FAISS index
        if self.index:
            faiss.write_index(self.index, os.path.join(directory, "chunks.index"))
            
        # Save the chunks
        with open(os.path.join(directory, "chunks.pkl"), "wb") as f:
            pickle.dump(self.chunks, f)
            
        # Save the model name for later restoration
        with open(os.path.join(directory, "model_info.pkl"), "wb") as f:
            pickle.dump({
                "embedding_dim": self.embedding_dim,
                "model_name": self.model.get_sentence_embedding_dimension
            }, f)
    
    def load(self, directory="./vector_db"):
        """Load the index and chunks from disk"""
        if not os.path.exists(directory):
            return False
            
        try:
            # Load the FAISS index
            self.index = faiss.read_index(os.path.join(directory, "chunks.index"))
            
            # Load the chunks
            with open(os.path.join(directory, "chunks.pkl"), "rb") as f:
                self.chunks = pickle.load(f)
                
            return True
        except Exception as e:
            print(f"Error loading vector database: {e}")
            return False


# Function to be used in Streamlit app
def setup_chunk_manager():
    """Setup and initialize the chunk manager in session state"""
    if "chunk_manager" not in st.session_state:
        with st.spinner("Setting up semantic text processing..."):
            st.session_state.chunk_manager = SemanticChunkManager()


def chunk_text_semantic(text, chunk_size=2000, overlap=200):
    """
    Interface function for chunking text semantically
    """
    setup_chunk_manager()
    chunks = st.session_state.chunk_manager.semantic_chunking(
        text, chunk_size=chunk_size, overlap=overlap
    )
    return chunks


def search_context_semantic(query, top_k=3):
    """
    Interface function for semantic search through chunks
    """
    setup_chunk_manager()
    
    if not st.session_state.context_chunks:
        return []
    
    # If index hasn't been built yet, build it
    if not st.session_state.chunk_manager.index:
        with st.spinner("Building vector database from context..."):
            st.session_state.chunk_manager.build_vector_db(st.session_state.context_chunks)
    
    # Search for relevant chunks
    return st.session_state.chunk_manager.search(query, top_k)