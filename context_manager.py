import streamlit as st
import re

def chunk_text(text, chunk_size):
    """
    Split text into chunks of approximately chunk_size characters.
    Try to split at paragraph or sentence boundaries when possible.
    """
    # If text is shorter than chunk_size, return it as a single chunk
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    current_pos = 0
    
    while current_pos < len(text):
        # If we're near the end, just take the rest
        if current_pos + chunk_size >= len(text):
            chunks.append(text[current_pos:])
            break
        
        # Try to find paragraph break within the chunk_size from current position
        end_pos = text.rfind('\n\n', current_pos, current_pos + chunk_size)
        
        # If no paragraph break, try to find sentence break
        if end_pos == -1:
            end_pos = text.rfind('. ', current_pos, current_pos + chunk_size)
            if end_pos != -1:
                end_pos += 2  # Include the period and space
        
        # If no sentence break, try to find any newline
        if end_pos == -1:
            end_pos = text.rfind('\n', current_pos, current_pos + chunk_size)
            if end_pos != -1:
                end_pos += 1  # Include the newline
        
        # If still no natural break, just cut at chunk_size
        if end_pos == -1 or end_pos <= current_pos:
            end_pos = current_pos + chunk_size
        
        # Add the chunk
        chunks.append(text[current_pos:end_pos])
        current_pos = end_pos
    
    return chunks

def get_active_chunk_context():
    """Get the active chunk or return empty string if no chunks available"""
    if st.session_state.context_chunks and len(st.session_state.context_chunks) > st.session_state.active_chunk:
        return st.session_state.context_chunks[st.session_state.active_chunk]
    return ""

def search_context(query, top_k=3):
    """
    Simple keyword-based search through all chunks.
    Returns the top k chunks that have the most keyword matches.
    """
    if not st.session_state.context_chunks:
        return []
    
    # Extract keywords from the query (simple approach)
    keywords = re.findall(r'\b\w{3,}\b', query.lower())
    
    # Score each chunk based on keyword matches
    chunk_scores = []
    for i, chunk in enumerate(st.session_state.context_chunks):
        chunk_lower = chunk.lower()
        score = sum(1 for keyword in keywords if keyword in chunk_lower)
        chunk_scores.append((i, score))
    
    # Sort by score and get top k
    chunk_scores.sort(key=lambda x: x[1], reverse=True)
    top_chunks = [st.session_state.context_chunks[i] for i, score in chunk_scores[:top_k] if score > 0]
    
    return top_chunks