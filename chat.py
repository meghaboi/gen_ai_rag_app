import streamlit as st
from datetime import datetime
from config import CHAT_MODEL
from context_manager import get_active_chunk_context, search_context

def process_chat(prompt, message_placeholder, client):
    """
    Process a chat message with Gemini, handling context integration
    """
    try:
        # Determine what context to use
        context_to_use = ""
        
        if st.session_state.context_chunks:
            # Get context setting from tab5
            context_option = "Auto-search relevant chunks"  # Default
            if "context_option" in st.session_state:
                context_option = st.session_state.context_option
                
            if context_option == "Use active chunk only":
                context_to_use = get_active_chunk_context()
            elif context_option == "Auto-search relevant chunks":
                relevant_chunks = search_context(prompt)
                if relevant_chunks:
                    context_to_use = "\n\n".join(relevant_chunks)
                else:
                    # Fallback to active chunk if no relevant chunks found
                    context_to_use = get_active_chunk_context()
            else:  # Use all chunks
                context_to_use = "\n\n".join(st.session_state.context_chunks)
        else:
            context_to_use = st.session_state.context_text
        
        # Include context if available
        if context_to_use:
            final_prompt = f"""
            Context information:
            {context_to_use}
            
            Now, please respond to the following question or request using the context above when relevant:
            {prompt}
            """
        else:
            final_prompt = prompt
        
        # Save prompt to history
        st.session_state.prompt_history.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type": "Gemini Chat",
            "prompt": final_prompt
        })
        
        # Show the prompt being sent in an expandable section
        with st.expander("View prompt sent to Gemini"):
            st.code(final_prompt, language="text")
        
        response = client.models.generate_content(
            model=CHAT_MODEL,
            contents=final_prompt
        )
        response_text = response.text

        message_placeholder.markdown(response_text)

        st.session_state.messages.append({"role": "assistant", "content": response_text})

    except Exception as e:
        message_placeholder.error(f"Error: {str(e)}")