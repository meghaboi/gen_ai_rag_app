import streamlit as st
import google.generativeai as genai
import io
from PIL import Image
from datetime import datetime
import requests

# Import custom modules
from image_generation import generate_image
from chat import process_chat
from context_manager import chunk_text, get_active_chunk_context, search_context
from config import GEMINI_API_KEY, MODEL_ID

# Initialize Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)

# Set page configuration
st.set_page_config(page_title="AI Assistant", layout="wide")

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_image" not in st.session_state:
    st.session_state.current_image = None

if "image_history" not in st.session_state:
    st.session_state.image_history = []

if "image_prompts" not in st.session_state:
    st.session_state.image_prompts = []

if "context_text" not in st.session_state:
    st.session_state.context_text = ""

if "context_chunks" not in st.session_state:
    st.session_state.context_chunks = []

if "active_chunk" not in st.session_state:
    st.session_state.active_chunk = 0

if "chunk_size" not in st.session_state:
    st.session_state.chunk_size = 2000

if "prompt_history" not in st.session_state:
    st.session_state.prompt_history = []

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["Chat", "Image Generation", "Context Manager", "Prompt History"])

# Tab 4: Prompt History
with tab4:
    st.header("Prompt History")
    
    if st.session_state.prompt_history:
        st.write("This tab shows all prompts sent to AI models")
        
        for i, entry in enumerate(reversed(st.session_state.prompt_history)):
            with st.expander(f"{entry['timestamp']} - {entry['type']}"):
                st.code(entry['prompt'], language="text")
        
        if st.button("Clear Prompt History"):
            st.session_state.prompt_history = []
            st.success("Prompt history cleared!")
            st.experimental_rerun()
    else:
        st.info("No prompts have been sent yet. Interact with the application to see prompts here.")

# Tab 3: Context Manager
with tab3:
    st.header("Upload Context for Chat")
    
    st.markdown("""
    Upload a text file to provide context for your conversations with the AI assistant. 
    The content will be split into manageable chunks if it's large.
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        context_file = st.file_uploader("Upload a text file for context", type=["txt"])
        chunk_size = st.slider("Chunk size (characters)", 500, 5000, st.session_state.chunk_size, 100)
        st.session_state.chunk_size = chunk_size
        
        if context_file is not None:
            text_content = context_file.read().decode("utf-8")
            st.session_state.context_text = text_content
            
            # Chunk the text
            chunks = chunk_text(text_content, chunk_size)
            st.session_state.context_chunks = chunks
            st.session_state.active_chunk = 0
            
            st.success(f"Context uploaded and split into {len(chunks)} chunks!")
            
            if st.button("Clear Context"):
                st.session_state.context_text = ""
                st.session_state.context_chunks = []
                st.session_state.active_chunk = 0
                st.success("Context cleared successfully!")
                st.experimental_rerun()
                
    with col2:
        if st.session_state.context_chunks:
            st.subheader("Context Overview")
            st.write(f"Total chunks: {len(st.session_state.context_chunks)}")
            st.write(f"Average chunk size: {sum(len(c) for c in st.session_state.context_chunks) // len(st.session_state.context_chunks)} characters")
            
            # Chunk navigation
            st.subheader("Browse Chunks")
            active_chunk = st.number_input("Current chunk", 1, len(st.session_state.context_chunks), st.session_state.active_chunk + 1)
            st.session_state.active_chunk = active_chunk - 1
            
            st.write("Active chunk preview:")
            st.text_area("Chunk content", get_active_chunk_context(), height=300, disabled=True)
            
            # Chat settings
            st.subheader("Chat Context Settings")
            context_option = st.radio(
                "How to use context in chat:",
                ["Use active chunk only", "Auto-search relevant chunks", "Use all chunks"]
            )
            st.session_state.context_option = context_option
            
            if context_option == "Auto-search relevant chunks":
                st.info("The system will automatically find the most relevant chunks based on your query.")
            elif context_option == "Use all chunks":
                st.warning("Using all chunks may exceed context limits for very large documents.")
                
        else:
            if st.session_state.context_text:
                st.info("Context is loaded as a single chunk.")
                st.text_area("Context content", st.session_state.context_text, height=300, disabled=True)
            else:
                st.info("No context loaded. Upload a text file to provide context for the chat.")

# Tab 1: Chat with Gemini
with tab1:
    st.header("Chat with Gemini")
    
    if st.session_state.context_chunks:
        st.info(f"Using context ({len(st.session_state.context_chunks)} chunks) to inform responses.")
    elif st.session_state.context_text:
        st.info("Using context to inform responses.")
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask something..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            process_chat(prompt, message_placeholder, client)

# Tab 2: Image Generation
with tab2:
    st.header("Image Generation with Gemini")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("Generate New Image")
        image_prompt = st.text_area("Describe the image you want to generate",
                                    height=100,
                                    key="new_image_prompt",
                                    placeholder="Example: A 3D rendered image of a flying car over a futuristic city with tall skyscrapers and green parks")

        if st.button("Generate Image"):
            if image_prompt:
                with st.spinner("Generating image with Gemini..."):
                    enhanced_prompt = f"Create a detailed image of on the style of a Sketch: {image_prompt}. Make it high quality and visually appealing."
                    
                    # Show the prompt being sent
                    st.info(f"**Prompt sent to Gemini**: {enhanced_prompt}")

                    image_data = generate_image(enhanced_prompt, client)

                    if image_data:
                        st.session_state.current_image = image_data
                        st.session_state.image_history.append(image_data)
                        st.session_state.image_prompts.append(image_prompt)

                        st.success("Image generated successfully!")
                    else:
                        st.error("Failed to generate image.")
            else:
                st.warning("Please enter a prompt to generate an image.")

        if st.session_state.current_image:
            st.subheader("Edit Current Image")

            st.image(st.session_state.current_image['image'],
                      caption="Current image to edit",
                      width=300)

            edit_prompt = st.text_area("Describe the changes you want to make",
                                        height=100,
                                        key="edit_image_prompt",
                                        placeholder="Example: Make the sky more blue, add a sunset...")

            if st.button("Edit Image"):
                if edit_prompt:
                    with st.spinner("Editing image with Gemini..."):
                        if st.session_state.image_prompts:
                            current_prompt = ""

                            for prompt in reversed(st.session_state.image_prompts):
                                current_prompt += f"{prompt}. Keep the main part of the image but change the image to the following points: "

                            full_prompt = f"{current_prompt}. {edit_prompt}"
                        else:
                            full_prompt = f"{edit_prompt}"
                            
                        # Show the prompt being sent
                        st.info(f"**Prompt sent to Gemini**: {full_prompt}")

                        edited_image_data = generate_image(full_prompt, client)

                        if edited_image_data:
                            st.session_state.current_image = edited_image_data
                            st.session_state.image_history.append(edited_image_data)
                            st.session_state.image_prompts.append(edit_prompt)

                            st.success("Image edited successfully!")
                        else:
                            st.error("Failed to edit image.")
                else:
                    st.warning("Please enter a prompt to edit the image.")

    with col2:
        st.subheader("Current Image")

        if st.session_state.current_image:
            st.text(f"Prompt: {st.session_state.current_image['prompt']}")
            st.text(f"Generated: {st.session_state.current_image['timestamp']}")

            st.image(st.session_state.current_image['image'], use_column_width=True)

            if st.session_state.current_image.get('text'):
                with st.expander("Image Description"):
                    st.markdown(st.session_state.current_image['text'])

            st.download_button(
                label="Download Image",
                data=st.session_state.current_image['image_data'],
                file_name=f"gemini_image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                mime="image/png"
            )

            if st.button("Clear Current Image"):
                st.session_state.current_image = None
                st.experimental_rerun()
        else:
            st.info("No image generated yet.")

        st.subheader("Image History")
        if st.session_state.image_history:
            for i, img in enumerate(reversed(st.session_state.image_history[-5:])):
                with st.expander(f"Image {len(st.session_state.image_history) - i}"):
                    st.text(f"Prompt: {img['prompt']}")
                    st.text(f"Generated: {img['timestamp']}")
                    st.image(img['image'], use_column_width=True)
        else:
            st.info("No image history yet.")

st.markdown("---")
st.markdown("Powered by Meghanadh Pamidi")