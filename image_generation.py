import streamlit as st
from google.genai import types
import io
import base64
from PIL import Image
from datetime import datetime
import requests
from config import MODEL_ID

def generate_image(prompt, client):
    """
    Generate an image using Gemini based on the provided prompt
    """
    try:
        # Save prompt to history
        st.session_state.prompt_history.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type": "Gemini Image Generation",
            "prompt": prompt
        })
        
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=['Text', 'Image']
            )
        )

        image_data = None
        image_text = None

        for part in response.candidates[0].content.parts:
            if hasattr(part, 'text') and part.text is not None:
                image_text = part.text
            elif hasattr(part, 'inline_data') and part.inline_data is not None:
                image_data = part.inline_data.data

        if image_data:
            img = Image.open(io.BytesIO(image_data))
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)

            return {
                "image": img,
                "image_data": img_byte_arr,
                "prompt": prompt,
                "text": image_text,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        else:
            # Fallback to a placeholder image
            width, height = 512, 512
            img_url = f"https://picsum.photos/{width}/{height}?random={datetime.now().timestamp()}"
            response = requests.get(img_url)
            img = Image.open(io.BytesIO(response.content))

            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)

            return {
                "image": img,
                "image_data": img_byte_arr,
                "prompt": prompt + " (placeholder)",
                "text": "Generated using a placeholder image service since Gemini didn't return an image.",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
    except Exception as e:
        st.error(f"Error generating image: {str(e)}")

        # Fallback to a placeholder image on error
        width, height = 512, 512
        img_url = f"https://picsum.photos/{width}/{height}?random={datetime.now().timestamp()}"
        response = requests.get(img_url)
        img = Image.open(io.BytesIO(response.content))

        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        return {
            "image": img,
            "image_data": img_byte_arr,
            "prompt": prompt + " (placeholder due to error)",
            "text": f"Error with Gemini API: {str(e)}. Using placeholder image.",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }