import os
import json
import glob
import requests
from pathlib import Path
import re

import streamlit as st
from streamlit.logger import get_logger

PROJECT_DIR = Path(__file__).parent
UPLOAD_DATA_DIR = "/data/upload_video"
Path(UPLOAD_DATA_DIR).mkdir(parents=True, exist_ok=True)

IMAGE_DATA_DIR = "/data/image"
Path(IMAGE_DATA_DIR).mkdir(parents=True, exist_ok=True)

logger = get_logger(__name__)

class API:
    PROCESS_URL = "http://backend:8080/process"
    CHAT_URL = "http://backend:8080/chat"

def send_request(url: str, payload: dict):
    res = requests.post(url, json=payload)
    result = json.loads(res.content)
    return result

def get_valid_milvus_name(name: str):
    return re.sub("[^0-9a-zA-Z]+", "_", name)

def get_video_info(doc_dir: str = UPLOAD_DATA_DIR) -> dict:
    """List all docs in a directory and generate meta data for each file"""
    file_names = []
    for file in glob.glob(os.path.join(doc_dir, "*")):
        file_name = Path(file).stem
        file_names.append(file_name)
    return file_names

def main():
    title = "Video Understanding Chatbot"
    st.title(title)

    # Get availble collections
    uploaded_videos = get_video_info()
    collection_names = [get_valid_milvus_name(name) for name in uploaded_videos]

    # Introductory
    collection_names_str = ', '.join(uploaded_videos)
    st.markdown(f"Ask anything about {collection_names_str}")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Upload image (optional)
    image_query = None

    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0
    
    upload_image = st.file_uploader("Upload image (Optional)", type="jpg", key=f"uploader_{st.session_state.uploader_key}")

    def update_key():
        st.session_state.uploader_key += 1
         
    if upload_image:
        image_query = os.path.join(IMAGE_DATA_DIR, upload_image.name)
        with open(image_query, "wb") as f:
            f.write(upload_image.read())
        update_key()
    
    # Accept user input
    if text_query := st.chat_input():

        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(text_query)
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": text_query})

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            payload = {
                "collection_names": collection_names,
                "text_query": text_query,
                "image_query": image_query
            }
            response = send_request(API.CHAT_URL, payload)
            answer = response["generation"]
            st.markdown(answer)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": answer})
        image_query = None


if __name__ == "__main__":
    main()