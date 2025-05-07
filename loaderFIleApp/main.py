import streamlit as st
import os

# Directory to save uploaded images
UPLOAD_DIR = 'uploaded_images'
# Create the directory if it doesn't exist
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# Initialize session state for storing image file paths
if 'images' not in st.session_state:
    st.session_state.images = []

st.title('Image Uploader and Manager')

# File uploader allows multiple image uploads
uploaded_files = st.file_uploader(
    label='Upload Images',
    type=['png', 'jpg', 'jpeg', 'gif', 'bmp'],
    accept_multiple_files=True
)

# Handle uploaded files
if uploaded_files:
    for uploaded_file in uploaded_files:
        # Define file path
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        # Save the file if it's not already saved
        if file_path not in st.session_state.images:
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            # Add the saved file path to session state
            st.session_state.images.append(file_path)
            st.success(f"Saved file: {uploaded_file.name}")

st.header('Uploaded Images')
# Display a select box for picking saved images
if st.session_state.images:
    selected_image = st.selectbox('Select an image to view', st.session_state.images)
    st.image(selected_image, caption=os.path.basename(selected_image))
else:
    st.write('No images uploaded yet.')
