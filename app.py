import streamlit as st
import io
from utils import load_image, convert_image_to_bytes
from background import remove_bg
from zip_creator import create_zip_from_images

# --- Streamlit UI Layout ---
st.markdown("<h1 style='color: black; text-align: center;'>Background Remover</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='color: black; text-align: center;'>A simple AI-powered tool to remove the background of an image.</h5>", unsafe_allow_html=True)

# --- Initialize session state ---
if "processing_mode" not in st.session_state:
    st.session_state.processing_mode = None
if "processed_images" not in st.session_state:
    st.session_state.processed_images = []
if "file_names" not in st.session_state:
    st.session_state.file_names = []
if "zip_buffer" not in st.session_state:  # Initialize an empty zip buffer
    st.session_state.zip_buffer = io.BytesIO()

# --- Mode Selection (Buttons) ---
st.sidebar.header("⚙️ Select Mode")

col1, col2 = st.sidebar.columns(2)

if col1.button("Single Image"):
    st.session_state.processing_mode = "Single Image"
if col2.button("Batch Images"):
    st.session_state.processing_mode = "Batch Images"

# Only proceed if a mode is selected
if st.session_state.processing_mode:
    st.sidebar.subheader(f"📂 Upload Your {st.session_state.processing_mode}")

    # Image upload logic
    if st.session_state.processing_mode == "Single Image":
        uploaded_files = st.sidebar.file_uploader("Upload a single image", type=["jpg", "jpeg", "png"], accept_multiple_files=False)
        uploaded_files = [uploaded_files] if uploaded_files else []  # Convert to list for consistency
    else:
        uploaded_files = st.sidebar.file_uploader("Upload multiple images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    # Process images if files are uploaded
    if uploaded_files:
        uploaded_images = []
        file_names = []

        for file in uploaded_files:
            if file:
                image = load_image(file)
                uploaded_images.append(image)
                file_names.append(file.name)

        st.markdown("<h3 style='text-align: center;'>📤 Uploaded & Processed Images</h3>", unsafe_allow_html=True)

        # Process images
        processed_images = []
        for i, image in enumerate(uploaded_images):
            st.markdown(f"### Uploaded Image {i+1}")
            col1, col2 = st.columns(2)

            with col1:
                st.image(image, caption=f"Uploaded Image {i+1}", use_container_width=True)

            processed_image = remove_bg(image)
            processed_images.append(processed_image)

            with col2:
                st.image(processed_image, caption=f"Processed Image {i+1}", use_container_width=True)

            img_bytes = convert_image_to_bytes(processed_image)
            st.download_button(
                label=f"Download Processed Image {i+1}",
                data=img_bytes,
                file_name=f"processed_image_{i+1}.png",
                mime="image/png",
                use_container_width=True,
                key=f"download_{i}"  # Unique key for each button
            )

        # Store processed images for zip download
        st.session_state.processed_images = processed_images
        st.session_state.file_names = file_names

    # Enable the zip download button only if images have been processed
    if st.session_state.processed_images:
        zip_buffer = create_zip_from_images(st.session_state.processed_images, st.session_state.file_names)
        st.session_state.zip_buffer = zip_buffer  # Store the buffer

        st.sidebar.download_button(
            label="Download Processed Images as Zip",
            data=st.session_state.zip_buffer.getvalue(),
            file_name="processed_images.zip",
            mime="application/zip",
            disabled=False,
            key="final_zip_download"  # Unique key to avoid conflict
        )
    else:
    	st.sidebar.download_button(
        label="Download Processed Images as Zip",
        data=st.session_state.zip_buffer.getvalue(),
        file_name="processed_images.zip",
        mime="application/zip",
        disabled=True,
        key="initial_zip_download"  # Unique key to avoid duplication error
        )
