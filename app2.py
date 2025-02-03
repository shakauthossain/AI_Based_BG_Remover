import streamlit as st
from PIL import Image, ImageColor
from utils import load_image, convert_image_to_bytes
from background import remove_bg, replace_background
from zip_creator import create_zip_from_images
import io

# --- Streamlit UI Layout ---
st.markdown("<h1 style='color: black; text-align: center;'>Background Remover</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='color: black; text-align: center;'>A simple AI-powered tool to remove and replace the background of an image.</h5>", unsafe_allow_html=True)

# --- Initialize session state only once ---
if "processing_mode" not in st.session_state:
    st.session_state.processing_mode = None
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = None
if "processed_images" not in st.session_state:
    st.session_state.processed_images = []
if "file_names" not in st.session_state:
    st.session_state.file_names = []
if "zip_buffer" not in st.session_state:
    st.session_state.zip_buffer = io.BytesIO()
if "background_mode" not in st.session_state:
    st.session_state.background_mode = "None"
if "color_rgba" not in st.session_state:
    st.session_state.color_rgba = None
if "bg_image" not in st.session_state:
    st.session_state.bg_image = None
if "images_processed" not in st.session_state:
    st.session_state.images_processed = False  # Prevents reprocessing

# --- Mode Selection (Buttons) ---
st.sidebar.header("⚙️ Select Mode")

col1, col2 = st.sidebar.columns(2)

if col1.button("Single Image") and st.session_state.processing_mode != "Single Image":
    st.session_state.processing_mode = "Single Image"
    st.session_state.images_processed = False  # Reset processing flag

if col2.button("Batch Images") and st.session_state.processing_mode != "Batch Images":
    st.session_state.processing_mode = "Batch Images"
    st.session_state.images_processed = False  # Reset processing flag

# Only proceed if a mode is selected
if st.session_state.processing_mode:
    st.sidebar.subheader(f"📂 Upload Your {st.session_state.processing_mode}")

    # Image upload logic
    if st.session_state.processing_mode == "Single Image":
        uploaded_files = st.sidebar.file_uploader("Upload a single image", type=["jpg", "jpeg", "png"], accept_multiple_files=False)
        uploaded_files = [uploaded_files] if uploaded_files else []
    else:
        uploaded_files = st.sidebar.file_uploader("Upload multiple images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    # If new images are uploaded, update session state
    if uploaded_files:
        if uploaded_files != st.session_state.uploaded_files:  # Only update if new files are uploaded
            st.session_state.uploaded_files = uploaded_files
            st.session_state.images_processed = False  # Reset processing flag

    # Choose Background Mode
    st.sidebar.subheader("Choose Background Mode for Image")
    background_mode = st.sidebar.radio(
        "Select Background Mode",
        ("None", "Solid Color", "Custom Image"),
        index=("None", "Solid Color", "Custom Image").index(st.session_state.background_mode)
    )
    if background_mode != st.session_state.background_mode:
        st.session_state.background_mode = background_mode  # Update only on change
        st.session_state.images_processed = False  # Reset processing flag

    ## Solid color picker
    if background_mode == "Solid Color":
        color = st.sidebar.color_picker("Pick a background color", "#ffffff")  # Default to white
        st.session_state.color_rgba = ImageColor.getrgb(color) + (255,)  # Convert HEX to RGBA

    elif background_mode == "Custom Image":
        uploaded_bg = st.sidebar.file_uploader("Upload a background image", type=["jpg", "jpeg", "png"])
        if uploaded_bg:
            st.session_state.bg_image = Image.open(uploaded_bg)
            st.session_state.images_processed = False  # Reset processing flag

    # Process images only once
    if st.session_state.uploaded_files and not st.session_state.images_processed:
        uploaded_images = []
        file_names = []

        for file in st.session_state.uploaded_files:
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

            # Remove background first
            bg_removed_image = remove_bg(image)

            # Apply background replacement (if needed)
            if st.session_state.background_mode in ["Solid Color", "Custom Image"]:
                processed_image = replace_background(
                    bg_removed_image,
                    background_type=st.session_state.background_mode,
                    bg_image=st.session_state.bg_image,
                    color=st.session_state.color_rgba
                )
            else:
                processed_image = bg_removed_image  # No replacement

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
        st.session_state.images_processed = True  # Set flag to True once images are processed

    # Enable the zip download button only if images have been processed
    if st.session_state.images_processed and st.session_state.processed_images:
        zip_buffer = create_zip_from_images(st.session_state.processed_images, st.session_state.file_names)
        st.session_state.zip_buffer = zip_buffer  # Store the buffer

        st.sidebar.download_button(
            label="Download Processed Images as Zip",
            data=st.session_state.zip_buffer.getvalue(),
            file_name="processed_images.zip",
            mime="application/zip",
            key="final_zip_download"  # Unique key to avoid conflict
        )