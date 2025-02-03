import zipfile
from io import BytesIO
from utils import convert_image_to_bytes

# --- Function to Create Zip from Processed Images ---
def create_zip_from_images(processed_images, file_names):
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for img, file_name in zip(processed_images, file_names):
            img_bytes = convert_image_to_bytes(img)
            zip_file.writestr(file_name, img_bytes)
    zip_buffer.seek(0)
    return zip_buffer