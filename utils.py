from PIL import Image
from io import BytesIO

# --- Function to Load Image ---
def load_image(image_file):
    return Image.open(image_file)

# --- Function to Convert Image to Bytes for Download ---
def convert_image_to_bytes(image):
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format="PNG")
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr