import torch
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image, ImageColor
import numpy as np
import os
import sys
from models.u2net import U2NET
import cv2

def load_u2net_model():
    model = U2NET()  # Initialize the U2-Net model
    model_path = "models/u2net.pth"  # Path to your downloaded pre-trained model
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))  # Load the pre-trained weights
    model.load_state_dict(state_dict)  # Load the weights into the model
    model.eval()  # Set the model to evaluation mode
    return model
u2net_model = load_u2net_model()
# Function to preprocess image for U2-Net
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)

# Assuming the necessary transformations for the image are already defined
def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize((320, 320)),  # Resize the image to match the model input size
        transforms.ToTensor(),          # Convert the image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# def remove_bg_with_u2net(image):
    # Load the pre-trained U2-Net model
    model = load_u2net_model()

    # Preprocess the input image
    image_resized = image.resize((320, 320))  # Resize the input image to 320x320

    image_tensor = transform_image(image_resized)

    # Perform inference on the image
    with torch.no_grad():
        output = model(image_tensor)  # Get model output

    # Extract the mask (first element of the output tuple, if it is a tuple)
    mask, *_ = output

    # Ensure the mask is a tensor and apply .squeeze(0) to remove the batch dimension
    mask = mask.squeeze(0)  # Shape: [1, 320, 320] -> [320, 320]

    # Post-process the mask (e.g., thresholding, resizing, etc.)
    mask = np.where(mask > 0.5, 1, 0)  # Convert to binary mask using thresholding

    # Convert the mask to a tensor (ensure it's a torch tensor)
    mask_tensor = torch.tensor(mask, dtype=torch.float32)  # Convert to tensor of the correct type

    # Add a batch dimension to the mask tensor
    mask_tensor = mask_tensor.unsqueeze(0)  # Shape: [1, 320, 320]

    # Get the original image dimensions (height, width)
    original_size = image.size  # This returns (width, height) in pixels
    print("Original Image Size: ", original_size)  # Debugging

    # Correct target size using (height, width)
    target_size = (original_size[1], original_size[0])  # (height, width)
    print("Target Resize Size: ", target_size)  # Debugging

    # Resize the mask to match the original input image size
    mask_resized = F.interpolate(mask_tensor, size=target_size, mode='bilinear', align_corners=False)

    # Remove extra dimensions after interpolation
    mask_resized = mask_resized.squeeze(0)  # Shape: [1, H, W] -> [H, W]

    # Convert the resized mask back to a NumPy array
    mask_resized = mask_resized.cpu().numpy()

    # Ensure the mask is in the correct format for Streamlit
    if mask_resized.ndim == 2:  # If it's a single channel
        mask_resized = np.expand_dims(mask_resized, axis=-1)  # Shape: [H, W, 1]
    elif mask_resized.ndim == 3 and mask_resized.shape[0] == 1:  # If it's [1, H, W]
        mask_resized = mask_resized[0]  # Shape: [H, W]

    # Ensure mask is in the correct type for conversion
    mask_resized = (mask_resized * 255).astype(np.uint8)  # Convert binary mask to 0-255 range

    # Convert to an image using PIL
    try:
        mask_image = Image.fromarray(mask_resized)  # Convert to an image
    except TypeError as e:
        print(f"Error while converting mask to image: {e}")
        print(f"Mask shape: {mask_resized.shape}, dtype: {mask_resized.dtype}")
        raise e

    return mask_image

# Update the remove_bg function in your processing flow
def remove_bg(image):
    image = image.convert("RGB")  # Ensure RGB format
    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor()
    ])
    
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = u2net_model(input_tensor)[0]  # Get mask
        mask = output.squeeze().cpu().numpy()
        mask = (mask - mask.min()) / (mask.max() - mask.min())  # Normalize mask
        mask = (mask * 255).astype(np.uint8)  # Convert to 8-bit format
        mask = cv2.resize(mask, (image.width, image.height))  # Resize to original image size

    # Convert to RGBA (with transparency)
    image_np = np.array(image)
    alpha_channel = mask[..., np.newaxis]  # Convert grayscale mask to 4th channel
    rgba_image = np.concatenate((image_np, alpha_channel), axis=-1)  # Merge RGB with Alpha

    return Image.fromarray(rgba_image)

# Function to replace background with either a custom image or solid color
def replace_background(image, background_type="None", bg_image=None, color=None):
    """
    Replaces the background of an image after removing it with U2-Net.
    
    Parameters:
    - image: PIL Image, the input image.
    - background_type: str, either "Solid Color" or "Custom Image".
    - bg_image: PIL Image, optional, custom background image.
    - color: tuple, optional, solid color in (R, G, B, A) format.

    Returns:
    - Processed PIL Image with the new background applied.
    """

    # Step 1: Remove background first
    mask = remove_bg(image)  # This should return a binary mask (white = foreground, black = removed)
    
    # Convert image to RGBA if not already
    image = image.convert("RGBA")

    # Step 2: Create a blank background (either solid color or custom image)
    if background_type == "Solid Color":
        new_bg = Image.new("RGBA", image.size, color)  # Create solid color background
    
    elif background_type == "Custom Image" and bg_image:
        bg_image = bg_image.resize(image.size)  # Resize background to match the image
        new_bg = bg_image.convert("RGBA")  # Ensure it has an alpha channel

    else:
        return image  # If no background replacement is needed, return the original

    # Step 3: Composite foreground (masked) onto the new background
    final_image = Image.composite(image, new_bg, mask)

    return final_image

# --- Function to Blur Background ---
def blur_bg(image, blur_intensity):
    img_cv = np.array(image)
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV

    # Generate the background mask using rembg
    mask = remove(image, only_mask=True)
    mask = np.array(mask.convert("L"))
    _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)

    # Ensure blur intensity is odd (required for GaussianBlur)
    if blur_intensity % 2 == 0:
        blur_intensity += 1

    # Apply Gaussian Blur
    blurred = cv2.GaussianBlur(img_rgb, (blur_intensity, blur_intensity), 0)

    # Convert mask to 3-channel for blending
    mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask_float = mask_3channel.astype(float) / 255

    # Blend foreground with blurred background
    blended = (img_rgb * mask_float + blurred * (1 - mask_float)).astype(np.uint8)

    # Convert BGR back to RGB
    return Image.fromarray(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))