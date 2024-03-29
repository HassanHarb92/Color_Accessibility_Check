import streamlit as st
import cv2
import numpy as np
from PIL import Image
from colorblind import colorblind

def resize_image(image, max_size=178956970, max_width=4000):
    """
    Resize the image to avoid DecompressionBombError and to ensure
    the size is under a specific threshold.
    """
    width, height = image.size
    new_width = width
    new_height = height
    # Calculate the new size to keep the aspect ratio
    aspect_ratio = width / height

    if width * height > max_size:
        # Calculate the new height based on the max_width
        # while maintaining the aspect ratio
        new_width = max_width
        new_height = int(max_width / aspect_ratio)
    
    return image.resize((new_width, new_height), Image.ANTIALIAS)

def apply_colorblind_simulation(img, cb_type):
    # Convert BGR to RGB
    img_rgb = img[..., ::-1]

    # Apply color vision deficiency simulation
    simulated_img = colorblind.simulate_colorblindness(img_rgb, colorblind_type=cb_type)

    # Convert RGB back to BGR for saving
    return simulated_img[..., ::-1]

st.title('Color Accessibility Simulation App')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)

    # Convert to PIL Image for easy resizing
    pil_image = Image.fromarray(cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB))
    
    # Resize if necessary
    pil_image_resized = resize_image(pil_image)
    
    # Convert back to OpenCV image to display with Streamlit
    opencv_image_resized = cv2.cvtColor(np.array(pil_image_resized), cv2.COLOR_RGB2BGR)
    
    st.image(opencv_image_resized, channels="BGR", caption="Original Image")

    # Display a message to indicate processing
    with st.spinner('Processing...'):
        # Color blindness types to simulate
        cb_types = ['protanopia', 'deuteranopia', 'tritanopia']
        
        # For each color blindness type, simulate and display the image
        for cb_type in cb_types:
            simulated_image = apply_colorblind_simulation(opencv_image_resized, cb_type)
            
            # Display each simulated image
            st.image(simulated_image, channels="BGR", caption=f"Simulated {cb_type.capitalize()}")

st.caption("Developed with ❤️ using Streamlit and the Colorblind Library")

