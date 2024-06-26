import streamlit as st
import cv2
import numpy as np
from PIL import Image
from colorblind import colorblind

def resize_image(image, max_width=4000):
    """
    Resize the image to maintain the aspect ratio and ensure it's under a max width.
    """
    width, height = image.size
    if width > max_width:
        new_height = int((max_width / width) * height)
        new_width = max_width
        image = image.resize((new_width, new_height), Image.ANTIALIAS)
    return image

def apply_colorblind_simulation(img, cb_type):
    """
    Applies color vision deficiency simulation, except for monochromacy
    which is handled separately due to its unique grayscale conversion.
    """
    if cb_type == 'monochromacy':
        # Convert image to grayscale to simulate Monochromacy
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        # Convert BGR to RGB for colorblind simulation
        img_rgb = img[..., ::-1]
        simulated_img = colorblind.simulate_colorblindness(img_rgb, colorblind_type=cb_type)
        # Convert RGB back to BGR for saving
        return simulated_img[..., ::-1]

st.title('Color Accessibility Simulation App')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    
    pil_image = Image.fromarray(cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB))
    pil_image_resized = resize_image(pil_image)
    opencv_image_resized = cv2.cvtColor(np.array(pil_image_resized), cv2.COLOR_RGB2BGR)

    with st.spinner('Processing...'):
        col1, col2 = st.columns(2)
        with col1:
            st.image(opencv_image_resized, channels="BGR", caption="Original")
        with col2:
            simulated_image_protanopia = apply_colorblind_simulation(opencv_image_resized, 'protanopia')
            st.image(simulated_image_protanopia, channels="BGR", caption="Protanopia")
        
        col3, col4 = st.columns(2)
        with col3:
            simulated_image_deuteranopia = apply_colorblind_simulation(opencv_image_resized, 'deuteranopia')
            st.image(simulated_image_deuteranopia, channels="BGR", caption="Deuteranopia")
        with col4:
            simulated_image_tritanopia = apply_colorblind_simulation(opencv_image_resized, 'tritanopia')
            st.image(simulated_image_tritanopia, channels="BGR", caption="Tritanopia")

        col5, _ = st.columns([1, 3])  # Use _ for placeholder to align the monochromacy image to the left
        with col5:
            simulated_image_monochromacy = apply_colorblind_simulation(opencv_image_resized, 'monochromacy')
            st.image(simulated_image_monochromacy, channels="GRAY", caption="Monochromacy")

