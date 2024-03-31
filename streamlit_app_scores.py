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
    Applies color vision deficiency simulation.
    """
    # For Monochromacy, convert image to grayscale
    if cb_type == 'monochromacy':
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # For other types, use the colorblind library
    img_rgb = img[..., ::-1]  # Convert BGR to RGB
    simulated_img = colorblind.simulate_colorblindness(img_rgb, colorblind_type=cb_type)
    return simulated_img[..., ::-1]  # Convert RGB back to BGR

def luminance_change_metric(original_img, simulated_img):
    """
    Calculate the average change in luminance between the original
    and colorblind-simulated images.
    """
    original_lab = cv2.cvtColor(original_img, cv2.COLOR_BGR2Lab)
    simulated_lab = cv2.cvtColor(simulated_img, cv2.COLOR_BGR2Lab)
    luminance_diff = np.abs(original_lab[:,:,0] - simulated_lab[:,:,0])
    return np.mean(luminance_diff)

st.title('Color Accessibility Simulation App')
st.markdown('### Beta version')
st.markdown('*PSA DEIA Committee – Lab*')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)

    pil_image = Image.fromarray(cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB))
    pil_image_resized = resize_image(pil_image)
    opencv_image_resized = cv2.cvtColor(np.array(pil_image_resized), cv2.COLOR_RGB2BGR)

    with st.spinner('Processing...'):
        simulated_images = {
            'Protanopia': apply_colorblind_simulation(opencv_image_resized, 'protanopia'),
            'Deuteranopia': apply_colorblind_simulation(opencv_image_resized, 'deuteranopia'),
            'Tritanopia': apply_colorblind_simulation(opencv_image_resized, 'tritanopia'),
            # 'Monochromacy': apply_colorblind_simulation(opencv_image_resized, 'monochromacy') # Optional
        }
        
        # Calculate luminance change metric for each type, except Monochromacy
        luminance_changes = {name: luminance_change_metric(opencv_image_resized, img) for name, img in simulated_images.items() if name != 'Monochromacy'}
        
        # Display luminance change scores
        st.markdown("### Color Accessibility Scores")
        for cb_type, change in luminance_changes.items():
            st.write(f"{cb_type}: {change:.2f} average luminance change")

        # Display images
        st.image(opencv_image_resized, channels="BGR", caption="Original", width=300)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(simulated_images['Protanopia'], channels="BGR", caption="Protanopia", width=300)
        with col2:
            st.image(simulated_images['Deuteranopia'], channels="BGR", caption="Deuteranopia", width=300)
        with col3:
            st.image(simulated_images['Tritanopia'], channels="BGR", caption="Tritanopia", width=300)
        # Optionally add Monochromacy if included

