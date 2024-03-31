import streamlit as st
import cv2
import numpy as np
from PIL import Image
# Assume daltonize or a similar functionality is correctly imported
from daltonize import daltonize

def apply_daltonization(image, deficiency_type):
    """
    Apply Daltonization to the image based on the selected color vision deficiency.
    """
    if deficiency_type == 'protanopia':
        return daltonize.protanopia(image)
    elif deficiency_type == 'deuteranopia':
        return daltonize.deuteranopia(image)
    elif deficiency_type == 'tritanopia':
        return daltonize.tritanopia(image)
    else:
        return image  # Return the original if no match

st.title('Make Images Colorblind Friendly')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
deficiency_type = st.selectbox("Select the type of color vision deficiency:", 
                                options=["protanopia", "deuteranopia", "tritanopia"])

if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Apply Daltonization
    daltonized_image = apply_daltonization(image, deficiency_type)
    
    # Convert numpy array (image) to PIL Image for display in Streamlit
    daltonized_image_pil = Image.fromarray(daltonized_image)
    
    # Display the original and daltonized images
    st.image(image, caption="Original Image", use_column_width=True)
    st.image(daltonized_image_pil, caption=f"Daltonized Image ({deficiency_type})", use_column_width=True)

