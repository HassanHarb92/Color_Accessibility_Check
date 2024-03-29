import numpy as np
import cv2
from colorblind import colorblind
import matplotlib.pyplot as plt
def apply_colorblind_simulation(image_path, cb_type):
    img = cv2.imread(image_path)
    if img is None:
        print("Failed to read the image. Please check the file path.")
        return

    # Resize the image to reduce memory usage
    scale_percent = 50  # Example: resize to 50% of the original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    # Convert BGR to RGB
    img_rgb = resized_img[..., ::-1]

    # Apply color vision deficiency simulation
    simulated_img = colorblind.simulate_colorblindness(img_rgb, colorblind_type=cb_type)

    # Convert RGB back to BGR for saving
    simulated_img_bgr = simulated_img[..., ::-1]
    output_path = image_path.replace('.jpg', f'_{cb_type}.jpg')
    cv2.imwrite(output_path, simulated_img_bgr)
    print(f"Saved {cb_type} simulated image to {output_path}")

# Example usage:
image_path = 'image1.jpg'
apply_colorblind_simulation(image_path, 'tritanopia')

