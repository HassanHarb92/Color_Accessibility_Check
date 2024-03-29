import cv2
import numpy as np

def read_and_save_image(image_path):
    """
    Reads an image, converts it to grayscale (as a simple operation),
    and saves it back to disk with a new filename.
    """
    # Attempt to read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to read {image_path}. Please check the file path.")
        return

    # Convert the image to grayscale as a simple operation
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Save the result
    output_path = image_path.replace('.jpg', '_gray.jpg')
    cv2.imwrite(output_path, gray_img)
    print(f"Saved the processed image to {output_path}")

# Replace 'path_to_your_image.jpg' with the actual path to your image file
image_path = 'image1.jpg'
read_and_save_image(image_path)

