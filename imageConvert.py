import os
import cv2
import numpy as np

def create_speaking_not_speaking(input_folder, output_folder):
    # Define output paths
    speaking_folder = os.path.join(output_folder, "speaking")
    not_speaking_folder = os.path.join(output_folder, "not_speaking")
    os.makedirs(speaking_folder, exist_ok=True)
    os.makedirs(not_speaking_folder, exist_ok=True)

    # Process each image
    for img_name in os.listdir(input_folder):
        img_path = os.path.join(input_folder, img_name)
        img = cv2.imread(img_path)
        
        if img is None:
            continue
        
        # Resize image to 256x256 (optional)
        img = cv2.resize(img, (256, 256))
        
        # Save non-speaking version (original image)
        cv2.imwrite(os.path.join(not_speaking_folder, img_name), img)
        
        # Create speaking version (add a blue ring)
        overlay = img.copy()
        height, width, _ = img.shape
        center = (width // 2, height // 2)
        radius = min(width, height) // 2 - 10
        cv2.circle(overlay, center, radius, (255, 0, 0), 10)  # Blue ring
        cv2.imwrite(os.path.join(speaking_folder, img_name), overlay)
        
    print("✅ Image processing complete!")

# Example usage
input_folder = "images"  # Change this to your folder
output_folder = "output_images"
create_speaking_not_speaking(input_folder, output_folder)
