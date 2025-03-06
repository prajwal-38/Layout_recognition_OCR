import os
import cv2
import numpy as np

def create_pyramid(image_path, output_size=(512, 512)):
    """
    Reduce image resolution and save the resized image
    
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Failed to load image: {image_path}")
            return f"Failed to load image: {image_path}"
        
        #Resize image
        resized = cv2.resize(img, output_size, interpolation=cv2.INTER_AREA)
        
        #Overwrite the original image
        cv2.imwrite(image_path, resized)
        
        return f"Processed: {image_path}"
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return f"Error processing {image_path}: {e}"

def process_images_in_directory(base_dir):
    """
    Process all images in subfolders of the given base directory

    """
    image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']
    processed_count = 0
    error_count = 0

    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                full_path = os.path.join(root, file)
                
                result = create_pyramid(full_path)

                if "Processed:" in result:
                    processed_count += 1
                else:
                    error_count += 1
                
                print(result)
    
    print(f"\nProcessing complete:")
    print(f"Total images processed: {processed_count}")
    print(f"Total errors: {error_count}")

base_dir = 'data/processed/images'

process_images_in_directory(base_dir)
