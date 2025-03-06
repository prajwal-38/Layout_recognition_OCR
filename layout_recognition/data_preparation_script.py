import os
import json
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def load_annotation(annotation_path):
    try:
        with open(annotation_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading annotation {annotation_path}: {e}")
        return None

def load_image(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Cannot load image at {image_path}")
        return image
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def create_mask_from_regions(image_shape, text_regions):
    """
    Create a binary mask from text regions.
    
    """
    height, width = image_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Debug information
    print(f"Image dimensions: {width}x{height}")
    print(f"Text regions: {text_regions}")
    
    if text_regions and isinstance(text_regions, list):
        for i, region in enumerate(text_regions):
            if len(region) == 4:  # Ensure region has 4 coordinates
                x1, y1, x2, y2 = region
                
                # Debug
                print(f"Region {i}: original coordinates [{x1}, {y1}, {x2}, {y2}]")
                x1 = max(0, min(x1, width-1))
                y1 = max(0, min(y1, height-1))
                x2 = max(0, min(x2, width))
                y2 = max(0, min(y2, height))
                
                print(f"Region {i}: adjusted coordinates [{x1}, {y1}, {x2}, {y2}]")
                
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)  # -1 means filled rectangle
            else:
                print(f"Warning: Invalid region format: {region}")
    else:
        print(f"Warning: Invalid text_regions: {text_regions}")
    
    nonzero = np.count_nonzero(mask)
    print(f"Mask has {nonzero} non-zero pixels out of {height*width} ({nonzero/(height*width)*100:.2f}%)")
        
    return mask

def prepare_data(images_dir, annotations_dir, output_dir, image_size=(512, 512)):
    """
    Prepare data for layout recognition training.
    
    """
    print(f"Preparing data from:")
    print(f"  Images: {images_dir}")
    print(f"  Annotations: {annotations_dir}")
    print(f"  Output: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(images_dir):
        print(f"Error: Images directory does not exist: {images_dir}")
        return []
        
    if not os.path.exists(annotations_dir):
        print(f"Error: Annotations directory does not exist: {annotations_dir}")
        return []
    
    source_folders = [f for f in os.listdir(images_dir) if os.path.isdir(os.path.join(images_dir, f))]
    
    if not source_folders:
        print(f"Warning: No source folders found in {images_dir}")
        return []
        
    print(f"Found source folders: {source_folders}")
    data_items = []
    
    for source_folder in source_folders:
        image_folder = os.path.join(images_dir, source_folder)
        annotation_folder = os.path.join(annotations_dir, source_folder)
        
        print(f"Processing source: {source_folder}")
        print(f"  Image folder: {image_folder}")
        print(f"  Annotation folder: {annotation_folder}")
        
        if not os.path.isdir(image_folder):
            print(f"Warning: Image folder does not exist: {image_folder}")
            continue
            
        if not os.path.isdir(annotation_folder):
            print(f"Warning: Annotation folder does not exist: {annotation_folder}")
            continue
        
    
        image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not image_files:
            print(f"Warning: No image files found in {image_folder}")
            continue
            
        print(f"  Found {len(image_files)} image files")
        processed_count = 0
        
        for image_file in image_files:
            annotation_file = os.path.splitext(image_file)[0] + '.json'
            annotation_path = os.path.join(annotation_folder, annotation_file)
            
            if not os.path.exists(annotation_path):
                print(f"Warning: No annotation found for {image_file}")
                continue
            
            image_path = os.path.join(image_folder, image_file)
            
            image = load_image(image_path)
            if image is None:
                continue
                
            annotation = load_annotation(annotation_path)
            if annotation is None:
                continue
            
            if 'text_regions' not in annotation:
                print(f"Warning: No text_regions in annotation {annotation_path}")
                continue
            original_height, original_width = image.shape[:2]
        
            scale_x = image_size[0] / original_width
            scale_y = image_size[1] / original_height
            scaled_regions = []
            for region in annotation['text_regions']:
                if len(region) == 4:
                    x1, y1, x2, y2 = region
                    scaled_regions.append([
                        int(x1 * scale_x),
                        int(y1 * scale_y),
                        int(x2 * scale_x),
                        int(y2 * scale_y)
                    ])
            
            image_resized = cv2.resize(image, image_size)
            
            mask_resized = np.zeros((image_size[1], image_size[0]), dtype=np.uint8)
            
            for region in scaled_regions:
                x1, y1, x2, y2 = region
                x1 = max(0, min(x1, image_size[0]-1))
                y1 = max(0, min(y1, image_size[1]-1))
                x2 = max(0, min(x2, image_size[0]))
                y2 = max(0, min(y2, image_size[1]))
                
                cv2.rectangle(mask_resized, (x1, y1), (x2, y2), 255, -1)
            output_base = f"{source_folder}_{os.path.splitext(image_file)[0]}"
            output_image_path = os.path.join(output_dir, f"{output_base}_image.png")
            output_mask_path = os.path.join(output_dir, f"{output_base}_mask.png")
            
            cv2.imwrite(output_image_path, image_resized)
            cv2.imwrite(output_mask_path, mask_resized)
            
            data_items.append({
                'image_path': output_image_path,
                'mask_path': output_mask_path,
                'source': source_folder,
                'original_image': image_path,
                'original_annotation': annotation_path
            })
            
            processed_count += 1
            
        print(f"  Processed {processed_count} images from {source_folder}")
    
    print(f"Total processed data items: {len(data_items)}")
    return data_items

def split_dataset(data_items, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """
    Split dataset into training, validation, and test sets.
    
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-10, "Ratios must sum to 1"
    
    if not data_items:
        print("Warning: No data items to split")
        return [], [], []
    
    train_val, test = train_test_split(data_items, test_size=test_ratio, random_state=random_state)
    train, val = train_test_split(train_val, test_size=val_ratio/(train_ratio+val_ratio), random_state=random_state)
    
    return train, val, test

def visualize_sample(data_item):
    """
    Visualize a sample (image and mask).
    data_item : Data item containing image and mask paths.
    """
    if not data_item:
        print("Error: No data item to visualize")
        return
        
    if 'image_path' not in data_item or 'mask_path' not in data_item:
        print("Error: Data item missing image_path or mask_path")
        return
        
    if not os.path.exists(data_item['image_path']):
        print(f"Error: Image file not found: {data_item['image_path']}")
        return
        
    if not os.path.exists(data_item['mask_path']):
        print(f"Error: Mask file not found: {data_item['mask_path']}")
        return
    
    try:
        image = cv2.imread(data_item['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(data_item['mask_path'], cv2.IMREAD_GRAYSCALE)
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title(f"Original Image: {os.path.basename(data_item['image_path'])}")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(mask, cmap='gray')
        plt.title('Text Regions Mask')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error visualizing sample: {e}")

def main():
    data_dir = os.path.join('data', 'processed', 'images')
    annotations_dir = os.path.join('data', 'processed', 'annotations')
    output_dir = os.path.join('data', 'prepared')

    data_items = prepare_data(data_dir, annotations_dir, output_dir)
    train_data, val_data, test_data = split_dataset(data_items)
    
    splits = {
        'train': [item['image_path'] for item in train_data],
        'val': [item['image_path'] for item in val_data],
        'test': [item['image_path'] for item in test_data]
    }
    
    with open(os.path.join(output_dir, 'splits.json'), 'w') as f:
        json.dump(splits, f)
    
    print(f"Prepared {len(data_items)} samples")
    print(f"Train: {len(train_data)}, Validation: {len(val_data)}, Test: {len(test_data)}")
    
    if data_items:
        visualize_sample(data_items[0])
    else:
        print("No data items to visualize")

if __name__ == "__main__":
    main()