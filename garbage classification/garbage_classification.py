import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_garbage_dataset():
    # Point to the nested dataset archive path visible in your VS Code explorer
    dataset_path = "./dataset/archive (4)"
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path '{dataset_path}' not found.")
        return

    print("Scanning garbage classification dataset directories...")
    
    image_paths = []
    labels = []
    
    # Walk through folders to find category subdirectories (e.g., cardboard, glass, plastic, etc.)
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))
                # Extract category name from the parent folder name
                category = os.path.basename(root)
                labels.append(category)

    if not image_paths:
        print("No image files found in the archive folders. Please check the dataset path.")
        return

    print(f"Successfully discovered {len(image_paths)} garbage sample images across categories!")

    # Select up to 16 samples to plot in a grid
    sample_indices = np.random.choice(len(image_paths), min(16, len(image_paths)), replace=False)
    
    plt.figure(figsize=(12, 12))
    for i, idx in enumerate(sample_indices):
        img_path = image_paths[idx]
        label = labels[idx]
        
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, (150, 150))
        
        plt.subplot(4, 4, i + 1)
        plt.imshow(img_resized)
        plt.title(f"Class: {label}", fontsize=9)
        plt.axis("off")
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    load_garbage_dataset()