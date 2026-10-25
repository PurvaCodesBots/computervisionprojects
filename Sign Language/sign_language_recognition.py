import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    csv_path = "./dataset/archive (2)/sign_mnist_train.csv"
    if not os.path.exists(csv_path):
        print(f"Error: Could not find dataset at {csv_path}")
        return
        
    print("Loading Sign MNIST dataset from CSV...")
    df = pd.read_csv(csv_path)
    
    labels = df.iloc[:, 0].values
    pixels = df.iloc[:, 1:].values
    
    alphabet_mapping = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 
                        'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

    plt.figure(figsize=(10, 10))
    
    for i in range(16):
        label_idx = labels[i]
        mapped_idx = label_idx if label_idx < 9 else label_idx - 1
        letter = alphabet_mapping[mapped_idx] if mapped_idx < len(alphabet_mapping) else "?"
        
        img = pixels[i].reshape(28, 28).astype(np.uint8)
        
        plt.subplot(4, 4, i + 1)
        # interpolation='bicubic' smooths out the pixel edges
        plt.imshow(img, cmap='gray', interpolation='bicubic')
        plt.title(f"Label: {label_idx} -> '{letter}'", fontsize=10)
        plt.axis("off")
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()