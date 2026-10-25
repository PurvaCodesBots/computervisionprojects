import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Automatically locate the test image
    image_dir = "./dataset/archive (1)/road_line_images"
    if os.path.exists("./dataset/archive (1)/road_line_images/road_line_images"):
        image_dir = "./dataset/archive (1)/road_line_images/road_line_images"
        
    images = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg', '.JPG'))]
    if not images:
        print(f"Error: No test images found in {image_dir}")
        return
        
    image_path = os.path.join(image_dir, images[0])
    print(f"Testing on image: {images[0]}")
    
    original = cv2.imread(image_path)
    if original is None:
        print(f"Error: Could not load image from {image_path}")
        return
        
    image_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    height, width = image_rgb.shape[:2]

    # 1. Convert to HLS color space to isolate bright white lane dashes accurately
    hls = cv2.cvtColor(original, cv2.COLOR_BGR2HLS)
    
    # Define color range for bright white road lines
    lower_white = np.array([0, 200, 0], dtype=np.uint8)
    upper_white = np.array([180, 255, 255], dtype=np.uint8)
    white_mask = cv2.inRange(hls, lower_white, upper_white)

    # 2. Region of Interest (ROI) Masking to focus strictly on the road surface
    mask = np.zeros_like(white_mask)
    polygons = np.array([[
        (int(width * 0.05), height),             # Bottom-left
        (int(width * 0.4), int(height * 0.55)),  # Top-left
        (int(width * 0.6), int(height * 0.55)),  # Top-right
        (int(width * 0.95), height)              # Bottom-right
    ]], dtype=np.int32)
    
    cv2.fillPoly(mask, polygons, 255)
    masked_white = cv2.bitwise_and(white_mask, mask)

    # 3. Hough Transform on the clean white color mask
    lines = cv2.HoughLinesP(
        masked_white,
        rho=1,
        theta=np.pi/180,
        threshold=20,     # Lower threshold to catch dashed segments easily
        minLineLength=20, # Shorter minimum length to catch individual dashes
        maxLineGap=50     # Allowed gap between dashed segments
    )

    # 4. Draw clean green lines over the original image
    line_image = original.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 4)

    final_output_rgb = cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB)

    # 5. Visualize the improved output
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb)
    plt.title("Original Road Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(final_output_rgb)
    plt.title("Optimized Color-Filtered Lane Detection")
    plt.axis("off")

    plt.show()

if __name__ == "__main__":
    main()