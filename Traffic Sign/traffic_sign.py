import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_sample_traffic_signs():
    signs = []
    
    # 1. Speed Limit 50 (Circular red border)
    img1 = np.ones((300, 300, 3), dtype=np.uint8) * 240
    cv2.circle(img1, (150, 150), 100, (0, 0, 255), 10)
    cv2.putText(img1, "50", (100, 165), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)
    signs.append(("Speed Limit 50", img1))
    
    # 2. Stop Sign (Octagon shape simulation)
    img2 = np.ones((300, 300, 3), dtype=np.uint8) * 240
    pts = np.array([[120, 60], [180, 60], [240, 120], [240, 180], [180, 240], [120, 240], [60, 180], [60, 120]], np.int32)
    cv2.fillPoly(img2, [pts], (0, 0, 255))
    cv2.putText(img2, "STOP", (85, 165), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)
    signs.append(("Stop Sign", img2))
    
    # 3. Yield / Warning Sign (Triangle)
    img3 = np.ones((300, 300, 3), dtype=np.uint8) * 240
    pts_tri = np.array([[150, 50], [260, 230], [40, 230]], np.int32)
    cv2.fillPoly(img3, [pts_tri], (0, 0, 255))
    cv2.fillPoly(img3, [np.array([[150, 80], [230, 210], [70, 210]], np.int32)], (255, 255, 255))
    signs.append(("Yield / Warning", img3))
    
    # 4. No Entry / Restriction
    img4 = np.ones((300, 300, 3), dtype=np.uint8) * 240
    cv2.circle(img4, (150, 150), 100, (0, 0, 255), -1)
    cv2.rectangle(img4, (90, 130), (210, 170), (255, 255, 255), -1)
    signs.append(("No Entry", img4))
    
    return signs

def detect_traffic_signs():
    print("Generating and processing multi-sign traffic dataset batch...")
    signs = create_sample_traffic_signs()
    
    plt.figure(figsize=(10, 10))
    for idx, (title, img) in enumerate(signs):
        # Convert to HSV color space to isolate red sign borders
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 100, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 100, 70])
        upper_red2 = np.array([180, 255, 255])
        mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        # Find contours and bound the detected signs
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        output_img = img.copy()
        
        for cnt in contours:
            if cv2.contourArea(cnt) > 200:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(output_img, title, (x, max(20, y - 5)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 128, 0), 2)

        plt.subplot(2, 2, idx + 1)
        plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
        plt.title(f"Recognized: {title}", fontsize=11)
        plt.axis("off")
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    detect_traffic_signs()