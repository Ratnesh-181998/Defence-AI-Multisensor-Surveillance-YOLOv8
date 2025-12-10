import cv2
import numpy as np
import os

def generate_assets():
    os.makedirs("examples/images", exist_ok=True)
    os.makedirs("examples/videos", exist_ok=True)

    # 1. Generate Detection Example Image
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw background gradient
    for i in range(480):
        img[i, :] = (int(i/2), 0, int(60 + i/4)) # Dark blue/purple gradient
    
    # Draw a "Drone view" grid
    cv2.line(img, (320, 0), (320, 480), (0, 255, 0), 1)
    cv2.line(img, (0, 240), (640, 240), (0, 255, 0), 1)
    
    # Draw target box
    cv2.rectangle(img, (280, 200), (360, 280), (0, 255, 255), 2)
    cv2.putText(img, "TARGET LOCKED", (260, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Save image
    cv2.imwrite("examples/images/detection_example.jpg", img)
    print("Generated examples/images/detection_example.jpg")

    # 2. Generate Tracking Example
    # For now just copy the detection one or make another
    cv2.imwrite("examples/images/tracking_example.jpg", img)

if __name__ == "__main__":
    generate_assets()
