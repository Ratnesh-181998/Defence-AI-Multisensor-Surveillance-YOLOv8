import cv2
import numpy as np
import os

def generate_thermal_video():
    os.makedirs("examples/videos", exist_ok=True)
    output_path = "examples/videos/thermal_sample.mp4"
    
    width, height = 640, 512
    fps = 30
    seconds = 10
    
    # Use 'mp4v' or 'avc1' for Windows
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print("Failed to create video writer")
        return

    frames = fps * seconds
    for i in range(frames):
        # Create a "thermal" look (grayscale/jet)
        # Background
        frame = np.ones((height, width, 3), dtype=np.uint8) * 30 
        
        # Moving "hot" object
        x = int(width/2 + 200 * np.sin(i * 0.05))
        y = int(height/2 + 100 * np.cos(i * 0.03))
        
        # Draw hot spot (White centers, fading out)
        # Simple simulated heat blob
        cv2.circle(frame, (x, y), 50, (255, 255, 255), -1)
        cv2.circle(frame, (x, y), 60, (200, 200, 200), 2)
        cv2.circle(frame, (x, y), 80, (100, 100, 100), 2)
        
        # Apply pseudo-color map to look like thermal
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_thermal = cv2.applyColorMap(frame_gray, cv2.COLORMAP_JET)
        
        cv2.putText(frame_thermal, f"SIMULATED THERMAL FEED {i}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        
        out.write(frame_thermal)
        
    out.release()
    print(f"Generated {output_path}")

if __name__ == "__main__":
    generate_thermal_video()
