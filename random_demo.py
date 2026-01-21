import cv2
import torch
import numpy as np
from pathlib import Path
from boxmot import BoostTrack

# Initialize tracker
tracker = BoostTrack(
    reid_weights=Path('osnet_x0_25_msmt17.pt'),
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    half=False
)

# Open video source
cap = cv2.VideoCapture(0)  # 0 for webcam, or 'video.mp4' for file

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Your detection model outputs: M x 6 array (x1, y1, x2, y2, conf, cls)
    # Example: detections = your_detector(frame)
    detections = np.array([
        [100, 100, 200, 300, 0.9, 0],  # x1, y1, x2, y2, confidence, class_id
        [300, 150, 400, 350, 0.85, 0]
    ])
    
    # Update tracker - INPUT: M x (x, y, x, y, conf, cls)
    #                  OUTPUT: M x (x, y, x, y, id, conf, cls, ind)
    tracks = tracker.update(detections, frame)
    
    # Visualize results
    tracker.plot_results(frame, show_trajectories=True)
    
    cv2.imshow('BoxMOT Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()