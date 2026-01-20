import warnings
warnings.filterwarnings('ignore')
import cv2, os, shutil
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from boxmot import DeepOcSort, ByteTrack, BotSort, StrongSort, OcSort, HybridSort


def get_video_cfg(path):
    video = cv2.VideoCapture(path)
    size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    return cv2.VideoWriter_fourcc(*'XVID'), size, fps


def counting(image_plot, result):
    box_count = result.boxes.shape[0]
    cv2.putText(image_plot, f'Object Counts:{box_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
    return image_plot


def transform_mot(result):
    mot_result = []
    for i in range(result.boxes.shape[0]):
        mot_result.append(result.boxes.xyxy[i].cpu().detach().cpu().numpy().tolist() + [float(result.boxes.conf[i]),
                                                                                        float(result.boxes.cls[i])])
    return np.array(mot_result)


def display_frame(image_plot, delay=1):
    """
    Display frame in cv2 window and handle exit key.
    
    Args:
        image_plot: Image to display
        delay: Key wait delay in ms
        
    Returns:
        bool: True if user pressed 'q' to quit, False otherwise
    """
    cv2.imshow("BoxMOT Tracking", image_plot)
    key = cv2.waitKey(delay) & 0xFF
    if key == ord('q'):
        cv2.destroyAllWindows()
        return True
    return False


# boxmot                        10.0.57
if __name__ == '__main__':
    # Toggle settings
    SAVE_VIDEO = False  # Set to True to save video, False to only display
    SHOW_WINDOW = True  # Set to True to display cv2 window
    
    output_dir = 'result'
    if SAVE_VIDEO:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
    
    model = YOLO('yolo11s.pt') # select your model.pt path
    
    video_base_path = 'videos'
    for video_path in os.listdir(video_base_path):
        print(f"\n{'='*60}")
        print(f"Processing: {video_path}")
        print(f"{'='*60}")
        
        # tracker = DeepOcSort(
        # reid_weights=Path('osnet_x0_25_msmt17.pt'), # which ReID model to use
        # device='cuda:0',
        # half=False,
        # )
        # tracker = BotSort(
        #     reid_weights=Path('osnet_x0_25_msmt17.pt'), # which ReID model to use
        #     device='cuda:0',
        #     half=False,
        # )
        # tracker = StrongSort(
        #     reid_weights=Path('osnet_x0_25_msmt17.pt'), # which ReID model to use
        #     device='cuda:0',
        #     half=False,
        # )
        # tracker = HybridSort(
        #     reid_weights=Path('osnet_x0_25_msmt17.pt'), # which ReID model to use
        #     device='cuda:0',
        #     half=False,
        #     det_thresh=0.3,
        # )
        tracker = ByteTrack()
        # tracker = OcSort()

        # Initialize video writer only if saving is enabled
        video_output = None
        if SAVE_VIDEO:
            fourcc, size, fps = get_video_cfg(f'{video_base_path}/{video_path}')
            video_output = cv2.VideoWriter(f'{output_dir}/{video_path}', fourcc, fps, size)
        
        should_quit = False
        for result in model.predict(source=f'{video_base_path}/{video_path}',
                      stream=True,
                      imgsz=640,
                      save=False,
                      # conf=0.2,
                      # classes=1
                      ):
            image_plot = result.orig_img
            mot_input = transform_mot(result)
            try:
                tracker.update(mot_input, image_plot)
                tracker.plot_results(image_plot, show_trajectories=True)
            except:
                continue
            counting(image_plot, result)
            
            # Save to video if enabled
            if SAVE_VIDEO and video_output is not None:
                video_output.write(image_plot)
            
            # Display in cv2 window if enabled
            if SHOW_WINDOW:
                should_quit = display_frame(image_plot, delay=1)
                if should_quit:
                    print(f"User interrupted playback of {video_path}")
                    break
        
        # Release video writer if it was used
        if video_output is not None:
            video_output.release()
            print(f"✓ Video saved: {output_dir}/{video_path}")
        
        if should_quit:
            break
    
    cv2.destroyAllWindows()
    print("\n✓ Processing complete!")
