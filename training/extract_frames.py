import cv2
import os
import random

def extract_frames(video_path, output_folder, target_count=100):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps
    
    print(f"Video loaded: {duration:.2f} seconds, {total_frames} frames.")

    # Calculate the interval to get exactly target_count frames spread evenly
    # We add a small buffer to avoid end-of-video issues
    interval = total_frames // target_count
    
    saved_count = 0
    current_frame = 0

    while saved_count < target_count and current_frame < total_frames:
        # Set the position of the video
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()

        if ret:
            # Construct filename
            filename = f"{output_folder}/frame_{saved_count:03d}.jpg"
            
            # Save the frame
            cv2.imwrite(filename, frame)
            saved_count += 1
            
            # Move to next interval
            current_frame += interval
        else:
            break

    cap.release()
    print(f"Successfully extracted {saved_count} images to '{output_folder}/'")

# Run the extraction
extract_frames('test5.mp4', 'scrabble_dataset_raw')