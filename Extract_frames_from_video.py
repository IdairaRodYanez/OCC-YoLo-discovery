"""
Extract Frames from Videos Script

This script is used to extract all the images from each video of the dataset and store them without modifications.
"""

import cv2
import os

def main():

    # Loop through directories
    for i in range(12, 80):
        try: 
            input_path = f'data/{i}' 
            input_video_file = os.path.join(input_path, 'video.mp4')
            
            # Open the video file
            cap = cv2.VideoCapture(input_video_file)
            
            # Get video properties
            total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Control parameters for progress
            frame_number = 0

            while frame_number < total_video_frames:
                # Extract frames from the video
                ret, frame = cap.read()
                if not ret:
                    break
                # Store frame
                frame_filename = f'data/{i}/notx/frame_{frame_number}.jpg'
                cv2.imwrite(frame_filename, frame)
                
                frame_number += 1

            # Release objects
            cap.release()
        except (FileNotFoundError, ZeroDivisionError):
            print(f'video {i} no existe')

        

if __name__ == "__main__":
    main()
