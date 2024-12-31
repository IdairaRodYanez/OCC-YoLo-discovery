import os
from moviepy.editor import VideoFileClip
import numpy as np
import os

# Get the path of the current directory where the script is located
script_directory = os.path.dirname(os.path.abspath(__file__))
destin_videos_directory = os.path.join(script_directory, "data")

# Directory where the videos are located (inside the script's directory)
videos_directory = os.path.join(script_directory, "github_videos/copy")

# List all the files in the directory and subdirectories
videos = []
for root, dirs, files in os.walk(videos_directory):
    for file in files:
        if file.endswith(".jpg"):  # Filter video files, adjust the extension if needed
            videos.append(os.path.join(root, file))
   
# Play each video
for video in videos:
    clip = VideoFileClip(video).subclip(0,7)
    duration = clip.duration
    # Check if the duration is less than 5 seconds
    if duration < 5:
        print(f"Video deleted.")
        os.remove(video)
    else:
        clip.preview()
        clip.close()
        print(video)
        if "casijdda" in video:
            print("The string contains the word 'casia'.")
            response = "yes"
        else:
            # Ask if they want to keep the video
            response = "yes"
        
        if response.lower() == "yes":
            print("added")
            # Create a new folder with an index equal to the number of folders + 1
            new_folder_1 = os.path.join("data", f"{len(os.listdir(destin_videos_directory)) + 1}")
            new_folder_2 = os.path.join(f"data/{len(os.listdir(destin_videos_directory)) + 1}", "tx")
            new_folder_3 = os.path.join(f"data/{len(os.listdir(destin_videos_directory)) + 1}", "notx")
            os.makedirs(new_folder_1)
            os.makedirs(new_folder_2)
            os.makedirs(new_folder_3)
            
            # Copy the video to the new folder
            video_name = "video.mp4"
            destination_path = os.path.join(new_folder_1, video_name)
            os.replace(video, destination_path)
        elif response.lower() == "no":
            print("Deleted")
            os.remove(video)
        else:
            print("Invalid response. The video was neither added nor deleted.")
