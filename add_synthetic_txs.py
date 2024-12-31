
"""
Extract Info From Real Life Video Script

"""

import cv2
import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt
import os
from moviepy.editor import VideoFileClip

bit_duration = 0.13333  # Bit duration in seconds
frame_duration = 2  # Frame duration in seconds
points = []


def get_info(frame):
    print("Select transmitter points on the image by clicking. Press Enter to finish.")
    img = frame.copy()
    cv2.imshow("Image", img)

    global points
    points = []
    cv2.setMouseCallback("Image", click_event)

    while True:
        key = cv2.waitKey(1)
        if key == 13:  # Press "Enter" to finish the selection
            break

    cv2.destroyWindow("Image")
    tx_centers = np.array(points)

    radius = []
    energy = []

    for center in tx_centers:
        x = input(f"Enter the radius for the transmitter at {center} (or press Enter to finish): ")
        if not x:
            break
        radius.append(int(x))

    for center in tx_centers:
        x = input(f"Enter the energy value for the transmitter at {center} (or press Enter to finish): ")
        if not x:
            break
        energy.append(int(x))

    return tx_centers, radius, energy

def click_event(event, x, y, flags, param):
    """
    Handle mouse click events for selecting points.

    Args:
        event (int): The type of mouse event (e.g., cv2.EVENT_LBUTTONDOWN).
        x (int): The x-coordinate of the clicked point.
        y (int): The y-coordinate of the clicked point.
        flags (int): Additional flags for the mouse event.
        param (object): Additional parameter passed to the function.

    Returns:
        None
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])  
   
def create_YoLo_file(transmitters, radius, width, height, output_file):
    """
    Get and store transmitters boxes in YoLo format.
    
    Args:
        transmitters(np.array): List of (x,y) coordinates of selected transmitters.
        radius(np.array): Wrotten radius for each transmitter.
        width(int): Width of the image.
        height(int): Height of the image.
        output_file(str): Name of the output file.

    Returns:
        None
    """
    with open(output_file, 'w') as file:
        for i in range(len(transmitters)):
            x_center = transmitters[i][0] / width  # Normalizing x-coordinate
            y_center = transmitters[i][1] / height  # Normalizing y-coordinate
            box_width = radius[i] / width  # Normalizing box width
            box_height = radius[i] / height  # Normalizing box height
            file.write(f"0 {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")

def store_pixels(transmitters, fps, output_file):
    """
    Store pixels positions of selected transmitters and non transmitters
    
    Args:
        transmitters (np.array): List of (x,y) coordinates of selected transmitters.
        non_transmitters (np.array): List of (x,y) coordinates of selected non transmitters.
        fps (float): Frames per second of the video
        output_file (str): The name of the output file to store the data.

    Returns:
        None
    """
    with open(output_file, 'w') as file:
        for coordinates in transmitters:
            file.write(f"T, {coordinates[0]}, {coordinates[1]}\n")
        file.write(f"FPS, {fps}\n")



def main():
    """
    Access each of the videos in each directory and extract its images and 
    main parameters (FPS, duration, number of images).

    Displays the first image of each video and asks for the position of 
    transmitters and non-transmitters, size of each light region.

    Generates and store random data frames for each transmitter and modifies  
    the images to represent those data frames in the specified positions and sizes.
    """
    # Loop through directories
    for i in range(6, 7):
        frame = cv2.imread("data/%d/tx/frame_%d.jpg" % (i,1))
        tx_centers, radius, energy = get_info(frame)
        fps = 25.0
        store_pixels(tx_centers, fps, f'data/{i}/info.txt')
        frame_height, frame_width, _ = frame.shape
        print(frame.shape)

        # Create YoLo - format txt file
        for j in range(0, 1795):
            txt_filename = f'data/{i}/tx/frame_{j}.txt'
            create_YoLo_file(tx_centers, radius, frame_width, frame_height, txt_filename)
