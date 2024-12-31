import numpy as np
import matplotlib.pyplot as plt
import Indicators
import cv2 as cv
import math
import importlib
from itertools import combinations

def generate_rgb_images(images, combinaciones, case, i_index, serie_index):
    """
    Replace RGB image components with mathematical parameters that represent temporal information

    Args:
        array (np.ndarray): A NumPy array of shape (T, N, M) representing the sequence of images. 
        T represents the temporal axis.

    Returns:
        np.ndarray: A 3D array of shape (N, M, 3) representing the RGB images.
    """
    # Import the first script as a module
    script_Indicators = importlib.import_module('Indicators')

    for d_index, combinacion in enumerate(combinaciones):
        image_converted = np.zeros((640, 640, 3), dtype=np.uint8)
        for i, funcion_nombre in enumerate(combinacion):
            funcion = getattr(script_Indicators, funcion_nombre)
            # Replace R, G, and B components with temporal information
            image_converted[:, :, i] = funcion(images) # component
        # Store frame
        frame_filename = f'datasets/dataset_{d_index}/images/{case}/image_{i_index}_{serie_index}.jpg'
        cv.imwrite(frame_filename, image_converted)
        # Store label
        with open(f'data/{i_index}/tx/frame_0.txt', 'r') as origin_path:
            content = origin_path.read()

        with open(f'datasets/dataset_{d_index}/labels/{case}/image_{i_index}_{serie_index}.txt', 'w') as destin_path:
            destin_path.write(content)


def extract_info(txt_file_directory):
    """
    Extract information from a text file containing pixel positions and FPS data.

    Args:
        txt_file_directory (str): The directory and filename of the text file.

    Returns:
        transmitters (list): A list of (x, y) coordinates of selected transmitters.
        non_transmitters (list): A list of (x, y) coordinates of selected non-transmitters.
        fps (float): Frames per second of the video.
    """
    transmitters = []
    fps = None
    
    with open(txt_file_directory, 'r') as file:
        # Read line by line
        for line in file:
            # Split line into parts based on commas
            parts = line.strip().split(', ')
            if parts[0] == "T":
                    transmitters.append((int(parts[1]), int(parts[2])))
            elif parts[0] == "FPS":
                    fps = float(parts[1])
    
    return transmitters, fps

def main():
    """
    Get images from directory 
    Convert from RGB to mathematical parameters
    Store images _video_last_index

    Get and store labels in the database

    7 - 500: train
    500 - 630: val
    0 - 7 && 630 - 647: inference
    """
    # Filter only desired functions
    functions = ['percentil_difference', 'max_peek_dif', 'mean', 'median', 'norme', 'std']

    # Get combinations of three
    combinations = list(combinations(functions, 3))
    
    num_combinations = len(combinations)
    print(f"There are {num_combinations} combinations of three functions.")

    for i in range(500, 630): 
        try:
            # Get info of fps and transmitter positions
            _, fps = extract_info(f"data/{i}/info.txt")

            # Random init image for each directory (videos always start with the header of the data frame)
            images = []
            serie_index = 2
            init = int(np.random.randint((serie_index * 2) * fps, (((serie_index * 2) * fps) + 2*fps)))
            steps = int(math.trunc(fps/15))
            end = int(init + 2*fps + 1)
            # Extract and convert all images in a data frame to grayscale
            for count in range(init, end, steps): # max 2 images per bit
                image = cv.imread("data/%d/tx/frame_%d.jpg" % (i, count))
                gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)                     
                images.append(gray) 
            print(f'IMAGE --> {np.shape(image)}')
            # Convert from RGB to mathematical parameters
            print(f'IMAGES --> {np.shape(images)}')
            case = 'val'
            index = i
            generate_rgb_images(images, combinations, case, index, serie_index)
            
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        

if __name__ == "__main__":
    main()
