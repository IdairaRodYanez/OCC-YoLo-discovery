"""
Indicators library

In this library, a set of indicator functions are defined in order to be called at the pre-processing stage of the
discovery algorithm training engine.

All the functions should receive a chunk of size LxNxM and return just a 2D tensor (N, M). The main idea is collapsing
the time-dimension in order to feed the result to a discovery algorithm.
"""
import numpy as np
from scipy.stats import skew as sk, kurtosis as ku

def mean (array):
    """
    Calculate the mean of optical signals for multiple pixels.

    Args:
        array (np.ndarray): A NumPy array of shape (T, N, M) representing the sequence of images. 
        T represents temportal axis.

    Returns:
        np.ndarray: A 2D array of shape (M, 1), containing the mean optical signal value for each pixel.
    """
    return np.mean(array, axis=0)

def median (array):
    """
    Calculate the median of optical signals for multiple pixels.

    Args:
        array (np.ndarray): A NumPy array of shape (T, N, M) representing the sequence of images. 
        T represents temportal axis..

    Returns:
        np.ndarray: A 2D array of shape (M, 1), containing the median optical signal value for each pixel.
    """
    return np.median(array, axis=0)

def var(array):
    """
    Calculate the moment of third order of optical signals for multiple pixels.

   Args:
        array (np.ndarray): A NumPy array of shape (T, N, M) representing the sequence of images. 
        T represents temportal axis.

    Returns:
        np.ndarray: A 2D array of shape (M, 1) containing the third moment values.
    """
    
    return np.var(array, axis=0)

def std(array):
    """
    Calculate the standard deviation of optical signals for multiple pixels.

    Args:
        array (np.ndarray): A NumPy array of shape (T, N, M) representing the sequence of images. 
        T represents temportal axis.

    Returns:
        np.ndarray: A 2D array of shape (M, 1), containing the standard deviation of optical signals 
        for each pixel.
    """
    return np.std(array, axis=0)

def moment_3(array):
    """
    Calculate the moment of third order of optical signals for multiple pixels.

    Args:
        array (np.ndarray): A NumPy array of shape (T, N, M) representing the sequence of images. 
        T represents temportal axis.

    Returns:
        np.ndarray: A 2D array of shape (M, 1) containing the third moment values.
    """
    # Calculate the mean (first moment) along axis=2 for each pixel
    mean = np.mean(array, axis=0, keepdims=True)

    # Calculate the third-order moment for each pixel
    moment_3 = np.mean((array - mean)**3, axis=0)

    return moment_3

def max_peek_dif(array):
    """
    Calculate the maximum peak difference of optical signals for multiple pixels.

    Args:
        array (np.ndarray): A NumPy array of shape (T, N, M) representing the sequence of images. 
        T represents temportal axis.

    Returns:
        np.ndarray: A 2D array of shape (M, 1), containing the maximum peak difference of optical 
        signals for each pixel.
    """
    return (np.max(array, axis=0) - np.min(array, axis=0))


def kurtosis(array):
    """
    Calculate the kurtosis of optical signals for multiple pixels.

    Args:
        array (np.ndarray): A NumPy array of shape (T, N, M) representing the sequence of images. 
        T represents temportal axis.

    Returns:
        np.ndarray: A 2D array of shape (M, 1) containing the kurtosis values.
    """
    curtosis = ku(array, axis=0) 

    return curtosis

def skewness(array):
    """
    Calculate the skewness of optical signals for multiple pixels.

    Args:
        array (np.ndarray): A NumPy array of shape (T, N, M) representing the sequence of images. 
        T represents temportal axis.

    Returns:
        np.ndarray: A 2D array of shape (M, 1) containing the skewness values.
    """
    skewness = sk(array, axis=0)

    return skewness

def norme(array):
    """
    Calculate the norm (magnitude) of optical signals for multiple pixels.

    Args:
        array (np.ndarray): A NumPy array of shape (T, N, M) representing the sequence of images. 
        T represents temportal axis.

    Returns:
        np.ndarray: A 2D array of shape (M, 1) containing the norm values of the vectors.
    """
    norm_values = np.linalg.norm(array, axis=0)
    
    return norm_values 

def norme_05(array):
    """
    Calculate the norm (magnitude) of optical signals for multiple pixels.

    Args:
        array (np.ndarray): A NumPy array of shape (T, N, M) representing the sequence of images. 
        T represents temportal axis.

    Returns:
        np.ndarray: A 2D array of shape (M, 1) containing the norm values of the vectors.
    """
    norm_values = np.power(np.sum(np.power(array, 0.5), axis=0), 1/0.5)
    
    return norm_values 


def inter_quartil_distance(array):
    """
    Calculate the specified percentile of optical signals for multiple pixels.

    Args:
        array (np.ndarray): A NumPy array of shape (T, N, M) representing the sequence of images. 
        T represents temportal axis.

    Returns:
        np.ndarray: A 2D array of shape (M, 1) containing the distance inter quartil values of the vectors.
    """
    percentil_25 = np.percentile(array, 25, axis=0)
    percentil_50 = np.percentile(array, 50, axis=0)
    percentil_75 = np.percentile(array, 75, axis=0)

    distance_inter_quartil = (percentil_75 - percentil_25) / percentil_50

    return distance_inter_quartil

def percentil_difference(array):
    """
    Calculate the specified percentile of optical signals for multiple pixels.

    Args:
        array (np.ndarray): A NumPy array of shape (T, N, M) representing the sequence of images. 
        T represents temportal axis.

    Returns:
        np.ndarray: A 2D array of shape (M, 1) containing the distance inter quartil values of the vectors.
    """
    percentil_25 = np.percentile(array, 25, axis=0)
    percentil_75 = np.percentile(array, 75, axis=0)

    distance_inter_quartil = (percentil_75 - percentil_25) 

    return distance_inter_quartil


def percentil(array):
    """
    Calculate the specified percentile of optical signals for multiple pixels.

    Args:
        array (np.ndarray): A NumPy array of shape (T, N, M) representing the sequence of images. 
        T represents temportal axis.

    Returns:
        np.ndarray: A 2D array of shape (M, 1) containing the percentil values of the vectors.
    """
    
    percentile_values = np.percentile(array, 75, axis=0)

    return percentile_values
    

def standardization(array):
    """
    Calculates the z-score for each pixel in the last image of the sequence.

    Args:
        array (np.ndarray): A NumPy array of shape (T, N, M) representing the sequence of images. 
        T represents temportal axis.

    Returns:
        np.ndarray: A NumPy array with the z-score standardization of the pixels in the last image.
    """ 
    last_image = array[-1, ...]
    mean_value = np.mean(array)
    std_value = np.std(array)

    return (last_image - mean_value) / (std_value + 1e-10)

