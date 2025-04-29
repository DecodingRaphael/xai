"""Edge detection functions for image analysis."""
import numpy as np
from skimage import color, feature, filters
from typing import List, Optional, Tuple, Dict, Union

from utils.config import EDGE_CONFIG

def detect_edges(
    image: np.ndarray, 
    method: str = 'sobel', 
    weights: Optional[List[float]] = None
) -> np.ndarray:
    """
    Detect edges in an image using various methods.
    
    Parameters:
    -----------
    image : numpy.ndarray
        The image to detect edges in (RGB, values in [0,1])
    method : str
        Edge detection method ('sobel', 'canny', 'laplacian', 'scharr', 'combined')
    weights : list or None
        Weights for combined edge detection [Canny, Sobel, Laplacian, Scharr].
        Only used when method='combined'. If None, default weights are used.
        
    Returns:
    --------
    numpy.ndarray
        Edge map (2D array, values in [0,1])
    """
    # Ensure image is in [0,1] range
    if image.max() > 1.0:
        image = image / 255.0
    
    # Convert to grayscale for edge detection
    gray = color.rgb2gray(image)
    
    # Initialize default weights for combined method
    if weights is None and method == 'combined':
        weights = EDGE_CONFIG["default_weights"]
    
    # Apply the specified edge detection method
    if method == 'canny':
        return _canny_edges(gray)
    elif method == 'sobel':
        return _sobel_edges(gray)
    elif method == 'laplacian':
        return _laplacian_edges(gray)
    elif method == 'scharr':
        return _scharr_edges(gray)
    elif method == 'combined':
        return _combined_edges(gray, weights)
    else:
        raise ValueError(f"Unsupported edge method: {method}")

def _canny_edges(gray: np.ndarray) -> np.ndarray:
    """
    Detect edges using Canny edge detector.
    
    Parameters:
    -----------
    gray : numpy.ndarray
        Grayscale image
        
    Returns:
    --------
    numpy.ndarray
        Normalized edge map
    """
    edges = feature.canny(gray, sigma=1.0)
    # Convert boolean array to float
    edges = edges.astype(float)
    # Normalize to [0,1]
    if edges.max() > 0:
        edges = edges / edges.max()
    return edges

def _sobel_edges(gray: np.ndarray) -> np.ndarray:
    """
    Detect edges using Sobel operator.
    
    Parameters:
    -----------
    gray : numpy.ndarray
        Grayscale image
        
    Returns:
    --------
    numpy.ndarray
        Normalized edge map
    """
    sobelx = filters.sobel_h(gray)
    sobely = filters.sobel_v(gray)
    edges = np.sqrt(sobelx**2 + sobely**2)
    # Normalize to [0,1]
    if edges.max() > 0:
        edges = edges / edges.max()
    return edges

def _laplacian_edges(gray: np.ndarray) -> np.ndarray:
    """
    Detect edges using Laplacian operator.
    
    Parameters:
    -----------
    gray : numpy.ndarray
        Grayscale image
        
    Returns:
    --------
    numpy.ndarray
        Normalized edge map
    """
    edges = np.abs(filters.laplace(gray))
    # Normalize to [0,1]
    if edges.max() > 0:
        edges = edges / edges.max()
    return edges

def _scharr_edges(gray: np.ndarray) -> np.ndarray:
    """
    Detect edges using Scharr operator.
    
    Parameters:
    -----------
    gray : numpy.ndarray
        Grayscale image
        
    Returns:
    --------
    numpy.ndarray
        Normalized edge map
    """
    scharrx = filters.scharr_h(gray)
    scharry = filters.scharr_v(gray)
    edges = np.sqrt(scharrx**2 + scharry**2)
    # Normalize to [0,1]
    if edges.max() > 0:
        edges = edges / edges.max()
    return edges

def _combined_edges(
    gray: np.ndarray, 
    weights: List[float] = None
) -> np.ndarray:
    """
    Combine edges from multiple detection methods.
    
    Parameters:
    -----------
    gray : numpy.ndarray
        Grayscale image
    weights : list
        Weights for [Canny, Sobel, Laplacian, Scharr]
        
    Returns:
    --------
    numpy.ndarray
        Combined edge map
    """
    if weights is None:
        weights = EDGE_CONFIG["default_weights"]
    
    # Get all edge maps individually
    canny_edges = _canny_edges(gray)
    sobel_edges = _sobel_edges(gray)
    laplacian_edges = _laplacian_edges(gray)
    scharr_edges = _scharr_edges(gray)
    
    # Combine using weights
    edges = (
        weights[0] * canny_edges + 
        weights[1] * sobel_edges + 
        weights[2] * laplacian_edges + 
        weights[3] * scharr_edges
    )
    
    # Normalize the combined result
    if edges.max() > 0:
        edges = edges / edges.max()
        
    return edges 