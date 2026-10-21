"""Visualization functions for edge detection."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from pathlib import Path
from typing import List, Optional, Dict, Union

from utils.config import EDGE_CONFIG, VIZ_CONFIG
from edge_detection.detector import detect_edges

def visualize_edge_heatmap_overlay(
    image: np.ndarray,
    heatmap: np.ndarray,
    output_path: Union[str, Path],
    title: str = "Edge-Enhanced RISE Map",
    edge_method: str = 'combined',
    edge_weights: Optional[List[float]] = None,
    edge_alpha: float = None,
    heatmap_alpha: float = None,
    edge_color: str = None,
    heatmap_cmap: str = None,
    show_plot: bool = False
) -> str:
    """
    Create visualization overlaying RISE heatmaps with edge detection maps.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Original image (RGB format, values in [0,1])
    heatmap : numpy.ndarray
        RISE relevance map
    output_path : str or Path
        Path to save the visualization
    title : str
        Title for the plot
    edge_method : str
        Edge detection method ('sobel', 'canny', 'laplacian', 'scharr', 'combined')
    edge_weights : list or None
        Weights for combined edge detection. Only used when edge_method='combined'.
    edge_alpha : float
        Opacity of edge overlay (0-1). If None, use config default.
    heatmap_alpha : float
        Opacity of heatmap overlay (0-1). If None, use config default.
    edge_color : str
        Color for edge highlighting. If None, use config default.
    heatmap_cmap : str
        Colormap for heatmap. If None, use config default.
    show_plot : bool
        Whether to display the plot
        
    Returns:
    --------
    str
        Path to the saved combined visualization
    """
    # Use default values from config if not provided
    edge_alpha = edge_alpha if edge_alpha is not None else EDGE_CONFIG["edge_alpha"]
    heatmap_alpha = heatmap_alpha if heatmap_alpha is not None else EDGE_CONFIG["heatmap_alpha"]
    edge_color = edge_color if edge_color is not None else EDGE_CONFIG["edge_color"]
    heatmap_cmap = heatmap_cmap if heatmap_cmap is not None else EDGE_CONFIG["heatmap_cmap"]
    
    # Get edge map
    edges = detect_edges(image, method=edge_method, weights=edge_weights)
    
    # Normalize heatmap
    heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-10)
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # First subplot: Standard RISE heatmap visualization
    ax1.imshow(image)
    im1 = ax1.imshow(heatmap, cmap=heatmap_cmap, alpha=heatmap_alpha)
    ax1.set_title("Standard RISE Heatmap")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    ax1.axis('off')
    
    # Second subplot: Edge-enhanced visualization
    ax2.imshow(image)
    im2 = ax2.imshow(heatmap, cmap=heatmap_cmap, alpha=heatmap_alpha)
    
    # Create a mask of edges above a threshold (only show strong edges)
    edge_threshold = EDGE_CONFIG["edge_threshold"]
    edge_mask = edges > edge_threshold
    
    # Only show edges in regions with significant relevance
    heatmap_threshold = EDGE_CONFIG["heatmap_threshold"]
    combined_mask = edge_mask & (heatmap_norm > heatmap_threshold)
    
    # Convert mask to RGB for overlay
    edge_overlay = np.zeros((*combined_mask.shape, 4))  # RGBA
    edge_overlay[combined_mask, :3] = matplotlib.colors.to_rgb(edge_color)  # RGB for the edge color
    edge_overlay[combined_mask, 3] = edge_alpha  # Alpha channel
    
    # Overlay edges on second subplot
    ax2.imshow(edge_overlay)
    
    # Update the title to reflect the edge method used
    if edge_method == 'combined':
        method_title = "Combined Edges (Canny, Sobel, Laplacian, Scharr)"
    else:
        method_title = f"{edge_method.capitalize()} Edges"
    
    ax2.set_title(f"Edge-Enhanced RISE Map ({method_title})")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    ax2.axis('off')
    
    # Add an overall title
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_path, dpi=VIZ_CONFIG["dpi"], bbox_inches='tight')
    if not show_plot:
        plt.close(fig)
    
    # Create a single image with the combined visualization
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.imshow(heatmap, cmap=heatmap_cmap, alpha=heatmap_alpha)
    
    # Create a 3-channel overlay to highlight edges in areas of high relevance
    edge_highlight = np.zeros((*edges.shape, 3))  # RGB
    
    # Scale edges by heatmap intensity
    weighted_edges = edges * heatmap_norm
    weighted_edges = (weighted_edges - weighted_edges.min()) / (weighted_edges.max() - weighted_edges.min() + 1e-10)
    
    # Apply a threshold to reduce noise
    important_edges = weighted_edges > 0.2
    edge_highlight[important_edges] = matplotlib.colors.to_rgb(edge_color)
    
    # Scale the brightness by the edge importance
    for i in range(3):
        edge_highlight[:, :, i] *= weighted_edges
    
    plt.imshow(edge_highlight, alpha=edge_alpha)
    
    # Update the title for the combined visualization
    if edge_method == 'combined':
        method_text = "Combined Edge Detection (Canny, Sobel, Laplacian, Scharr)"
        if edge_weights:
            weight_text = f" [Weights: C={edge_weights[0]}, S={edge_weights[1]}, L={edge_weights[2]}, Sc={edge_weights[3]}]"
            method_text += weight_text
    else:
        method_text = f"{edge_method.capitalize()} Edge Detection"
    
    plt.title(f"Brushstroke Analysis: RISE Relevance + {method_text}")
    plt.axis('off')
    plt.tight_layout()
    
    # Save the combined single visualization
    combined_path = str(output_path).replace('.png', '_combined.png')
    plt.savefig(combined_path, dpi=VIZ_CONFIG["dpi"], bbox_inches='tight')
    if not show_plot:
        plt.close()
    
    return combined_path 