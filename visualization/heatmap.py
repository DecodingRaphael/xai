"""Heatmap visualization functions for XAI analysis."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from pathlib import Path
from typing import Optional, Dict, Tuple, Union, List

from utils.config import VIZ_CONFIG

def plot_image_heatmap(
    heatmap: np.ndarray, 
    image: np.ndarray, 
    output_filename: Optional[Union[str, Path]] = None, 
    heatmap_cmap: str = 'jet', 
    show_plot: bool = True, 
    alpha: float = 0.5,
    title: Optional[str] = None
) -> None:
    """
    Plot an image with a heatmap overlay.
    
    Parameters:
    -----------
    heatmap : numpy.ndarray
        The heatmap to overlay
    image : numpy.ndarray
        The original image
    output_filename : str or Path, optional
        If provided, save the figure to this path
    heatmap_cmap : str
        Colormap for the heatmap
    show_plot : bool
        Whether to display the plot
    alpha : float
        Opacity of the heatmap overlay
    title : str, optional
        Title for the plot
    """
    plt.figure(figsize=(VIZ_CONFIG["fig_width"], VIZ_CONFIG["fig_height"]))
    plt.imshow(image)
    plt.imshow(heatmap, cmap=heatmap_cmap, alpha=alpha)
    plt.colorbar(label=VIZ_CONFIG["colorbar_label"])
    
    if title:
        plt.title(title)
        
    plt.axis('off')
    plt.tight_layout()
    
    if output_filename:
        plt.savefig(output_filename, dpi=VIZ_CONFIG["dpi"], bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close()

def plot_difference_map(
    image: np.ndarray,
    diff_map: np.ndarray,
    output_filename: Optional[Union[str, Path]] = None,
    cmap: str = 'RdBu_r',
    alpha: float = 0.7,
    title: str = 'Difference Map (Red = Raphael, Blue = Non-Raphael)',
    show_plot: bool = True
) -> None:
    """
    Plot a difference map between two classes.
    
    Parameters:
    -----------
    image : numpy.ndarray
        The original image
    diff_map : numpy.ndarray
        The difference map (class1 - class2)
    output_filename : str or Path, optional
        If provided, save the figure to this path
    cmap : str
        Colormap for the difference map
    alpha : float
        Opacity of the difference map overlay
    title : str
        Title for the plot
    show_plot : bool
        Whether to display the plot
    """
    # Scale for better visualization
    abs_max = np.max(np.abs(diff_map))
    
    plt.figure(figsize=(VIZ_CONFIG["fig_width"], VIZ_CONFIG["fig_height"]))
    plt.imshow(image)
    plt.imshow(diff_map, cmap=cmap, alpha=alpha, vmin=-abs_max, vmax=abs_max)
    plt.colorbar(label='Raphael - Non-Raphael')
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    
    if output_filename:
        plt.savefig(output_filename, dpi=VIZ_CONFIG["dpi"], bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close()

def plot_side_by_side(
    image: np.ndarray,
    maps: Dict[str, np.ndarray],
    output_filename: Optional[Union[str, Path]] = None,
    cmaps: Optional[Dict[str, str]] = None,
    alphas: Optional[Dict[str, float]] = None,
    titles: Optional[Dict[str, str]] = None,
    main_title: Optional[str] = None,
    show_plot: bool = True
) -> None:
    """
    Plot multiple heatmaps side by side.
    
    Parameters:
    -----------
    image : numpy.ndarray
        The original image
    maps : Dict[str, np.ndarray]
        Dictionary of heatmaps to display
    output_filename : str or Path, optional
        If provided, save the figure to this path
    cmaps : Dict[str, str], optional
        Dictionary of colormaps for each heatmap
    alphas : Dict[str, float], optional
        Dictionary of opacity values for each heatmap
    titles : Dict[str, str], optional
        Dictionary of titles for each subplot
    main_title : str, optional
        Main title for the figure
    show_plot : bool
        Whether to display the plot
    """
    n_maps = len(maps)
    if n_maps == 0:
        return
    
    # Set up defaults
    if cmaps is None:
        cmaps = {key: 'jet' for key in maps}
    if alphas is None:
        alphas = {key: 0.5 for key in maps}
    if titles is None:
        titles = {key: key for key in maps}
    
    fig, axes = plt.subplots(1, n_maps, figsize=(VIZ_CONFIG["fig_width"] * n_maps // 2, VIZ_CONFIG["fig_height"]))
    
    # Handle case with only one map
    if n_maps == 1:
        axes = [axes]
    
    for ax, (key, heatmap) in zip(axes, maps.items()):
        ax.imshow(image)
        im = ax.imshow(heatmap, cmap=cmaps.get(key, 'jet'), alpha=alphas.get(key, 0.5))
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(titles.get(key, key))
        ax.axis('off')
    
    if main_title:
        fig.suptitle(main_title, fontsize=16)
        
    plt.tight_layout()
    
    if output_filename:
        plt.savefig(output_filename, dpi=VIZ_CONFIG["dpi"], bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close()

def create_confidence_map(
    mean_map: np.ndarray,
    std_map: np.ndarray
) -> np.ndarray:
    """
    Create a confidence map from mean and std maps.
    
    Parameters:
    -----------
    mean_map : numpy.ndarray
        The mean heatmap
    std_map : numpy.ndarray
        The standard deviation heatmap
        
    Returns:
    --------
    numpy.ndarray
        Confidence map (high relevance AND low variability)
    """
    # Normalize std map to [0,1]
    norm_std = std_map / (np.max(std_map) + 1e-10)
    
    # High confidence = high relevance AND low variability
    confidence_map = mean_map * (1 - norm_std)
    
    return confidence_map 