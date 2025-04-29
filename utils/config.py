"""Configuration parameters for the XAI system."""

# RISE parameters
RISE_CONFIG = {
    "n_masks": 100,         # Number of masks to generate (increased for more stable results)
    "p_keep": 0.5,           # Probability of keeping pixels in masks (balanced for better coverage)
    "feature_res": 8,         # Resolution of low-res mask before upsampling (increased for finer detail)
    "runs": 5                 # Number of runs for stability (increased for more reliable results)
}

# Edge detection parameters
EDGE_CONFIG = {
    # Default weights for combined edge detection [Canny, Sobel, Laplacian, Scharr]
    "default_weights": [0.2, 0.3, 0.2, 0.3],

    # Edge detection methods available
    "methods": ["sobel", "canny", "laplacian", "scharr", "combined"],

    # Edge visualization parameters
    "edge_alpha": 0.7,        # Opacity of edge overlay
    "heatmap_alpha": 0.6,     # Opacity of heatmap overlay
    "edge_color": "white",    # Color for edge highlighting
    "heatmap_cmap": "jet",    # Default colormap for heatmap
    "edge_threshold": 0.2,    # Threshold for edge detection
    "heatmap_threshold": 0.5  # Threshold for significant relevance
}

# Visualization parameters
VIZ_CONFIG = {
    "dpi": 300,               # DPI for saved figures
    "fig_width": 10,          # Default figure width
    "fig_height": 8,          # Default figure height
    "colorbar_label": "Relevance",  # Label for colorbar
    "difference_cmap": "RdBu_r"     # Colormap for difference maps
}

# File paths - Reorganized for cleaner structure
PATH_CONFIG = {
    # Base directories
    "output_base": "results",                 # Main results directory
    "data_dir": "data",                       # Data directory

    # Run-specific directories (will be created for each run)
    "run_dir_template": "results/run_{run_id}",  # Template for run-specific directories

    # Analysis output types
    "heatmaps_dir": "heatmaps",              # Basic heatmap visualizations
    "metrics_dir": "metrics",                # Metrics data
    "raw_data_dir": "raw_data",              # Raw data files (.npz)

    # Summary directories (for integrated results)
    "summary_dir": "results/summary",        # Integrated results from all runs

    # Visualization types for summary
    "viz_types": {
        "mean_maps": "mean_maps",            # Mean relevance maps
        "uncertainty": "uncertainty",        # Standard deviation maps
        "confidence": "confidence",          # Confidence maps
        "difference": "difference",          # Difference maps
        "edge_analysis": "edge_analysis"     # Edge-enhanced visualizations
    },

    # Default image path
    "default_image": "data/0_Edinburgh_Nat_Gallery.jpg"
}

def get_class_name(idx: int) -> str:
    """Get the class name for a given index."""
    if idx == 0:
        return 'Raphael'
    elif idx == 1:
        return 'Non-Raphael'
    else:
        return f'class_idx={idx}'

def get_run_dir(run_id: int) -> str:
    """ Get the directory path for a specific run."""
    return PATH_CONFIG["run_dir_template"].format(run_id=run_id)

def get_summary_dir() -> str:
    """Get the directory path for summary results."""
    return PATH_CONFIG["summary_dir"]