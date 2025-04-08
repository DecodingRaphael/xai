import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
from typing import Optional
import pandas as pd
from model import Model
import numpy as np
from pathlib import Path
from dianna import visualization
from skimage import io, color, feature, filters, transform
from tqdm import tqdm
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib.colors

# Silence other common warnings
np.seterr(all='ignore')
plt.rcParams['figure.max_open_warning'] = 0

# Custom RISE implementation to ensure dimension compatibility
def custom_rise(model_fn, image, n_masks=50, p_keep=0.3, feature_res=6):
    """
Custom implementation of the RISE algorithm for generating saliency maps.

This function creates random binary masks, applies them to the input image,
and uses the provided model function to obtain predictions. The predictions
are then used to compute saliency maps for each class, highlighting the
importance of different regions in the image.

Parameters:
    model_fn (callable): The model function to obtain predictions.
    image (np.ndarray): The input image in (1, 1, h, w) format.
    n_masks (int, optional): Number of random masks to generate. Default is 50.
    p_keep (float, optional): Probability of keeping a pixel in the mask. 
    Default is 0.3.
    feature_res (int, optional): Resolution for the low-res mask. Default is 6.

Returns:
    dict: A dictionary where keys are class indices and values are saliency maps
    with dimensions (1, h, w).
"""
    
    # Create masks for RISE, height and width from (1, 1, h, w) format
    h, w = image.shape[2:4]
    
    # Generate random masks
    masks = []
    #cell_size = min(h, w) // feature_res       
    
    # Generate random masks
    for _ in range(n_masks):
        # Create a low-res binary mask
        mask_low_res = np.random.binomial(1, p_keep, size=(feature_res, feature_res))
        
        # Upsample to image size using skimage's transform.resize with nearest
        # neighbor interpolation order=0 specifies nearest-neighbor interpolation
        mask = transform.resize(mask_low_res, (h, w), order=0, mode='constant', 
                                preserve_range=True).astype(mask_low_res.dtype)
        
        # Reshape to match image format for DIANNA
        mask = np.expand_dims(np.expand_dims(mask, axis=0), axis=0)  # (1, 1, h, w)
        masks.append(mask)
    
    # Stack masks: shape (n_masks, 1, h, w)
    masks = np.vstack(masks)     
    
    # Apply masks to image; repeat image to match number of masks
    masked_images = []
    batch_size = 1
    
    # Process masks in small batches
    for i in tqdm(range(0, n_masks, batch_size), desc="Processing masks", disable=n_masks < 20):
        batch_end = min(i + batch_size, n_masks)
        batch_masks = masks[i:batch_end]
        
        # Broadcast image to match number of masks in batch
        batch_images = np.repeat(image, batch_end - i, axis=0)  # Shape: (batch_size, 1, h, w)
        
        # Apply each mask to the image
        masked = batch_images * batch_masks  # Element-wise multiplication
        masked_images.append(masked)
    
    # Stack the masked images; shape (n_masks, 1, h, w)
    masked_images = np.vstack(masked_images)    
    
    # Use the model function to get predictions for each masked image
    predictions = []
    for i in tqdm(range(0, n_masks, batch_size), desc="Getting predictions", disable=n_masks < 20):
        batch_end = min(i + batch_size, n_masks)
        batch_preds = model_fn(masked_images[i:batch_end])
        predictions.append(batch_preds)
    
    # Concatenate all predictions
    predictions = np.vstack(predictions)  # Shape: (n_masks, num_classes)
        
    # Compute saliency maps
    saliency = {}
    num_classes = predictions.shape[1]
    
    for class_idx in range(num_classes):
        # Extract class predictions
        class_preds = predictions[:, class_idx]
        
        # Weight masks by predictions
        weighted_masks = masks.reshape(n_masks, -1) * class_preds[:, np.newaxis]
        
        # Sum weighted masks
        saliency_map = weighted_masks.sum(axis=0) / (n_masks * p_keep)
        
        # Reshape to image dimensions
        saliency_map = saliency_map.reshape(1, h, w)
        
        saliency[class_idx] = saliency_map
    
    return saliency

def explain_painting(
        image_path: Path = Path('data/0_Edinburgh_Nat_Gallery.jpg'),
        p_keep: float = 0.3,
        n_masks: int = 50,
        feature_res: int = 6,
        file_name_appendix: Optional[str] = None,
):
    model = Model()
    labels = [0, 1]
    file_name_base = create_file_name_base(feature_res, file_name_appendix, image_path, n_masks, p_keep)
        
    x = io.imread(str(image_path))
    
    # Convert to RGB if it has an alpha channel
    if x.shape[-1] == 4:
        x = color.rgba2rgb(x).astype(np.float32)
    
    if x is None:
        raise ValueError(f"Image not found at {image_path}")    
    
    x_model = x.copy()    
    
    # Ensure the image is normalized to [0,1] range if it's not already
    if x.max() > 1.0:
        x = x / 255.0
    
    # First convert to grayscale since DIANNA's masks are single-channel
    x_gray = color.rgb2gray(x)    
    
    # Resize image to be square (DIANNA's RISE expects square images)
    target_size = max(x_gray.shape)
    x_resized = np.zeros((target_size, target_size))
    
    # Center the image in the square
    start_h = (target_size - x_gray.shape[0]) // 2
    start_w = (target_size - x_gray.shape[1]) // 2
    x_resized[start_h:start_h + x_gray.shape[0], start_w:start_w + x_gray.shape[1]] = x_gray
        
    # Add batch dimension: shape (1, height, height)
    x_input = np.expand_dims(x_resized, axis=0)
        
    # Add channel dimension to match mask shape: shape (1, height, height, 1)
    x_input = np.expand_dims(x_input, axis=-1)
        
    # Process image for our custom RISE implementation, which expects (batch, channels, height, width)
    x_rise = np.transpose(x_input, (0, 3, 1, 2))  # Move channel dim to position 1
    # Remove detailed shape output
    print("Processing image for RISE analysis...")
    
    # Create a wrapper function to ensure predictions are in the right format
    def model_wrapper(x):
        pred = model.run_on_batch(x)
        # Ensure predictions are 2D: [batch_size, num_classes]
        if len(pred.shape) == 1:
            pred = pred.reshape(1, -1)
        return pred
    
    # Run custom RISE implementation
    print(f"Generating relevance maps with {n_masks} masks...")
    relevances = custom_rise(model_wrapper, x_rise,n_masks=n_masks, 
                             feature_res=feature_res, p_keep=p_keep        
    )

    # Visualize the relevance scores for the predicted class on top of the input image
    predictions = model.run_on_batch(x_model[None, ...])
       
    # For visualization, we need to resize the relevance maps back to original size
    for class_idx in labels:
        relevance_map = relevances[class_idx][0]  # Remove batch dimension
        
        # Resize relevance map back to original dimensions
        if relevance_map.shape != x_gray.shape:
            # Extract the actual image region from the padded square
            relevance_map = relevance_map[start_h:start_h + x_gray.shape[0], start_w:start_w + x_gray.shape[1]]
        
        # Print only summary statistics, not the whole array
        print(f'Relevance map for {class_name(class_idx)} class (score: {predictions[0][class_idx]:.4f}), '
              f'stats: min={np.min(relevance_map):.4f}, max={np.max(relevance_map):.4f}, mean={np.mean(relevance_map):.4f}')
                
        visualization.plot_image(relevance_map, x, heatmap_cmap='jet',
                        output_filename=str(file_name_base) + f'_{class_name(class_idx)}.png', show_plot=False)    
    np.savez_compressed(str(file_name_base) + '.npz', relevances=relevances)

    # After creating relevance maps
    metrics = calculate_clarity_metrics(relevances)
    print("\nClarity Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    # Save metrics to CSV
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(str(file_name_base) + "_metrics.csv", index=False)


def create_file_name_base(feature_res, file_name_appendix, image_path, n_masks, p_keep):
    # Create output directory if it doesn't exist
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Create filename base in the output directory
    image_name = image_path.name
    base = f"{image_name}_nmasks_{n_masks}_pkeep_{p_keep}_res_{feature_res}"
    if file_name_appendix:
        base += f"_{file_name_appendix}"
    return output_dir / base  # Return as Path object

def class_name(idx):
    if idx == 0:
        name = 'Raphael'
    elif idx == 1:
        name = 'Non-Raphael'
    else:
        name = f'class_idx={idx}'
    return name

def calculate_clarity_metrics(relevance_maps):
    """Calculate metrics to quantify how clear/ambiguous the model's decision is"""
    raphael_map = relevance_maps[0]  # Raphael class
    non_raphael_map = relevance_maps[1]  # Non-Raphael class
    
    # 1. Contrast ratio (higher = clearer distinction)
    raphael_contrast = np.max(raphael_map) - np.min(raphael_map)
    non_raphael_contrast = np.max(non_raphael_map) - np.min(non_raphael_map)
    
    # 2. Overlap between heatmaps (lower = clearer distinction). Normalize both maps to [0,1] range
    r_norm = (raphael_map - np.min(raphael_map)) / max(1e-10, np.max(raphael_map) - np.min(raphael_map))
    nr_norm = (non_raphael_map - np.min(non_raphael_map)) / max(1e-10, np.max(non_raphael_map) - np.min(non_raphael_map))
    
    # Calculate overlap (intersection over union)
    intersection = np.sum(np.minimum(r_norm, nr_norm))
    union = np.sum(np.maximum(r_norm, nr_norm))
    iou = intersection / max(1e-10, union)  # Lower is better (less overlap)
    
    # 3. Focus ratio - how concentrated the attention is
    # Calculate entropy (lower = more focused on specific areas)
    r_entropy = scipy.stats.entropy(r_norm.flatten() + 1e-10)
    nr_entropy = scipy.stats.entropy(nr_norm.flatten() + 1e-10)
    
    # 4. Correlation between maps (lower = better differentiation)
    correlation = np.corrcoef(raphael_map.flatten(), non_raphael_map.flatten())[0, 1]
    
    metrics = {
        "raphael_contrast": raphael_contrast,
        "non_raphael_contrast": non_raphael_contrast,
        "overlap_iou": iou,
        "raphael_entropy": r_entropy,
        "non_raphael_entropy": nr_entropy,
        "map_correlation": correlation,
        "clarity_score": (raphael_contrast + non_raphael_contrast)/2 * (1-iou) * (1-abs(correlation))
    }
    
    return metrics

def visualize_edge_heatmap_overlay(image, heatmap, output_path, title="Edge-Enhanced RISE Map",
                                  edge_method='combined', edge_alpha=0.7, heatmap_alpha=0.6,
                                  edge_color='white', heatmap_cmap='jet', edge_weights=None):
    """
    Create visualization overlaying RISE heatmaps with edge detection maps to show
    if the model focuses on brushstroke patterns.
    
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
    edge_alpha : float
        Opacity of edge overlay (0-1)
    heatmap_alpha : float
        Opacity of heatmap overlay (0-1)
    edge_color : str
        Color for edge highlighting
    heatmap_cmap : str
        Colormap for heatmap
    edge_weights : list or None
        Weights for combined edge detection [Canny, Sobel, Laplacian, Scharr].
        Only used when edge_method='combined'. If None, default weights are used.
    """
    # Ensure image is in [0,1] range
    if image.max() > 1.0:
        image = image / 255.0
    
    # Convert to grayscale for edge detection
    gray = color.rgb2gray(image)
    
    # Initialize default weights for combined method
    if edge_weights is None and edge_method == 'combined':
        # Default weights as mentioned in the paper - should be determined experimentally
        # Setting reasonable defaults with higher weights to Sobel and Scharr which often
        # better capture brushwork characteristics
        edge_weights = [0.2, 0.3, 0.2, 0.3]  # [Canny, Sobel, Laplacian, Scharr]
    
    # Apply the specified edge detection method
    if edge_method == 'canny':
        edges = feature.canny(gray, sigma=1.0)
        # Convert boolean array to float
        edges = edges.astype(float)
        # Normalize to [0,1]
        if edges.max() > 0:
            edges = edges / edges.max()
    
    elif edge_method == 'sobel':
        sobelx = filters.sobel_h(gray)
        sobely = filters.sobel_v(gray)
        edges = np.sqrt(sobelx**2 + sobely**2)
        # Normalize to [0,1]
        if edges.max() > 0:
            edges = edges / edges.max()
    
    elif edge_method == 'laplacian':
        edges = np.abs(filters.laplace(gray))
        # Normalize to [0,1]
        if edges.max() > 0:
            edges = edges / edges.max()
    
    elif edge_method == 'scharr':
        scharrx = filters.scharr_h(gray)
        scharry = filters.scharr_v(gray)
        edges = np.sqrt(scharrx**2 + scharry**2)
        # Normalize to [0,1]
        if edges.max() > 0:
            edges = edges / edges.max()
    
    elif edge_method == 'combined':
        # Get all edge maps individually
        # Canny edges
        canny_edges = feature.canny(gray, sigma=1.0).astype(float)
        if canny_edges.max() > 0:
            canny_edges = canny_edges / canny_edges.max()
        
        # Sobel edges
        sobelx = filters.sobel_h(gray)
        sobely = filters.sobel_v(gray)
        sobel_edges = np.sqrt(sobelx**2 + sobely**2)
        if sobel_edges.max() > 0:
            sobel_edges = sobel_edges / sobel_edges.max()
        
        # Laplacian edges (LoG in the paper)
        laplacian_edges = np.abs(filters.laplace(gray))
        if laplacian_edges.max() > 0:
            laplacian_edges = laplacian_edges / laplacian_edges.max()
        
        # Scharr edges
        scharrx = filters.scharr_h(gray)
        scharry = filters.scharr_v(gray)
        scharr_edges = np.sqrt(scharrx**2 + scharry**2)
        if scharr_edges.max() > 0:
            scharr_edges = scharr_edges / scharr_edges.max()
        
        # Combine using weights: Ecombined = wcannyEcanny + wsobelEsobel + wLoGELoG + wscharrEscharr
        edges = (
            edge_weights[0] * canny_edges + 
            edge_weights[1] * sobel_edges + 
            edge_weights[2] * laplacian_edges + 
            edge_weights[3] * scharr_edges
        )
        
        # Normalize the combined result to [0,1]
        if edges.max() > 0:
            edges = edges / edges.max()
    
    else:
        raise ValueError(f"Unsupported edge method: {edge_method}")
    
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
    edge_threshold = 0.2  # Adjust as needed
    edge_mask = edges > edge_threshold
    
    # Create an edge overlay that only shows edges in regions highlighted by the heatmap
    heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-10)
    
    # Only show edges in regions with significant relevance
    heatmap_threshold = 0.5 
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
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Also create a single image with the combined visualization
    # This is more focused on the edge-heatmap overlap
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.imshow(heatmap, cmap=heatmap_cmap, alpha=heatmap_alpha)
    
    # Create a 3-channel overlay to highlight edges in areas of high relevance
    # The intensity of the edge color is proportional to both edge strength and heatmap value
    edge_highlight = np.zeros((*edges.shape, 3))  # RGB
    
    # Scale edges by heatmap intensity - this highlights edges in areas the model finds important
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
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return combined_path

def integrate_results(image_path, n_masks, p_keep, feature_res, runs=3):
    """
    Integrate results from multiple runs to create more robust explanations.
    
    This function:
    1. Finds all relevance maps matching the specified parameters
    2. Calculates mean and standard deviation across runs
    3. Creates visualizations including:
       - Mean relevance maps for each class
       - Standard deviation maps showing uncertainty
       - Confidence maps (high relevance + low variability)
       - Difference maps between Raphael and non-Raphael features
    4. Calculates integrated metrics and provides interpretation
    
    The integrated results will be saved in:
    - output/integrated/[filename]_integrated.npz: Raw data
    - output/integrated/visualizations/: Visual explanations
    - output/integrated/[filename]_integrated_metrics.csv: Aggregated metrics
    
    Parameters:
    -----------
    image_path : Path
        Path to the image being analyzed
    n_masks : int
        Number of masks used in the RISE analysis
    p_keep : float
        Proportion of pixels kept in each mask
    feature_res : int
        Resolution of the features in masks
    runs : int
        Number of runs to integrate
    """
    # Create output directory for integrated results
    output_dir = Path("output/integrated")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create base filename
    image_name = image_path.name
    base_pattern = f"{image_name}_nmasks_{n_masks}_pkeep_{p_keep}_res_{feature_res}"
    
    # Find all matching relevance maps from individual runs
    relevances_files = list(Path("output").glob(f"{base_pattern}_*npz"))
    
    if not relevances_files:
        print(f"No relevance maps found matching pattern {base_pattern}_*")
        return
    
    print(f"Found {len(relevances_files)} relevance maps to integrate")
    
    # Load all relevance maps
    all_relevances = []
    for file in relevances_files:
        try:
            data = np.load(file, allow_pickle=True)
            # Check if 'relevances' is already a dict or if it's a numpy array
            if isinstance(data['relevances'], dict):
                all_relevances.append(data['relevances'])
            else:
                # Try to convert to dict if it's a numpy array with .item() method
                try:
                    all_relevances.append(data['relevances'].item())
                except (AttributeError, ValueError):
                    print(f"Could not convert relevances from {file.name}")
                    continue
        except Exception as e:
            print(f"Error loading {file.name}: {str(e)}")
    
    # Process each class separately: First, identify all class indices across all runs
    all_classes = set()
    for relevance_map in all_relevances:
        all_classes.update(relevance_map.keys())
    
    # Process each class
    mean_relevances = {}
    std_relevances = {}
    
    for class_idx in all_classes:
        # Extract relevance maps for this class from all runs
        class_relevances = []
        for relevance_map in all_relevances:
            if class_idx in relevance_map:
                class_relevances.append(relevance_map[class_idx])
        
        if not class_relevances:
            continue
            
        # Stack class relevances and compute statistics
        try:
            stacked_class = np.stack(class_relevances)
            mean_relevances[class_idx] = np.mean(stacked_class, axis=0)
            std_relevances[class_idx] = np.std(stacked_class, axis=0)
        except Exception as e:
            print(f"Error processing class {class_idx}: {str(e)}")
    
    # Save the integrated results
    output_base = output_dir / base_pattern
    np.savez_compressed(str(output_base) + "_integrated.npz", 
                        mean=mean_relevances, 
                        std=std_relevances)
    
    print(f"Saved integrated results to {output_base}_integrated.npz")
    
    # Load the original image for visualization
    try:        
        # Load original image
        x = io.imread(str(image_path))
        
        # Normalize to [0,1] if needed
        if x.max() > 1.0:
            x = x / 255.0
            
        # Calculate padding dimensions similar to explain_painting
        # Get original image dimensions
        orig_height, orig_width = x.shape[:2]
        
        # Calculate target square size and padding
        target_size = max(orig_height, orig_width)
        start_h = (target_size - orig_height) // 2
        start_w = (target_size - orig_width) // 2
            
        # Make sure we have classes 0 and 1 (Raphael and non-Raphael)
        if 0 in mean_relevances and 1 in mean_relevances:
            # Create directory for visualizations
            (output_dir / "visualizations").mkdir(exist_ok=True)
            
            # 1. Create visualizations for mean relevance maps
            for class_idx in [0, 1]:  # 0=Raphael, 1=Non-Raphael
                mean_map = mean_relevances[class_idx][0]  # Get first batch item
                
                # Extract original image region if the map is padded
                if mean_map.shape[:2] != (orig_height, orig_width):
                    if len(mean_map.shape) == 2:  # Handle 2D case
                        mean_map = mean_map[start_h:start_h + orig_height, start_w:start_w + orig_width]
                    elif len(mean_map.shape) == 3:  # Handle 3D case with channels
                        mean_map = mean_map[start_h:start_h + orig_height, start_w:start_w + orig_width, :]
                
                # Use DIANNA visualization for individual maps
                try:
                    visualization.plot_image(
                        mean_map, x, heatmap_cmap='jet',
                        output_filename=str(output_dir / "visualizations" / f"{base_pattern}_mean_{class_name(class_idx)}.png"),
                        show_plot=False
                    )
                    print(f"Created mean visualization for {class_name(class_idx)}")
                except Exception as e:
                    print(f"Could not create visualization for {class_name(class_idx)}")
                
                # Visualize standard deviation (uncertainty) maps
                std_map = std_relevances[class_idx][0]  # Get first batch item
                
                # Extract original image region if the map is padded
                if std_map.shape[:2] != (orig_height, orig_width):
                    if len(std_map.shape) == 2:  # Handle 2D case
                        std_map = std_map[start_h:start_h + orig_height, start_w:start_w + orig_width]
                    elif len(std_map.shape) == 3:  # Handle 3D case with channels
                        std_map = std_map[start_h:start_h + orig_height, start_w:start_w + orig_width, :]
                
                try:
                    visualization.plot_image(
                        std_map, x, heatmap_cmap='viridis',
                        output_filename=str(output_dir / "visualizations" / f"{base_pattern}_std_{class_name(class_idx)}.png"),
                        show_plot=False
                    )
                    print(f"Created standard deviation visualization for {class_name(class_idx)}")
                except Exception as e:
                    print(f"Could not create standard deviation visualization for {class_name(class_idx)}")
                
                # Create confidence maps (mean * (1 - normalized std))
                # High confidence = high relevance AND low variability
                norm_std = std_map / (np.max(std_map) + 1e-10)
                confidence_map = mean_map * (1 - norm_std)
                try:
                    visualization.plot_image(
                        confidence_map, x, heatmap_cmap='jet',
                        output_filename=str(output_dir / "visualizations" / f"{base_pattern}_confidence_{class_name(class_idx)}.png"),
                        show_plot=False
                    )
                    print(f"Created confidence visualization for {class_name(class_idx)}")
                except Exception as e:
                    print(f"Could not create confidence visualization for {class_name(class_idx)}")
            
            # 2. Create a difference map (Raphael - Non-Raphael)
            try:
                plt.figure(figsize=(10, 8))
                
                # Extract difference map from original image region
                diff_map_raw = mean_relevances[0][0] - mean_relevances[1][0]
                
                # Extract original image region
                if diff_map_raw.shape[:2] != (orig_height, orig_width):
                    diff_map = diff_map_raw[start_h:start_h + orig_height, start_w:start_w + orig_width]
                else:
                    diff_map = diff_map_raw
                
                # Scale for better visualization
                abs_max = np.max(np.abs(diff_map))
                plt.imshow(x)
                plt.imshow(diff_map, cmap='RdBu_r', alpha=0.7, vmin=-abs_max, vmax=abs_max)
                plt.colorbar(label='Raphael - Non-Raphael')
                plt.title('Difference Map (Red = Raphael, Blue = Non-Raphael)')
                plt.tight_layout()
                plt.savefig(str(output_dir / "visualizations" / f"{base_pattern}_difference_map.png"), dpi=300)
                plt.close()
                print("Created difference map visualization")
            except Exception as e:
                print(f"Could not create difference map")
            
            # 3. Calculate integrated metrics across runs. First collect all metrics from individual runs
            all_metrics = []
            metrics_files = list(Path("output").glob(f"{base_pattern}_*_metrics.csv"))
            for file in metrics_files:
                try:
                    metrics_df = pd.read_csv(file)
                    all_metrics.append(metrics_df)
                except Exception as e:
                    print(f"Error reading metrics from {file.name}")
                    
            if all_metrics:
                # Concatenate all metrics and calculate mean, std
                try:
                    combined_metrics = pd.concat(all_metrics, ignore_index=True)
                    agg_metrics = combined_metrics.agg(['mean', 'std', 'min', 'max'])
                    
                    # Save to CSV
                    agg_metrics.to_csv(str(output_dir / f"{base_pattern}_integrated_metrics.csv"))
                    
                    # Print summary
                    print("\nIntegrated Clarity Metrics Summary:")
                    for metric in combined_metrics.columns:
                        mean_val = agg_metrics.loc['mean', metric]
                        std_val = agg_metrics.loc['std', metric]
                        print(f"{metric}: {mean_val:.4f} ± {std_val:.4f}")
                    
                    # Print interpretation based on metrics
                    clarity = agg_metrics.loc['mean', 'clarity_score']
                    overlap = agg_metrics.loc['mean', 'overlap_iou']
                    correlation = agg_metrics.loc['mean', 'map_correlation']
                    
                    print("\nInterpretation of Results:")
                    if clarity > 0.5:
                        print("- HIGH CLARITY: The model shows clear distinction between Raphael and non-Raphael features")
                    elif clarity > 0.2:
                        print("- MODERATE CLARITY: The model shows some distinction between Raphael and non-Raphael features")
                    else:
                        print("- LOW CLARITY: The model shows poor distinction between Raphael and non-Raphael features")
                        
                    if overlap < 0.3:
                        print("- LOW OVERLAP: The relevance maps for Raphael and non-Raphael have minimal overlap, suggesting distinct features")
                    elif overlap < 0.6:
                        print("- MODERATE OVERLAP: The relevance maps show some overlap between Raphael and non-Raphael features")
                    else:
                        print("- HIGH OVERLAP: The relevance maps show significant overlap, making feature distinction ambiguous")
                        
                    if abs(correlation) < 0.2:
                        print("- LOW CORRELATION: The model focuses on different regions for Raphael vs non-Raphael")
                    elif abs(correlation) < 0.5:
                        print("- MODERATE CORRELATION: The model shows some similarity in focus areas")
                    else:
                        print("- HIGH CORRELATION: The model focuses on similar regions, possibly indicating poor discrimination")
                except Exception as e:
                    print(f"Error calculating aggregated metrics")
        
        else:
            print(f"Warning: Expected to find classes 0 and 1 in results, but found {list(mean_relevances.keys())}")
            
        # After creating difference map, add edge-enhanced visualizations
        try:
            # Create edge-enhanced visualizations using different edge detection methods
            for class_idx in [0, 1]:  # 0=Raphael, 1=Non-Raphael
                if class_idx not in mean_relevances:
                    continue
                    
                mean_map = mean_relevances[class_idx][0]  # Get first batch item
                
                # Extract original image region if needed
                if mean_map.shape[:2] != (orig_height, orig_width):
                    if len(mean_map.shape) == 2:  # Handle 2D case
                        mean_map = mean_map[start_h:start_h + orig_height, start_w:start_w + orig_width]
                    elif len(mean_map.shape) == 3:  # Handle 3D case with channels
                        mean_map = mean_map[start_h:start_h + orig_height, start_w:start_w + orig_width, :]
                
                # Create directory for edge visualizations
                edge_dir = output_dir / "visualizations" / "edge_analysis"
                edge_dir.mkdir(exist_ok=True, parents=True)
                
                # Generate edge-enhanced visualizations with different edge detection methods
                for edge_method in ['combined']:  # Only use combined edge detection
                    output_path = edge_dir / f"{base_pattern}_{class_name(class_idx)}_{edge_method}_edges.png"
                    try:
                        # If using combined method, define weights
                        edge_weights = None
                        if edge_method == 'combined':
                            # These weights could be determined experimentally
                            edge_weights = [0.2, 0.3, 0.2, 0.3]  # Canny, Sobel, Laplacian, Scharr
                        
                        visualize_edge_heatmap_overlay(
                            image=x, 
                            heatmap=mean_map, 
                            output_path=output_path,
                            title=f"{class_name(class_idx)} Detection: Brushstroke Analysis",
                            edge_method=edge_method,
                            edge_weights=edge_weights
                        )
                        print(f"Created edge-enhanced visualization for {class_name(class_idx)}")
                    except Exception as edge_err:
                        print(f"Could not create edge-enhanced visualization for {class_name(class_idx)}")
                
                # Also create an edge-enhanced visualization for the confidence map
                if class_idx in std_relevances:
                    std_map = std_relevances[class_idx][0]
                    # Extract original image region if needed
                    if std_map.shape[:2] != (orig_height, orig_width):
                        if len(std_map.shape) == 2:
                            std_map = std_map[start_h:start_h + orig_height, start_w:start_w + orig_width]
                        elif len(std_map.shape) == 3:
                            std_map = std_map[start_h:start_h + orig_height, start_w:start_w + orig_width, :]
                    
                    # Create confidence map (high relevance and low variance)
                    norm_std = std_map / (np.max(std_map) + 1e-10)
                    confidence_map = mean_map * (1 - norm_std)
                    
                    # Create edge-enhanced visualization of confidence map with combined edges
                    try:
                        output_path = edge_dir / f"{base_pattern}_{class_name(class_idx)}_confidence_combined_edges.png"
                        edge_weights = [0.2, 0.3, 0.2, 0.3]  # Canny, Sobel, Laplacian, Scharr
                        visualize_edge_heatmap_overlay(
                            image=x, 
                            heatmap=confidence_map, 
                            output_path=output_path,
                            title=f"{class_name(class_idx)} Detection: Confident Brushstroke Patterns",
                            edge_method='combined',
                            edge_weights=edge_weights
                        )
                        print(f"Created edge-enhanced confidence map for {class_name(class_idx)}")
                    except Exception as e:
                        print(f"Could not create edge-enhanced confidence map for {class_name(class_idx)}")
                    
            # Create edge-enhanced difference map with various edge detection methods
            if 0 in mean_relevances and 1 in mean_relevances:
                diff_map_raw = mean_relevances[0][0] - mean_relevances[1][0]
                # Extract original image region if needed
                if diff_map_raw.shape[:2] != (orig_height, orig_width):
                    diff_map = diff_map_raw[start_h:start_h + orig_height, start_w:start_w + orig_width]
                else:
                    diff_map = diff_map_raw
                    
                # Normalize to [0,1] range for visualization
                diff_norm = (diff_map - diff_map.min()) / (diff_map.max() - diff_map.min() + 1e-10)
                
                # Create edge-enhanced visualization of difference map with sobel edges (keep this for compatibility)
                try:
                    output_path = edge_dir / f"{base_pattern}_difference_sobel_edges.png"
                    visualize_edge_heatmap_overlay(
                        image=x, 
                        heatmap=diff_norm, 
                        output_path=output_path,
                        title="Raphael vs Non-Raphael: Distinctive Brushstroke Patterns",
                        edge_method='sobel',
                        heatmap_cmap='RdBu_r'
                    )
                    print("Created edge-enhanced difference map with Sobel edges")
                except Exception as e:
                    print("Could not create edge-enhanced difference map with Sobel edges")
                
                # Now create the combined edge version of the difference map
                try:
                    output_path = edge_dir / f"{base_pattern}_difference_combined_edges.png"
                    edge_weights = [0.2, 0.3, 0.2, 0.3]  # Canny, Sobel, Laplacian, Scharr
                    visualize_edge_heatmap_overlay(
                        image=x, 
                        heatmap=diff_norm, 
                        output_path=output_path,
                        title="Raphael vs Non-Raphael: Distinctive Brushstroke Patterns (Combined Edge Analysis)",
                        edge_method='combined',
                        edge_weights=edge_weights,
                        heatmap_cmap='RdBu_r'
                    )
                    print("Created edge-enhanced difference map with combined edges")
                except Exception as e:
                    print("Could not create edge-enhanced difference map with combined edges")
        
        except Exception as e:
            print("Error creating edge-enhanced visualizations")
        
    except Exception as e:
        print("Error creating visualizations")
        
    print(f"Integration complete for {image_path.name}")


if __name__ == "__main__":
    
    # set to True to test. If correct, set to false and run real analysis
    is_classification_run = False
    
    # Verify data paths exist
    painting_paths = [Path(p) for p in ['data/0_Edinburgh_Nat_Gallery.jpg']]
    all_paths_exist = True
    for path in painting_paths:
        if not path.exists():
            print(f"WARNING: Image file {path} does not exist. Please check the path.")
            all_paths_exist = False
            
    # Check Non-Raphael directory exists
    if not Path('data/Not Raphael').exists():
        print(f"WARNING: Directory 'data/Not Raphael' does not exist. Please check the path.")
        all_paths_exist = False
    
    if not all_paths_exist:
        print("Exiting due to missing files.")
        exit(1)
    
    if is_classification_run:
        paths = painting_paths
        
        results = []        
        for path in paths:
            model = Model()            
            
            img = io.imread(str(path))
            
            result = model.run_on_batch(img)
            results.append(result)

        # Create a simple table of results
        result_df = pd.DataFrame(results, columns=[class_name(idx) for idx in [0, 1]])
        result_df.index = [p.name for p in paths]
        print("Classification Results:")
        print(result_df)

    else:        
        n_masks = 5
        p_keep = 0.3
        feature_res = 6
        
        print(f"Starting RISE analysis with {n_masks} masks...")
        
        for painting_path in painting_paths:
            print(f"\nProcessing: {painting_path.name}")
            
            for run in range(3): # Run 3 iterations for stability
                print(f"Run {run+1}/3")
                # Get heatmaps for the painting indicating the relevance of each pixel for the prediction
                explain_painting(n_masks            = n_masks,
                                 p_keep             = p_keep,
                                 feature_res        = feature_res,
                                 file_name_appendix = str(run),
                                 image_path         = painting_path)

        # After running all the individual analyses
        for painting_path in painting_paths:
            print(f"\nIntegrating results for {painting_path.name}")
            integrate_results(image_path=painting_path, n_masks=n_masks, p_keep=p_keep, feature_res=feature_res, runs=3)
        
        print("\nAnalysis complete. Results saved to the 'output' directory.")