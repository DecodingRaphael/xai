import warnings
warnings.filterwarnings('ignore')
from typing import Optional
import pandas as pd
from model import Model
import numpy as np
from pathlib import Path
#from keras import utils
#import dianna
from dianna import visualization
import cv2
from skimage import io, color
from tqdm import tqdm
import scipy.stats
import matplotlib.pyplot as plt


# Custom RISE implementation to ensure dimension compatibility
def custom_rise(model_fn, image, n_masks=10, p_keep=0.1, feature_res=6):
    """Custom RISE implementation that ensures dimension compatibility"""
    
    # Create masks for RISE
    h, w = image.shape[2:4]  # Height and width from (1, 1, h, w) format
    
    # Generate random masks
    masks = []
    cell_size = min(h, w) // feature_res
    
    print(f"Generating {n_masks} masks of size {h}x{w} with cell size {cell_size}")
    
    # Generate random masks
    for _ in range(n_masks):
        # Create a low-res binary mask
        mask_low_res = np.random.binomial(1, p_keep, size=(feature_res, feature_res))
        
        # Upsample to image size using nearest neighbor
        mask = cv2.resize(
            mask_low_res, 
            (w, h), 
            interpolation=cv2.INTER_NEAREST
        )
        
        # Reshape to match image format for DIANNA
        mask = np.expand_dims(np.expand_dims(mask, axis=0), axis=0)  # (1, 1, h, w)
        masks.append(mask)
    
    # Stack masks
    masks = np.vstack(masks)  # Shape: (n_masks, 1, h, w)
    print(f"Masks shape: {masks.shape}")
    
    # Apply masks to image
    # Repeat image to match number of masks
    masked_images = []
    batch_size = 1
    
    # Process masks in small batches
    for i in tqdm(range(0, n_masks, batch_size), desc="Processing masks"):
        batch_end = min(i + batch_size, n_masks)
        batch_masks = masks[i:batch_end]
        
        # Broadcast image to match number of masks in batch
        batch_images = np.repeat(image, batch_end - i, axis=0)  # Shape: (batch_size, 1, h, w)
        
        # Apply masks
        masked = batch_images * batch_masks  # Element-wise multiplication
        masked_images.append(masked)
    
    # Concatenate all masked images
    masked_images = np.vstack(masked_images)  # Shape: (n_masks, 1, h, w)
    print(f"Masked images shape: {masked_images.shape}")
    
    # Get predictions for all masked images
    predictions = []
    for i in tqdm(range(0, n_masks, batch_size), desc="Getting predictions"):
        batch_end = min(i + batch_size, n_masks)
        batch_preds = model_fn(masked_images[i:batch_end])
        predictions.append(batch_preds)
    
    # Concatenate all predictions
    predictions = np.vstack(predictions)  # Shape: (n_masks, num_classes)
    print(f"Predictions shape: {predictions.shape}")
    
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


# for plotting
def explain_painting(
        image_path: Path = Path('data/0_Edinburgh_Nat_Gallery.jpg'),
        p_keep: float = 0.1,
        n_masks: int = 10,
        feature_res: int = 6,
        file_name_appendix: Optional[str] = None,
):
    model = Model()
    labels = [0, 1]  # Both Raphael and Non-Raphael classes
    
    file_name_base = create_file_name_base(feature_res, file_name_appendix, image_path, n_masks, p_keep)      
    
    # Load image with scikit-image instead of cv2 for consistent RGB format
    x = io.imread(str(image_path))
    
    # Convert to RGB if it has an alpha channel
    if x.shape[-1] == 4:
        x = color.rgba2rgb(x).astype(np.float32)
    
    if x is None:
        raise ValueError(f"Image not found at {image_path}")    

    # Create a copy for model inference - skimage loads in RGB format
    x_model = x.copy()
    
    print("Original image shape:", x.shape)
    
    # Ensure the image is normalized to [0,1] range if it's not already
    if x.max() > 1.0:
        x = x / 255.0
    
    # First convert to grayscale since DIANNA's masks are single-channel
    x_gray = color.rgb2gray(x)
    print("Grayscale shape:", x_gray.shape)
    
    # Resize image to be square (DIANNA's RISE expects square images)
    target_size = max(x_gray.shape)
    x_resized = np.zeros((target_size, target_size))
    
    # Center the image in the square
    start_h = (target_size - x_gray.shape[0]) // 2
    start_w = (target_size - x_gray.shape[1]) // 2
    x_resized[start_h:start_h + x_gray.shape[0], start_w:start_w + x_gray.shape[1]] = x_gray
    print("Resized shape:", x_resized.shape)
    
    # Add batch dimension
    x_input = np.expand_dims(x_resized, axis=0)  # Shape: (1, height, height)
    print("After adding batch dim:", x_input.shape)
    
    # Add channel dimension to match mask shape
    x_input = np.expand_dims(x_input, axis=-1)  # Shape: (1, height, height, 1)
    print("After adding channel dim:", x_input.shape)
    
    # Process image for our custom RISE implementation
    # Custom RISE expects (batch, channels, height, width)
    x_rise = np.transpose(x_input, (0, 3, 1, 2))  # Move channel dim to position 1
    print("After transpose for RISE:", x_rise.shape)
    
    # Create a wrapper function to ensure predictions are in the right format
    def model_wrapper(x):
        pred = model.run_on_batch(x)
        # Ensure predictions are 2D: [batch_size, num_classes]
        if len(pred.shape) == 1:
            pred = pred.reshape(1, -1)
        return pred
    
    # Run custom RISE implementation instead of DIANNA
    print("Using custom RISE implementation")
    relevances = custom_rise(
        model_wrapper,
        x_rise,
        n_masks=n_masks,
        p_keep=p_keep,
        feature_res=feature_res
    )

    # # Visualize the relevance scores for the predicted class on top of the input image
    predictions = model.run_on_batch(x_model[None, ...])
    
    #pred_idx = np.argmax(predictions[0])  # Get prediction from first batch
    #print(f"Predicted class: {class_name(pred_idx)}")

    # For visualization, we need to resize the relevance maps back to original size
    for class_idx in labels:
        relevance_map = relevances[class_idx][0]  # Remove batch dimension
        
        # Resize relevance map back to original dimensions
        if relevance_map.shape != x_gray.shape:
            # Extract the actual image region from the padded square
            relevance_map = relevance_map[start_h:start_h + x_gray.shape[0], start_w:start_w + x_gray.shape[1]]
        
        print(f'Explanation for `{class_name(class_idx)}` ({predictions[0][class_idx]}), '
              f'relevances: min={np.min(relevance_map)}, max={np.max(relevance_map)}, mean={np.mean(relevance_map)}')
        
        # Use original RGB image for visualization
        visualization.plot_image(relevance_map, x, heatmap_cmap='jet',
                        output_filename=str(file_name_base) + f'_{class_name(class_idx)}.png', 
                        show_plot=False)
    
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
    
    # 2. Overlap between heatmaps (lower = clearer distinction)
    # Normalize both maps to [0,1] range
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

def integrate_results(image_path, n_masks, p_keep, feature_res, runs=3):
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
        data = np.load(file, allow_pickle=True)
        all_relevances.append(data['relevances'].item())  # Convert to Python dict
    
    # Process each class separately
    # First, identify all class indices across all runs
    all_classes = set()
    for relevance_map in all_relevances:
        all_classes.update(relevance_map.keys())
    
    print(f"Found classes: {all_classes}")
    
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
            print(f"Error processing class {class_idx}: {e}")
            print(f"Shapes: {[r.shape for r in class_relevances]}")
    
    # Save the integrated results
    output_base = output_dir / base_pattern
    np.savez_compressed(str(output_base) + "_integrated.npz", 
                        mean=mean_relevances, 
                        std=std_relevances)
    
    print(f"Integrated results saved to {str(output_base)}_integrated.npz")
    
    # Load the original image for visualization
    try:
        import matplotlib.pyplot as plt
        from skimage import io
        
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
        
        print(f"Original image dimensions: {orig_height}x{orig_width}")
        print(f"Padded square dimensions: {target_size}x{target_size}")
        print(f"Padding: top={start_h}, left={start_w}")
            
        # Make sure we have classes 0 and 1 (Raphael and non-Raphael)
        if 0 in mean_relevances and 1 in mean_relevances:
            # 1. Create visualizations for mean relevance maps
            for class_idx in [0, 1]:  # 0=Raphael, 1=Non-Raphael
                mean_map = mean_relevances[class_idx][0]  # Get first batch item
                
                # Extract original image region if the map is padded
                if mean_map.shape[:2] != (orig_height, orig_width):
                    if len(mean_map.shape) == 2:  # Handle 2D case
                        mean_map = mean_map[start_h:start_h + orig_height, start_w:start_w + orig_width]
                    elif len(mean_map.shape) == 3:  # Handle 3D case with channels
                        mean_map = mean_map[start_h:start_h + orig_height, start_w:start_w + orig_width, :]
                
                # Create directory for visualizations
                (output_dir / "visualizations").mkdir(exist_ok=True)
                
                # Use DIANNA visualization for individual maps
                visualization.plot_image(
                    mean_map, x, heatmap_cmap='jet',
                    output_filename=str(output_dir / "visualizations" / f"{base_pattern}_mean_{class_name(class_idx)}.png"),
                    show_plot=False
                )
                
                # Visualize standard deviation (uncertainty) maps
                std_map = std_relevances[class_idx][0]  # Get first batch item
                
                # Extract original image region if the map is padded
                if std_map.shape[:2] != (orig_height, orig_width):
                    if len(std_map.shape) == 2:  # Handle 2D case
                        std_map = std_map[start_h:start_h + orig_height, start_w:start_w + orig_width]
                    elif len(std_map.shape) == 3:  # Handle 3D case with channels
                        std_map = std_map[start_h:start_h + orig_height, start_w:start_w + orig_width, :]
                
                visualization.plot_image(
                    std_map, x, heatmap_cmap='viridis',
                    output_filename=str(output_dir / "visualizations" / f"{base_pattern}_std_{class_name(class_idx)}.png"),
                    show_plot=False
                )
                
                # Create confidence maps (mean * (1 - normalized std))
                # High confidence = high relevance AND low variability
                norm_std = std_map / (np.max(std_map) + 1e-10)
                confidence_map = mean_map * (1 - norm_std)
                visualization.plot_image(
                    confidence_map, x, heatmap_cmap='jet',
                    output_filename=str(output_dir / "visualizations" / f"{base_pattern}_confidence_{class_name(class_idx)}.png"),
                    show_plot=False
                )
            
            # 2. Create a difference map (Raphael - Non-Raphael)
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
            
            # 3. Calculate integrated metrics across runs
            # First collect all metrics from individual runs
            all_metrics = []
            metrics_files = list(Path("output").glob(f"{base_pattern}_*_metrics.csv"))
            for file in metrics_files:
                try:
                    metrics_df = pd.read_csv(file)
                    all_metrics.append(metrics_df)
                except Exception as e:
                    print(f"Error reading metrics from {file}: {e}")
                    
            if all_metrics:
                # Concatenate all metrics and calculate mean, std
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
        
        else:
            print(f"Warning: Expected to find classes 0 and 1 in results, but found {all_classes}")
            
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    
    # set to True to test. If correct, set to false asnd run real analysis
    is_classification_run = False
    
    if is_classification_run:
        paths = [Path(p) for p in ['data/0_Edinburgh_Nat_Gallery.jpg']]
        
        results = []
        
        for path in paths:
            model = Model()
            
            # Load image with scikit-image for consistent format
            img = io.imread(str(path))
            result = model.run_on_batch(img)
            results.append(result)

        for path, result in zip(paths, results):
            print(f'{result=}')
            print(f'{path=}')
            print(pd.DataFrame([result], columns=[class_name(idx) for idx in [0, 1]]))

    else:
        painting_paths = [Path(p) for p in ['data/0_Edinburgh_Nat_Gallery.jpg']]
        
        for painting_path in painting_paths:
            for n_masks in [5]:  #5000 wanneer code correct; results are then more stable. start with 500, when the image does not change, 500 would be enough                 
                for p_keep in [0.7]: # verhouding mask vs non-mask pixels                    
                    for feature_res in [12]: # als je maskeert, wil je groepen maskeren die naast gelegen zijn
                        for run in range(5):
                            print(f'Running {run} of {painting_path} with {n_masks} masks, {p_keep} keep ratio, and {feature_res} feature resolution')
                            # heatmaps for the painting indicating the relevance of each pixel for the prediction
                            explain_painting(n_masks            = n_masks,
                                             p_keep             = p_keep,
                                             feature_res        = feature_res,
                                             file_name_appendix = str(run),
                                             image_path         = painting_path)

        # After running all the individual analyses
        for painting_path in painting_paths:
            print(f"Integrating results for {painting_path}")
            integrate_results(image_path=painting_path, n_masks=5, p_keep=0.7, feature_res=12, runs=3)