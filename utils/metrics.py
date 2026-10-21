"""Functions for calculating and analyzing metrics from relevance maps."""
import numpy as np
import pandas as pd
import scipy.stats
from typing import Dict

def calculate_clarity_metrics(relevance_maps: Dict[int, np.ndarray]) -> Dict[str, float]:
    """
    Calculate metrics to quantify how clear/ambiguous the model's decision is.

    Parameters:
    -----------
    relevance_maps : Dict[int, np.ndarray]
        Dictionary of relevance maps for each class

    Returns:
    --------
    Dict[str, float]
        Dictionary of metrics
    """
    raphael_map = relevance_maps[0]  # Raphael class
    non_raphael_map = relevance_maps[1]  # Non-Raphael class

    # 1. Contrast ratio (higher = clearer distinction)
    raphael_contrast = np.max(raphael_map) - np.min(raphael_map)
    non_raphael_contrast = np.max(non_raphael_map) - np.min(non_raphael_map)

    # 2. Normalize maps to [0,1] range for comparison
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

def interpret_metrics(metrics: Dict[str, float]) -> Dict[str, str]:
    """
    Interpret metrics and return human-readable insights.

    Parameters:
    -----------
    metrics : Dict[str, float]
        Dictionary of metrics to interpret

    Returns:
    --------
    Dict[str, str]
        Dictionary of interpretations
    """
    interpretations = {}

    # Interpret clarity score
    clarity = metrics['clarity_score']
    if clarity > 0.5:
        interpretations['clarity'] = "HIGH CLARITY: The model shows clear distinction between Raphael and non-Raphael features"
    elif clarity > 0.2:
        interpretations['clarity'] = "MODERATE CLARITY: The model shows some distinction between Raphael and non-Raphael features"
    else:
        interpretations['clarity'] = "LOW CLARITY: The model shows poor distinction between Raphael and non-Raphael features"

    # Interpret overlap
    overlap = metrics['overlap_iou']
    if overlap < 0.3:
        interpretations['overlap'] = "LOW OVERLAP: The relevance maps for Raphael and non-Raphael have minimal overlap, suggesting distinct features"
    elif overlap < 0.6:
        interpretations['overlap'] = "MODERATE OVERLAP: The relevance maps show some overlap between Raphael and non-Raphael features"
    else:
        interpretations['overlap'] = "HIGH OVERLAP: The relevance maps show significant overlap, making feature distinction ambiguous"

    # Interpret correlation
    correlation = metrics['map_correlation']
    if abs(correlation) < 0.2:
        interpretations['correlation'] = "LOW CORRELATION: The model focuses on different regions for Raphael vs non-Raphael"
    elif abs(correlation) < 0.5:
        interpretations['correlation'] = "MODERATE CORRELATION: The model shows some similarity in focus areas"
    else:
        interpretations['correlation'] = "HIGH CORRELATION: The model focuses on similar regions, possibly indicating poor discrimination"

    return interpretations

def aggregate_metrics(metrics_files: list) -> pd.DataFrame:
    """
    Aggregate metrics from multiple files and calculate statistics.

    Parameters:
    -----------
    metrics_files : list
        List of paths to metrics CSV files

    Returns:
    --------
    pd.DataFrame
        DataFrame with aggregated metrics
    """
    all_metrics = []

    for file in metrics_files:
        try:
            metrics_df = pd.read_csv(file)
            all_metrics.append(metrics_df)
        except Exception as e:
            print(f"Error reading metrics from {file.name}: {str(e)}")

    if not all_metrics:
        return pd.DataFrame()

    # Concatenate all metrics and calculate statistics
    combined_metrics = pd.concat(all_metrics, ignore_index=True)
    agg_metrics = combined_metrics.agg(['mean', 'std', 'min', 'max'])

    return agg_metrics