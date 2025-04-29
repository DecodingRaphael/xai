"""Utility functions for file operations."""
from pathlib import Path
from typing import Optional, List, Dict

from utils.config import PATH_CONFIG, get_run_dir, get_summary_dir

def create_output_directories(run_id: Optional[int] = None) -> Dict[str, Path]:
    """ Create and return a structured directory layout for outputs."""
    # Create base directories
    base_dir = Path(PATH_CONFIG["output_base"])
    base_dir.mkdir(exist_ok=True, parents=True)

    dirs = {"base": base_dir}

    # Create summary directory
    summary_dir = Path(get_summary_dir())
    summary_dir.mkdir(exist_ok=True, parents=True)
    dirs["summary"] = summary_dir

    # Create visualization type directories under summary
    for viz_type, dirname in PATH_CONFIG["viz_types"].items():
        viz_dir = summary_dir / dirname
        viz_dir.mkdir(exist_ok=True, parents=True)
        dirs[f"summary_{viz_type}"] = viz_dir

    # If run_id is provided, create run-specific directories
    if run_id is not None:
        run_dir = Path(get_run_dir(run_id=run_id))
        run_dir.mkdir(exist_ok=True, parents=True)
        dirs["run"] = run_dir

        # Create subdirectories for different output types
        for output_type in ["heatmaps_dir", "metrics_dir", "raw_data_dir"]:
            output_dir = run_dir / PATH_CONFIG[output_type]
            output_dir.mkdir(exist_ok=True, parents=True)
            dirs[output_type.replace("_dir", "")] = output_dir

    return dirs

def get_run_directories() -> List[Path]:
    """Get a list of all run directories."""
    base_dir = Path(PATH_CONFIG["output_base"])
    if not base_dir.exists():
        return []

    # Only include directories that start with run_
    return [p for p in base_dir.iterdir()
            if p.is_dir() and p.name.startswith("run_")]

def create_file_name_base(
    feature_res: int,
    file_name_appendix: Optional[str],
    image_path: Path,
    n_masks: int,
    p_keep: float,
    run_id: Optional[int] = None
) -> str:
    """ Create a base filename for output files."""

    base_name = f"{image_path.name}_nmasks_{n_masks}_pkeep_{p_keep}_res_{feature_res}"
    if file_name_appendix:
        base_name += f"_{file_name_appendix}"

    return base_name

def get_heatmap_path(base_filename: str, class_name: str, run_id: int) -> Path:
    """Generate a path for a heatmap file."""
    dirs = create_output_directories(run_id)
    return dirs["heatmaps"] / f"{base_filename}_{class_name}.png"

def get_metrics_path(base_filename: str, run_id: int) -> Path:
    """Generate a path for a metrics file."""

    dirs = create_output_directories(run_id)
    return dirs["metrics"] / f"{base_filename}_metrics.csv"

def get_raw_data_path(base_filename: str, run_id: int) -> Path:
    """ Generate a path for a raw data file. """
    dirs = create_output_directories(run_id)
    return dirs["raw_data"] / f"{base_filename}.npz"

def get_summary_visualization_path(
    base_filename: str,
    viz_type: str,
    class_name: Optional[str] = None,
    subtype: Optional[str] = None
) -> Path:
    """ Generate a path for a summary visualization file."""
    dirs = create_output_directories()
    viz_dir = dirs[f"summary_{viz_type}"]

    # Build filename
    filename = base_filename
    if class_name:
        filename += f"_{class_name}"
    if subtype:
        filename += f"_{subtype}"

    return viz_dir / f"{filename}.png"

def get_summary_data_path(base_filename: str, data_type: str) -> Path:
    """  Generate a path for a summary data file."""
    dirs = create_output_directories()

    if data_type == "metrics":
        return dirs["summary"] / f"{base_filename}_integrated_metrics.csv"
    else:  # For generic data files like integrated.npz
        return dirs["summary"] / f"{base_filename}_{data_type}.npz"

def get_raw_data_files_for_pattern(pattern: str) -> List[Path]:
    """ Get all raw data files matching a pattern."""
    run_dirs = get_run_directories()
    matching_files = []

    for run_dir in run_dirs:
        raw_dir = run_dir / PATH_CONFIG["raw_data_dir"]
        if raw_dir.exists():
            matching_files.extend(list(raw_dir.glob(f"{pattern}*.npz")))

    return matching_files

def get_metrics_files_for_pattern(pattern: str) -> List[Path]:
    """ Get all metrics files matching a pattern."""
    run_dirs = get_run_directories()
    matching_files = []

    for run_dir in run_dirs:
        metrics_dir = run_dir / PATH_CONFIG["metrics_dir"]
        if metrics_dir.exists():
            matching_files.extend(list(metrics_dir.glob(f"{pattern}*_metrics.csv")))

    return matching_files