"""
Test configuration file for pytest fixtures shared across test modules.
"""
import pytest
import numpy as np
from pathlib import Path
import os
import sys

# Add the parent directory to sys.path to allow importing from the main package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


@pytest.fixture
def sample_grayscale_image():
    """Create a sample grayscale test image with a white square."""
    img = np.zeros((64, 64), dtype=np.float32)
    img[20:40, 20:40] = 1.0  # White square in center
    return img


@pytest.fixture
def sample_rgb_image():
    """Create a sample RGB test image with a white square."""
    img = np.zeros((64, 64, 3), dtype=np.float32)
    img[20:40, 20:40, :] = 1.0  # White square in center
    return img


@pytest.fixture
def sample_dianna_image():
    """Create a sample image in DIANNA format (batch, channel, height, width)."""
    img = np.zeros((1, 1, 64, 64), dtype=np.float32)
    img[0, 0, 20:40, 20:40] = 1.0  # White square in center
    return img


@pytest.fixture
def sample_batch_images():
    """Create a batch of sample RGB images."""
    batch = np.zeros((2, 64, 64, 3), dtype=np.float32)
    batch[0, 20:40, 20:40, :] = 1.0  # White square in first image
    batch[1, 10:30, 30:50, :] = 1.0  # White square in second image
    return batch


@pytest.fixture
def sample_relevances():
    """Create sample relevance maps for two classes."""
    relevances = {
        0: np.ones((1, 64, 64)) * 0.8,  # High relevance for class 0 (Raphael)
        1: np.ones((1, 64, 64)) * 0.2   # Low relevance for class 1 (Non-Raphael)
    }
    return relevances


@pytest.fixture
def complex_relevances():
    """Create more complex relevance maps with distinct patterns."""
    # Class 0 (Raphael) map with high values in the center
    raphael_map = np.zeros((1, 64, 64))
    raphael_map[0, 20:40, 20:40] = 0.8  # High relevance in center
    
    # Class 1 (Non-Raphael) map with high values on the edges
    non_raphael_map = np.zeros((1, 64, 64))
    non_raphael_map[0, 0:10, 0:64] = 0.7  # High relevance at top
    non_raphael_map[0, 54:64, 0:64] = 0.7  # High relevance at bottom
    
    return {
        0: raphael_map,
        1: non_raphael_map
    }


@pytest.fixture
def test_data_dir(tmp_path):
    """Create a temporary directory structure for test data."""
    # Create a directory for test images
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    # Create a directory for non-Raphael paintings
    non_raphael_dir = data_dir / "Not Raphael"
    non_raphael_dir.mkdir()
    
    # Create a directory for output
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    # Create a directory for models
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    
    return data_dir 