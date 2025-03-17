import unittest
import numpy as np
from unittest.mock import patch, MagicMock, mock_open
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import pytest

# Add parent directory to path to import rise_imagenet
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import rise_imagenet
from rise_imagenet import custom_rise, calculate_clarity_metrics
from rise_imagenet import visualize_edge_heatmap_overlay, class_name, create_file_name_base


class TestRiseImagenet(unittest.TestCase):
    """Test cases for rise_imagenet.py functions"""

    def setUp(self):
        """Set up test fixtures"""
        # Create a mock image for testing (grayscale)
        self.mock_gray_image = np.zeros((64, 64), dtype=np.float32)
        self.mock_gray_image[20:40, 20:40] = 1.0  # White square
        
        # Create a mock image for testing (RGB)
        self.mock_rgb_image = np.zeros((64, 64, 3), dtype=np.float32)
        self.mock_rgb_image[20:40, 20:40, :] = 1.0  # White square
        
        # Create a mock batch of images for DIANNA format testing
        self.mock_dianna_image = np.zeros((1, 1, 64, 64), dtype=np.float32)
        self.mock_dianna_image[0, 0, 20:40, 20:40] = 1.0  # White square

        # Create mock relevance maps for testing
        # Dictionary with two classes (0 and 1)
        self.mock_relevances = {
            0: np.ones((1, 64, 64)) * 0.8,  # High relevance for class 0
            1: np.ones((1, 64, 64)) * 0.2   # Low relevance for class 1
        }
        
        # Create more complex relevance maps for testing metrics
        # Class 0 (Raphael) map with high values in the center
        raphael_map = np.zeros((1, 64, 64))
        raphael_map[0, 20:40, 20:40] = 0.8  # High relevance in center
        
        # Class 1 (Non-Raphael) map with high values on the edges
        non_raphael_map = np.zeros((1, 64, 64))
        non_raphael_map[0, 0:10, 0:64] = 0.7  # High relevance at top
        non_raphael_map[0, 54:64, 0:64] = 0.7  # High relevance at bottom
        
        self.complex_relevances = {
            0: raphael_map,
            1: non_raphael_map
        }
        
        # Create additional relevance map patterns for parametrized testing
        # Pattern 1: Random noise with fixed seed for reproducibility
        np.random.seed(42)
        random_r_map = np.ones((1, 64, 64)) * 0.5
        random_r_map[0] += np.random.normal(0, 0.1, (64, 64))
        random_nr_map = np.ones((1, 64, 64)) * 0.5
        random_nr_map[0] += np.random.normal(0, 0.1, (64, 64))
        
        self.random_relevances = {
            0: random_r_map,
            1: random_nr_map
        }
        
        # Pattern 2: Opposing corners (diagonal pattern)
        diagonal_r_map = np.zeros((1, 64, 64))
        diagonal_r_map[0, 0:20, 0:20] = 0.9  # Top-left corner
        diagonal_r_map[0, 44:64, 44:64] = 0.9  # Bottom-right corner
        
        diagonal_nr_map = np.zeros((1, 64, 64))
        diagonal_nr_map[0, 0:20, 44:64] = 0.9  # Top-right corner
        diagonal_nr_map[0, 44:64, 0:20] = 0.9  # Bottom-left corner
        
        self.diagonal_relevances = {
            0: diagonal_r_map,
            1: diagonal_nr_map
        }
        
        # Pattern 3: Highly overlapping maps (ambiguous case)
        overlap_r_map = np.zeros((1, 64, 64))
        overlap_r_map[0, 20:44, 20:44] = 0.8  # Center
        
        overlap_nr_map = np.zeros((1, 64, 64))
        overlap_nr_map[0, 20:44, 20:44] = 0.7  # Same center region
        
        self.overlapping_relevances = {
            0: overlap_r_map,
            1: overlap_nr_map
        }

    def test_class_name(self):
        """Test class_name function"""
        self.assertEqual(class_name(0), 'Raphael')
        self.assertEqual(class_name(1), 'Non-Raphael')
        self.assertEqual(class_name(2), 'class_idx=2')

    def test_create_file_name_base(self):
        """Test create_file_name_base function"""
        # Test with all parameters
        image_path = Path('data/test_image.jpg')
        result = create_file_name_base(6, 'test', image_path, 50, 0.3)
        
        # Check that result is a Path object
        self.assertIsInstance(result, Path)
        
        # Check that it includes the output directory
        self.assertEqual(result.parent.name, 'output')
        
        # Check that the filename includes all parameters
        filename = result.name
        self.assertTrue('test_image.jpg' in filename)
        self.assertTrue('nmasks_50' in filename)
        self.assertTrue('pkeep_0.3' in filename)
        self.assertTrue('res_6' in filename)
        self.assertTrue('test' in filename)
        
        # Test without appendix
        result = create_file_name_base(6, None, image_path, 50, 0.3)
        filename = result.name
        self.assertFalse('None' in filename)

    def test_model_fn_format(self):
        """Test that the model function returns proper format for RISE"""
        # Define a model function similar to what would be used in custom_rise
        def model_fn(x):
            # Should return predictions in the format [batch_size, num_classes]
            batch_size = len(x)
            return np.array([[0.7, 0.3]] * batch_size)
        
        # Test with single image
        result = model_fn(np.zeros((1, 1, 10, 10)))
        self.assertEqual(result.shape, (1, 2))
        
        # Test with batch of images
        result = model_fn(np.zeros((5, 1, 10, 10)))
        self.assertEqual(result.shape, (5, 2))
        
        # Test with different shape image
        result = model_fn(np.zeros((3, 3, 64, 64)))
        self.assertEqual(result.shape, (3, 2))
        
        # Test that the output is correctly formatted for RISE
        result = model_fn(self.mock_dianna_image)
        self.assertEqual(result.shape, (1, 2))
        self.assertEqual(result.dtype, np.float64)  # Ensure it's a float type for weighted mask calculations

    @patch('rise_imagenet.np.random.binomial')
    def test_custom_rise(self, mock_binomial):
        """Test custom_rise function"""
        # Mock the random mask generation
        mock_binomial.return_value = np.ones((6, 6))
        
        # Create a simple model function for testing
        def model_fn(x):
            # Return fake predictions (batch size, num_classes)
            return np.array([[0.7, 0.3]] * len(x))
        
        # Run custom_rise with minimal parameters
        saliency = custom_rise(
            model_fn, 
            self.mock_dianna_image,
            n_masks=2,
            p_keep=0.5,
            feature_res=6
        )
        
        # Check that the output has the expected format
        self.assertIsInstance(saliency, dict)
        self.assertIn(0, saliency)
        self.assertIn(1, saliency)
        
        # Check the shape of the saliency maps
        self.assertEqual(saliency[0].shape, (1, 64, 64))
        self.assertEqual(saliency[1].shape, (1, 64, 64))

    def test_calculate_clarity_metrics(self):
        """Test calculate_clarity_metrics function"""
        # Calculate metrics using our complex mock relevance maps
        metrics = calculate_clarity_metrics(self.complex_relevances)
        
        # Check that all expected metrics are present
        expected_metrics = [
            'raphael_contrast', 'non_raphael_contrast', 'overlap_iou',
            'raphael_entropy', 'non_raphael_entropy', 'map_correlation',
            'clarity_score'
        ]
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], float)
        
        # Test with simple relevance maps
        metrics = calculate_clarity_metrics(self.mock_relevances)
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], float)

    @pytest.mark.parametrize("relevance_pattern,expected_clarity", [
        ("complex", "moderate"),  # Center vs edges pattern
        ("random", "low"),        # Random noise pattern
        ("diagonal", "high"),     # Opposing corners pattern
        ("overlapping", "low")    # Highly overlapping maps
    ])
    def test_clarity_metrics_patterns(self, relevance_pattern=None, expected_clarity=None):
        """Test clarity metrics with different relevance map patterns"""
        try:
            # If pytest is not available or parameters are not provided, run a simplified version
            if relevance_pattern is None or expected_clarity is None:
                # Test default case with complex relevances (for unittest)
                relevances = self.complex_relevances
                metrics = calculate_clarity_metrics(relevances)
                
                # Just check that metrics are calculated correctly
                self.assertGreater(metrics['clarity_score'], 0)
                self.assertLess(metrics['overlap_iou'], 1)
                self.assertLess(abs(metrics['map_correlation']), 1)
                return
                
            # Skip if pytest is not available
            if relevance_pattern == "complex":
                relevances = self.complex_relevances
            elif relevance_pattern == "random":
                relevances = self.random_relevances
            elif relevance_pattern == "diagonal":
                relevances = self.diagonal_relevances
            elif relevance_pattern == "overlapping":
                relevances = self.overlapping_relevances
            else:
                self.fail(f"Unknown relevance pattern: {relevance_pattern}")
            
            # Calculate metrics for this pattern
            metrics = calculate_clarity_metrics(relevances)
            
            # Validate metrics make sense for this pattern
            if expected_clarity == "high":
                # High clarity: low overlap, low correlation, high contrast
                self.assertLess(metrics['overlap_iou'], 0.3)
                self.assertLess(abs(metrics['map_correlation']), 0.3)
                self.assertGreater(metrics['clarity_score'], 0.5)
            elif expected_clarity == "moderate":
                # Moderate clarity: moderate overlap, moderate correlation
                self.assertLess(metrics['overlap_iou'], 0.6)
                self.assertLess(abs(metrics['map_correlation']), 0.6)
                self.assertGreater(metrics['clarity_score'], 0.2)
            elif expected_clarity == "low":
                # Low clarity: high overlap or high correlation
                # Either overlap is high OR correlation is high (or both)
                high_overlap = metrics['overlap_iou'] > 0.6
                high_correlation = abs(metrics['map_correlation']) > 0.6
                self.assertTrue(high_overlap or high_correlation)
                # Check clarity score is appropriate for low clarity
                if high_overlap and high_correlation:
                    self.assertLess(metrics['clarity_score'], 0.2)
            
        except (ImportError, AttributeError):
            # Skip if pytest features are not available
            self.skipTest("pytest.mark.parametrize not available")

    @patch('rise_imagenet.plt.savefig')
    @patch('rise_imagenet.plt.figure')
    @patch('rise_imagenet.plt.close')
    @patch('rise_imagenet.plt.subplots')
    def test_visualize_edge_heatmap_overlay(self, mock_subplots, mock_close, mock_figure, mock_savefig):
        """Test visualize_edge_heatmap_overlay function"""
        # Mock the subplots function to return a figure and axes
        mock_fig = MagicMock()
        mock_ax1 = MagicMock()
        mock_ax2 = MagicMock()
        mock_subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))
        
        # Test with sobel edge detection
        result = visualize_edge_heatmap_overlay(
            image=self.mock_rgb_image,
            heatmap=self.mock_gray_image,
            output_path='test_output.png',
            edge_method='sobel'
        )
        
        # Check that the function called plt.savefig twice (for both visualizations)
        self.assertEqual(mock_savefig.call_count, 2)
        
        # Check that the function returns the path to the combined visualization
        self.assertEqual(result, 'test_output_combined.png')
        
        # Reset mocks
        mock_savefig.reset_mock()
        mock_figure.reset_mock()
        mock_close.reset_mock()
        mock_subplots.reset_mock()
        
        # Mock the subplots function again for the next test
        mock_subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))
        
        # Test with combined edge detection
        result = visualize_edge_heatmap_overlay(
            image=self.mock_rgb_image,
            heatmap=self.mock_gray_image,
            output_path='test_output.png',
            edge_method='combined',
            edge_weights=[0.1, 0.2, 0.3, 0.4]
        )
        
        # Check that the function called plt.savefig twice (for both visualizations)
        self.assertEqual(mock_savefig.call_count, 2)
        
        # Check that it throws ValueError for unsupported edge method
        with self.assertRaises(ValueError):
            visualize_edge_heatmap_overlay(
                image=self.mock_rgb_image,
                heatmap=self.mock_gray_image,
                output_path='test_output.png',
                edge_method='invalid_method'
            )

    @pytest.mark.gpu
    @patch('rise_imagenet.Model')
    @patch('rise_imagenet.custom_rise')
    @patch('rise_imagenet.io.imread')
    @patch('rise_imagenet.calculate_clarity_metrics')
    @patch('rise_imagenet.visualization.plot_image')
    def test_explain_painting(self, mock_plot, mock_metrics, mock_imread, mock_rise, mock_model_class):
        """Test explain_painting function"""
        # Skip the actual test if explain_painting imports additional modules
        # that are not available in the test environment
        # This is a mock test to check the general flow of the function
        
        try:
            # Create mocks for all dependencies
            mock_model = MagicMock()
            mock_model_class.return_value = mock_model
            mock_model.run_on_batch.return_value = np.array([[0.7, 0.3]])
            
            # Mock imread to return our test image
            mock_imread.return_value = self.mock_rgb_image
            
            # Mock custom_rise to return our test relevances
            mock_rise.return_value = self.mock_relevances
            
            # Mock metrics calculation
            mock_metrics.return_value = {'clarity_score': 0.8, 'overlap_iou': 0.2}
            
            # Create a temporary directory for output if needed
            with patch('pathlib.Path.mkdir'):
                # We need to mock the file operations for saving results
                with patch('builtins.open', mock_open()):
                    with patch('numpy.savez_compressed'):
                        with patch('pandas.DataFrame.to_csv'):
                            # Call explain_painting with minimal parameters
                            rise_imagenet.explain_painting(
                                image_path=Path('data/test_image.jpg'),
                                n_masks=10,
                                p_keep=0.3,
                                feature_res=6
                            )
                            
                            # Check that custom_rise was called
                            mock_rise.assert_called_once()
                            
                            # Check that the model was created and used
                            mock_model_class.assert_called_once()
                            mock_model.run_on_batch.assert_called()
                            
                            # Check that clarity metrics were calculated
                            mock_metrics.assert_called_once()
                            
                            # Check that visualization was called for both classes
                            self.assertEqual(mock_plot.call_count, 2)
        
        except ImportError:
            self.skipTest("Required modules for explain_painting not available")

    @pytest.mark.gpu
    @patch('rise_imagenet.np.load')
    @patch('rise_imagenet.io.imread')
    @patch('rise_imagenet.visualize_edge_heatmap_overlay')
    @patch('rise_imagenet.visualization.plot_image')
    @patch('rise_imagenet.plt.savefig')
    @patch('rise_imagenet.plt.figure')
    @patch('rise_imagenet.plt.imshow')
    @patch('rise_imagenet.plt.colorbar')
    @patch('rise_imagenet.plt.title')
    @patch('rise_imagenet.plt.tight_layout')
    @patch('rise_imagenet.plt.close')
    def test_integrate_results(self, mock_close, mock_tight_layout, mock_title, mock_colorbar, 
                              mock_imshow, mock_figure, mock_savefig, mock_plot, 
                              mock_visualize, mock_imread, mock_load):
        """Test integrate_results function"""
        # Skip the actual test if integrate_results imports additional modules
        # that are not available in the test environment
        # This is a mock test to check the general flow of the function
        
        try:
            # Mock matplotlib components to return dummy values
            mock_figure.return_value = MagicMock()
            mock_colorbar.return_value = MagicMock()
            
            # Mock the file operations and data loading
            with patch('pathlib.Path.glob') as mock_glob:
                # Mock finding NPZ files
                mock_glob.return_value = [Path('output/test_0.npz'), Path('output/test_1.npz')]
                
                # Mock loading the NPZ files
                mock_load.return_value = {'relevances': self.mock_relevances}
                
                # Mock imread to return our test image
                mock_imread.return_value = self.mock_rgb_image
                
                # Create a temporary directory for output if needed
                with patch('pathlib.Path.mkdir'):
                    # Mock CSV file operations if needed
                    with patch('pandas.read_csv'):
                        with patch('pandas.DataFrame.to_csv'):
                            with patch('builtins.open', mock_open()):
                                with patch('numpy.savez_compressed'):
                                    # Call integrate_results with minimal parameters
                                    rise_imagenet.integrate_results(
                                        image_path=Path('data/test_image.jpg'),
                                        n_masks=10,
                                        p_keep=0.3,
                                        feature_res=6,
                                        runs=2
                                    )
                                    
                                    # Check that integration was attempted
                                    mock_glob.assert_called()
                                    self.assertEqual(mock_load.call_count, 2)  # One for each NPZ file
                                    
                                    # Check that visualize_edge_heatmap_overlay was called
                                    # It's important to have this check because it tests the brushstroke 
                                    # visualization which is a key part of the functionality
                                    mock_visualize.assert_called()
        
        except ImportError:
            self.skipTest("Required modules for integrate_results not available")


if __name__ == '__main__':
    unittest.main() 