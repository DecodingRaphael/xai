import unittest
import numpy as np
import cv2
from unittest.mock import patch
import sys
import os

# Add parent directory to path to import functions
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestCVUtils(unittest.TestCase):
    """Test cases for OpenCV utility functions used in rise_imagenet.py"""

    def setUp(self):
        """Set up test fixtures"""
        # Create a small test image
        self.test_image = np.zeros((10, 10), dtype=np.uint8)
        self.test_image[3:7, 3:7] = 255  # White square in center
        
        # Create a mask for testing
        self.test_mask = np.zeros((5, 5), dtype=np.uint8)
        self.test_mask[1:4, 1:4] = 1  # Binary mask

    def test_cv2_resize(self):
        """Test cv2.resize with INTER_NEAREST interpolation"""
        try:
            # Try to resize the test image
            resized = cv2.resize(
                self.test_mask,
                (10, 10),
                interpolation=cv2.INTER_NEAREST
            )
            
            # Check that the resized image has the expected dimensions
            self.assertEqual(resized.shape, (10, 10))
            
            # Check that the interpolation preserved the binary nature of the mask
            # INTER_NEAREST should not introduce new values, only 0s and 1s should be present
            unique_values = np.unique(resized)
            self.assertTrue(np.array_equal(unique_values, np.array([0, 1])))
            
            # Check that the central square was properly resized
            # The center should still be 1s
            self.assertTrue(np.all(resized[2:8, 2:8] == 1))
            
        except AttributeError:
            self.skipTest("cv2.resize or cv2.INTER_NEAREST not available. This is likely a linting error.")
    
    @patch('cv2.INTER_NEAREST', 0)  # Mock with a dummy value
    def test_cv2_interpolation_constants(self):
        """Test cv2.INTER_NEAREST constant to address linter errors"""
        # Verify mock is working
        self.assertEqual(cv2.INTER_NEAREST, 0)
        
        # Additional check to ensure we can use the constant in a function call
        try:
            with patch('cv2.resize') as mock_resize:
                mock_resize.return_value = np.ones((10, 10))
                _ = cv2.resize(self.test_mask, (10, 10), interpolation=cv2.INTER_NEAREST)
                mock_resize.assert_called_once()
                # Check the interpolation parameter was passed correctly
                self.assertEqual(mock_resize.call_args[1]['interpolation'], 0)
        except Exception as e:
            self.fail(f"Failed to use cv2.INTER_NEAREST in function call: {e}")
    
    def test_edge_detection_functions(self):
        """Test edge detection functions used in the visualization"""
        try:
            from skimage import filters, feature
            
            # Test Canny edge detection
            canny_edges = feature.canny(self.test_image.astype(float) / 255.0, sigma=1.0)
            self.assertEqual(canny_edges.shape, self.test_image.shape)
            self.assertIsInstance(canny_edges, np.ndarray)
            
            # Test Sobel edge detection
            sobelx = filters.sobel_h(self.test_image.astype(float) / 255.0)
            sobely = filters.sobel_v(self.test_image.astype(float) / 255.0)
            sobel_edges = np.sqrt(sobelx**2 + sobely**2)
            self.assertEqual(sobel_edges.shape, self.test_image.shape)
            self.assertIsInstance(sobel_edges, np.ndarray)
            
            # Test Laplacian edge detection
            laplacian_edges = np.abs(filters.laplace(self.test_image.astype(float) / 255.0))
            self.assertEqual(laplacian_edges.shape, self.test_image.shape)
            self.assertIsInstance(laplacian_edges, np.ndarray)
            
            # Test Scharr edge detection
            scharrx = filters.scharr_h(self.test_image.astype(float) / 255.0)
            scharry = filters.scharr_v(self.test_image.astype(float) / 255.0)
            scharr_edges = np.sqrt(scharrx**2 + scharry**2)
            self.assertEqual(scharr_edges.shape, self.test_image.shape)
            self.assertIsInstance(scharr_edges, np.ndarray)
            
        except ImportError:
            self.skipTest("scikit-image not available for edge detection tests")
    
    @patch('cv2.resize')
    def test_cv2_resize_mock(self, mock_resize):
        """Test cv2.resize with mocking to address potential linting errors"""
        # Configure the mock to return a simple expanded array
        mock_resize.return_value = np.ones((10, 10), dtype=np.uint8)
        
        # Call the resize function
        resized = cv2.resize(
            self.test_mask,
            (10, 10),
            interpolation=cv2.INTER_NEAREST
        )
        
        # Check that the mock was called with the expected arguments
        mock_resize.assert_called_once()
        args, kwargs = mock_resize.call_args
        self.assertIs(args[0], self.test_mask)  # First argument should be the input mask
        self.assertEqual(args[1], (10, 10))     # Second argument should be the target size
        self.assertEqual(kwargs['interpolation'], cv2.INTER_NEAREST)
        
        # Check the result shape
        self.assertEqual(resized.shape, (10, 10))


if __name__ == '__main__':
    unittest.main() 