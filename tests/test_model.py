import unittest
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os
from pathlib import Path

# Add parent directory to path to import model
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model import Model, extract_features, calculate_canny_edges, calculate_sobel_edges
from model import calculate_laplacian_edges, calculate_scharr_edges, calculate_features
from model import compare_image_with_dataset, scale_inverse_log


class TestModel(unittest.TestCase):
    """Test cases for the Model class and its methods"""

    def setUp(self):
        """Set up test fixtures"""
        # Create a mock image for testing
        self.mock_image = np.zeros((224, 224, 3), dtype=np.uint8)
        # Add some patterns for edge detection
        self.mock_image[100:120, 100:120, :] = 255  # White square
        
        # Create a mock batch of images
        self.mock_batch = np.zeros((2, 224, 224, 3), dtype=np.uint8)
        self.mock_batch[0, 100:120, 100:120, :] = 255  # White square in first image
        self.mock_batch[1, 50:70, 50:70, :] = 255  # White square in second image
        
        # Create a mock DIANNA-format image (batch, channels, height, width)
        self.mock_dianna_image = np.zeros((1, 1, 224, 224), dtype=np.uint8)
        self.mock_dianna_image[0, 0, 100:120, 100:120] = 255  # White square

    @patch('model.keras.models.load_model')
    def test_model_init(self, mock_load_model):
        """Test Model initialization"""
        # Test successful model loading
        model = Model()
        mock_load_model.assert_called_once()
        
        # Test handling when model loading fails
        mock_load_model.side_effect = Exception("Model not found")
        model = Model()
        self.assertIsNone(model.resnet_model)

    @patch.object(Model, 'extract_features')
    @patch('model.compare_image_with_dataset')
    def test_run_on_batch_with_file_path(self, mock_compare, mock_extract):
        """Test run_on_batch with file path input"""
        mock_compare.return_value = np.array([0.7, 0.3])
        
        with patch('model.io.imread') as mock_imread:
            mock_imread.return_value = self.mock_image
            model = Model()
            result = model.run_on_batch('fake/path.jpg')
            
            # Check that imread was called with the path
            mock_imread.assert_called_once_with('fake/path.jpg')
            # Check that compare_image_with_dataset was called
            mock_compare.assert_called_once()
            # Check the result shape
            self.assertEqual(result.shape, (1, 2))
            np.testing.assert_array_equal(result, np.array([[0.7, 0.3]]))

    def test_run_on_batch_with_numpy_array(self):
        """Test run_on_batch with numpy array input"""
        with patch('model.compare_image_with_dataset') as mock_compare:
            mock_compare.return_value = np.array([0.7, 0.3])
            
            model = Model()
            result = model.run_on_batch(self.mock_image)
            
            # Check that compare_image_with_dataset was called with the image
            mock_compare.assert_called_once()
            # Check the result shape
            self.assertEqual(result.shape, (1, 2))
            np.testing.assert_array_equal(result, np.array([[0.7, 0.3]]))

    def test_run_on_batch_with_batch(self):
        """Test run_on_batch with batch input"""
        with patch('model.compare_image_with_dataset') as mock_compare:
            mock_compare.return_value = np.array([0.7, 0.3])
            
            model = Model()
            result = model.run_on_batch(self.mock_batch)
            
            # Check that compare_image_with_dataset was called with the batch
            mock_compare.assert_called_once()
            # Check the result shape (should still be (1, 2) as our mock returns one prediction)
            self.assertEqual(result.shape, (1, 2))

    def test_run_on_batch_with_dianna_format(self):
        """Test run_on_batch with DIANNA format input"""
        with patch('model.compare_image_with_dataset') as mock_compare:
            mock_compare.return_value = np.array([0.7, 0.3])
            
            model = Model()
            result = model.run_on_batch(self.mock_dianna_image)
            
            # Check that compare_image_with_dataset was called
            mock_compare.assert_called_once()
            # Check the result shape
            self.assertEqual(result.shape, (1, 2))

    def test_extract_features(self):
        """Test extract_features function"""
        # Create a mock TensorFlow model
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([[0.1, 0.2, 0.3, 0.4]])
        
        # Test with a numpy array
        features = extract_features(self.mock_image, mock_model)
        self.assertIsNotNone(features)
        self.assertEqual(features.shape, (1, 4))
        
        # Test with a file path
        with patch('model.io.imread') as mock_imread:
            mock_imread.return_value = self.mock_image
            features = extract_features('fake/path.jpg', mock_model)
            self.assertIsNotNone(features)
            self.assertEqual(features.shape, (1, 4))
        
        # Test with model is None
        features = extract_features(self.mock_image, None)
        self.assertIsNone(features)
        
        # Test with model prediction failing
        mock_model.predict.side_effect = Exception("Prediction failed")
        features = extract_features(self.mock_image, mock_model)
        self.assertIsNone(features)

    def test_calculate_edge_features(self):
        """Test edge feature calculation functions"""
        # Test Canny edge detection
        canny_edges = calculate_canny_edges(self.mock_image)
        self.assertIsInstance(canny_edges, float)
        
        # Test Sobel edge detection
        sobel_x, sobel_y = calculate_sobel_edges(self.mock_image)
        self.assertIsInstance(sobel_x, float)
        self.assertIsInstance(sobel_y, float)
        
        # Test Laplacian edge detection
        laplacian_edges = calculate_laplacian_edges(self.mock_image)
        self.assertIsInstance(laplacian_edges, float)
        
        # Test Scharr edge detection
        scharr_x, scharr_y = calculate_scharr_edges(self.mock_image)
        self.assertIsInstance(scharr_x, float)
        self.assertIsInstance(scharr_y, float)
        
        # Test calculate_features that combines all edge features
        features = calculate_features(self.mock_image)
        self.assertEqual(features.shape, (6,))  # 6 features: canny, sobel_x, sobel_y, laplacian, scharr_x, scharr_y

    @patch('model.joblib.load')
    @patch('model.tf.keras.models.load_model')
    @patch('model.extract_features')
    def test_compare_image_with_dataset(self, mock_extract_features, mock_load_model, mock_joblib_load):
        """Test compare_image_with_dataset function"""
        # Mock the SVM model
        mock_svm = MagicMock()
        mock_svm.predict_proba.return_value = np.array([[0.3, 0.7]])
        mock_joblib_load.return_value = mock_svm
        
        # Mock the ResNet model
        mock_resnet = MagicMock()
        mock_load_model.return_value = mock_resnet
        
        # Mock extract_features to return a feature vector
        mock_extract_features.return_value = np.array([0.1, 0.2, 0.3, 0.4])
        
        # Test with a numpy array
        result = compare_image_with_dataset(self.mock_image, 'fake/dir/')
        self.assertEqual(len(result), 2)  # Should return [raphael_prob, non_raphael_prob]
        
        # Test with a batch of images (DIANNA format)
        result = compare_image_with_dataset(self.mock_dianna_image, 'fake/dir/')
        self.assertEqual(len(result), 2)
        
        # Test when SVM model fails to load
        mock_joblib_load.side_effect = Exception("SVM model not found")
        result = compare_image_with_dataset(self.mock_image, 'fake/dir/')
        self.assertEqual(result, [0.5, 0.5])  # Should return default probabilities
        
        # Test when ResNet model fails to load
        mock_joblib_load.side_effect = None
        mock_load_model.side_effect = Exception("ResNet model not found")
        result = compare_image_with_dataset(self.mock_image, 'fake/dir/')
        self.assertEqual(len(result), 2)  # Should still return probabilities
        
        # Test when feature extraction fails
        mock_extract_features.return_value = None
        result = compare_image_with_dataset(self.mock_image, 'fake/dir/')
        self.assertEqual(result, [0.5, 0.5])  # Should return default probabilities

    def test_scale_inverse_log(self):
        """Test scale_inverse_log function"""
        # Test normal case
        result = scale_inverse_log(0.5, 0.0, 1.0, 0.0, 1.0)
        self.assertIsInstance(result, float)
        self.assertTrue(0.0 <= result <= 1.0)
        
        # Test with x outside the range
        result = scale_inverse_log(-0.1, 0.0, 1.0, 0.0, 1.0)
        self.assertIsInstance(result, str)  # Should return an error message
        
        result = scale_inverse_log(1.1, 0.0, 1.0, 0.0, 1.0)
        self.assertIsInstance(result, str)  # Should return an error message
        
        # Test with different output range
        result = scale_inverse_log(0.5, 0.0, 1.0, -1.0, 1.0)
        self.assertIsInstance(result, float)
        self.assertTrue(-1.0 <= result <= 1.0)


if __name__ == '__main__':
    unittest.main() 