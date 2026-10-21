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
from model import compare_image_with_dataset, scale_inverse_log, preprocess_image


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
        self.mock_dianna_image = np.zeros((1, 3, 224, 224), dtype=np.uint8)
        self.mock_dianna_image[0, :, 100:120, 100:120] = 255  # White square

    @patch('model.models.load_model')
    def test_model_init(self, mock_load_model):
        """Test Model initialization"""
        # Test successful model loading
        model = Model()
        mock_load_model.assert_called_once()

        # Reset mock for the next call and set side effect
        mock_load_model.reset_mock()
        mock_load_model.side_effect = Exception("Model not found")

        # Test handling when model loading fails
        with self.assertRaises(Exception):  # Changed to catch any Exception
            model = Model()

    @patch('model.compare_image_with_dataset')
    def test_run_on_batch_with_file_path(self, mock_compare):
        """Test run_on_batch with file path input"""
        mock_compare.return_value = np.array([0.7, 0.3])

        with patch('model.io.imread') as mock_imread:
            mock_imread.return_value = self.mock_image

            # Need to patch the Model.__init__ to avoid loading the real model
            with patch.object(Model, '__init__', return_value=None):
                model = Model()
                model.resnet_model = MagicMock()  # Mock the resnet_model property

                # Test with file path input
                result = model.run_on_batch('fake/path.jpg')

                # Check that compare_image_with_dataset was called
                mock_compare.assert_called_once()
                # Check the result shape - for file paths, we expect direct probabilities
                self.assertEqual(len(result), 2)
                np.testing.assert_array_equal(result, np.array([0.7, 0.3]))

    def test_run_on_batch_with_numpy_array(self):
        """Test run_on_batch with numpy array input"""
        with patch('model.compare_image_with_dataset') as mock_compare:
            mock_compare.return_value = np.array([0.7, 0.3])

            # Need to patch the Model.__init__ to avoid loading the real model
            with patch.object(Model, '__init__', return_value=None):
                model = Model()
                model.resnet_model = MagicMock()  # Mock the resnet_model property

                result = model.run_on_batch(self.mock_image)

                # Check that compare_image_with_dataset was called
                mock_compare.assert_called_once()
                # Check the result shape - should be [batch_size, num_classes]
                self.assertEqual(result.shape[0], 1)  # Batch size 1
                self.assertEqual(result.shape[1], 2)  # Two classes
                np.testing.assert_array_equal(result[0], np.array([0.7, 0.3]))

    def test_run_on_batch_with_batch(self):
        """Test run_on_batch with batch input"""
        with patch('model.compare_image_with_dataset') as mock_compare:
            mock_compare.side_effect = [np.array([0.7, 0.3]), np.array([0.6, 0.4])]

            # Need to patch the Model.__init__ to avoid loading the real model
            with patch.object(Model, '__init__', return_value=None):
                model = Model()
                model.resnet_model = MagicMock()  # Mock the resnet_model property

                results = model.run_on_batch(self.mock_batch)

                # Check that compare_image_with_dataset was called twice, once for each image
                self.assertEqual(mock_compare.call_count, 2)
                # Check the results shape
                self.assertEqual(len(results), 2)
                # It should be a numpy array of shape (2, 2) - two images, two classes each
                self.assertEqual(results.shape[0], 2)
                self.assertEqual(results.shape[1], 2)

    def test_run_on_batch_with_dianna_format(self):
        """Test run_on_batch with DIANNA format input"""
        with patch('model.compare_image_with_dataset') as mock_compare:
            mock_compare.return_value = np.array([0.7, 0.3])

            # Need to patch the Model.__init__ to avoid loading the real model
            with patch.object(Model, '__init__', return_value=None):
                model = Model()
                model.resnet_model = MagicMock()  # Mock the resnet_model property

                # Patch preprocess_image to handle the DIANNA format correctly
                with patch('model.preprocess_image') as mock_preprocess:
                    # Return a properly formatted image and grayscale version
                    mock_preprocess.return_value = (np.zeros((224, 224, 3)), np.zeros((224, 224)))

                    result = model.run_on_batch(self.mock_dianna_image)

                    # Check that compare_image_with_dataset was called
                    mock_compare.assert_called_once()
                    # Check the result shape - should be [batch_size, num_classes]
                    self.assertEqual(result.shape[0], 1)  # Batch size 1
                    self.assertEqual(result.shape[1], 2)  # Two classes
                    np.testing.assert_array_equal(result[0], np.array([0.7, 0.3]))

    def test_extract_features(self):
        """Test extract_features function"""
        # Create a mock TensorFlow model
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([[0.1, 0.2, 0.3, 0.4]])

        # Test with a numpy array
        with patch('model.preprocess_image') as mock_preprocess:
            mock_preprocess.return_value = (np.zeros((224, 224, 3)), np.zeros((224, 224)))

            with patch('keras.applications.resnet50.preprocess_input') as mock_preprocess_input:
                mock_preprocess_input.return_value = np.zeros((1, 224, 224, 3))

                features = extract_features(self.mock_image, mock_model)
                self.assertIsNotNone(features)
                self.assertEqual(features.shape, (1, 4))

        # Test with a file path
        with patch('model.io.imread') as mock_imread:
            mock_imread.return_value = self.mock_image

            with patch('model.preprocess_image') as mock_preprocess:
                mock_preprocess.return_value = (np.zeros((224, 224, 3)), np.zeros((224, 224)))

                with patch('keras.applications.resnet50.preprocess_input') as mock_preprocess_input:
                    mock_preprocess_input.return_value = np.zeros((1, 224, 224, 3))

                    features = extract_features('fake/path.jpg', mock_model)
                    self.assertIsNotNone(features)
                    self.assertEqual(features.shape, (1, 4))

        # Test with model is None
        features = extract_features(self.mock_image, None)
        self.assertIsNone(features)

        # Test with model prediction failing
        with patch('model.preprocess_image') as mock_preprocess:
            mock_preprocess.return_value = (np.zeros((224, 224, 3)), np.zeros((224, 224)))

            with patch('keras.applications.resnet50.preprocess_input') as mock_preprocess_input:
                mock_preprocess_input.return_value = np.zeros((1, 224, 224, 3))

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
    @patch('model.keras.models.load_model')
    @patch('model.extract_features')
    def test_compare_image_with_dataset(self, mock_extract_features, mock_load_model, mock_joblib_load):
        """Test compare_image_with_dataset function"""
        # Mock the SVM model
        mock_svm = MagicMock()
        mock_svm.predict.return_value = np.array([0])
        mock_svm.predict_proba.return_value = np.array([[0.3, 0.7]])
        mock_joblib_load.return_value = mock_svm

        # Mock the ResNet model
        mock_resnet = MagicMock()
        mock_load_model.return_value = mock_resnet

        # Mock extract_features to return a feature vector
        mock_extract_features.return_value = np.array([0.1, 0.2, 0.3, 0.4])

        # Mock calculate_features
        with patch('model.calculate_features') as mock_calc_features:
            mock_calc_features.return_value = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

            # Mock glob to find reference images
            with patch('glob.glob') as mock_glob:
                mock_glob.return_value = ['ref1.jpg', 'ref2.jpg']

                # Mock cached feature loading
                with patch('model.load_image_and_calculate_features') as mock_load_cached:
                    mock_load_cached.return_value = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

                    # Test with a numpy array
                    with patch('builtins.print') as mock_print:  # Suppress output
                        result = compare_image_with_dataset(self.mock_image, 'fake/dir/')

                        # Check results
                        self.assertEqual(len(result), 2)
                        # Values should be between 0 and 1
                        self.assertTrue(0 <= result[0] <= 1)
                        self.assertTrue(0 <= result[1] <= 1)

    def test_scale_inverse_log(self):
        """Test scale_inverse_log function"""
        # Test normal case with non-zero x_min to avoid division by zero
        result = scale_inverse_log(0.5, 0.01, 1.0, 0.0, 1.0)
        self.assertIsInstance(result, float)
        self.assertTrue(0.0 <= result <= 1.0)

        # Test with x outside the range
        result = scale_inverse_log(-0.1, 0.01, 1.0, 0.0, 1.0)
        self.assertIsInstance(result, str)  # Should return error message as string
        self.assertTrue("Input x must be within the range" in result)


if __name__ == '__main__':
    unittest.main()