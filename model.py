import glob
import logging
import math
import os
import warnings
from pathlib import Path

import joblib
import keras
from keras import models
import numpy as np
from diskcache import Cache
from skimage import io, color, feature, filters
from tqdm import tqdm

# Suppress tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow INFO and WARNING messages
logging.getLogger('tensorflow').setLevel(logging.ERROR)  # Only show ERROR messages
warnings.filterwarnings('ignore', category=UserWarning)

class Model:
    def __init__(self):
        self.resnet_model = None
        try:          
            self.resnet_model = models.load_model("models/resnet50_model.h5", compile=False)
        except Exception as e:
            print(f"Error loading ResNet model: {str(e)}")
            raise RuntimeError(f"ResNet model is required but could not be loaded: {str(e)}")

    def extract_features(self, img):
        if self.resnet_model is not None:
            # Preprocess input for ResNet model
            img_preprocessed = keras.applications.resnet50.preprocess_input(img)
            return self.resnet_model.predict(img_preprocessed, verbose=0)
        return None

    def run_on_batch(self, input):
        """
        Run the model on a batch of input images.
        
        Args:
            input: Input images in the format [batch, height, width, channels] 
            or [batch, channels, height, width]
        
        Returns:
            A list of probabilities for each image in the batch
        """
        
        if input is None or input.shape[0] == 0:
            raise ValueError("Input cannot be None or empty batch")
            
        # Create a deep copy of the input to avoid modifying the original
        input_copy = input.copy()
        
        # Handle potential input in format [batch, channels, height, width]
        if len(input_copy.shape) == 4 and input_copy.shape[1] <= 4:
            input_copy = np.transpose(input_copy, (0, 2, 3, 1))
        
        # Ensure we have a batch dimension
        if len(input_copy.shape) == 3:
            input_copy = np.expand_dims(input_copy, axis=0)
        
        # Special handling for 5D input (e.g., multiple batches)
        if len(input_copy.shape) == 5:
            input_copy = input_copy.reshape(-1, *input_copy.shape[2:])
            
        # Convert grayscale to RGB if needed
        if input_copy.shape[-1] == 1:
            input_copy = np.repeat(input_copy, 3, axis=-1)
            
        # Ensure input is normalized to [0, 1] range
        if input_copy.max() > 1.0:
            input_copy = input_copy / 255.0
        
        # Process each image in the batch with a progress indicator
        results = []
        
        for i, img in enumerate(tqdm(input_copy, desc="Analyzing images", leave=False)):
            # Compare with the dataset - use the correct path to Not Raphael folder
            probabilities = compare_image_with_dataset(img, 'data/Not Raphael/')
            results.append(probabilities)
        
        # Ensure results are in the format [batch_size, num_classes]
        results = np.array(results)
        
        return results

cache = Cache('my_cache_directory')

def scale_inverse_log(x, x_min, x_max, y_min, y_max):
    # Check input boundaries
    if x < x_min or x > x_max:
        return "Input x must be within the range [x_min, x_max]"
    
    # Prevent division by zero
    epsilon = 1e-10
    
    # Calculate inverse log of x
    inv_log_x = -1 / (math.log(x + 1) + epsilon)
    
    # Calculate inverse log of x_min and x_max
    inv_log_x_min = -1 / (math.log(x_min + 1) + epsilon)
    inv_log_x_max = -1 / (math.log(x_max + 1) + epsilon)
    
    # Scale the inverse logarithmic value to the target range [y_min, y_max]
    y = y_min + (inv_log_x - inv_log_x_min) * (y_max - y_min) / (inv_log_x_max - inv_log_x_min)
    
    return y


def extract_features(img_path, model):
    # Handle both file paths and numpy arrays
    if isinstance(img_path, (str, Path)):
        img = io.imread(str(img_path))
    else:
        img = img_path
    
    # If img has extra dimensions (like masks), flatten it before passing to the model
    if len(img.shape) == 5:
        img = np.reshape(img, (-1, img.shape[2], img.shape[3], img.shape[4]))
    
    # Expand dimensions if needed (batch size dimension)
    if len(img.shape) == 3:
        img = np.expand_dims(img, axis=0)
    
    # Use the model's preprocessing if available
    if model is not None:
        try:
            img = keras.applications.resnet50.preprocess_input(img)
            # Use verbose=0 to suppress progress bar output
            features = model.predict(img)
            return features
        except Exception:
            return None
    return None


# Helper function to convert any image format to 2D grayscale
def _convert_to_grayscale(img):
    # Convert to grayscale if the image is in color
    if len(img.shape) > 2:
        # Handle DIANNA format (batch, channel, height, width)
        if len(img.shape) == 4 and img.shape[0] == 1 and img.shape[1] == 1:
            # Extract the image from batch and channel dimensions
            gray = img[0, 0]
        # Handle RGB format
        elif img.shape[-1] > 1:
            gray = color.rgb2gray(img)
        # Handle grayscale with extra dimensions
        else:
            gray = img.squeeze()
    else:
        gray = img
    
    # Ensure we have a 2D array
    if len(gray.shape) != 2:
        raise ValueError(f"Failed to convert image to 2D grayscale. Shape: {gray.shape}")
    
    return gray


# Function to calculate edge features using Canny edge detector
def calculate_canny_edges(img):
    # Convert to grayscale
    gray = _convert_to_grayscale(img)
    
    # Apply Canny edge detection using scikit-image
    edges = feature.canny(gray, sigma=1.0)
    
    # Return standard deviation of edge image
    return np.std(edges)


# Function to calculate edge features using Sobel operator
def calculate_sobel_edges(img):
    # Convert to grayscale
    gray = _convert_to_grayscale(img)
    
    # Apply Sobel filter using scikit-image
    sobelx = filters.sobel_h(gray)
    sobely = filters.sobel_v(gray)
    
    return np.std(sobelx), np.std(sobely)


# Function to calculate edge features using Laplacian operator
def calculate_laplacian_edges(img):
    # Convert to grayscale
    gray = _convert_to_grayscale(img)
    
    # Apply Laplacian filter using scikit-image
    laplacian = filters.laplace(gray)
    
    return np.std(laplacian)


# Function to calculate edge features using Scharr operator
def calculate_scharr_edges(img):
    # Convert to grayscale
    gray = _convert_to_grayscale(img)
    
    # Apply Scharr filter using scikit-image
    scharrx = filters.scharr_h(gray)
    scharry = filters.scharr_v(gray)
    
    return np.std(scharrx), np.std(scharry)


# Function to calculate all edge features
def calculate_features(img):
    canny_edges = calculate_canny_edges(img)
    sobel_edges_x, sobel_edges_y = calculate_sobel_edges(img)
    laplacian_edges = calculate_laplacian_edges(img)
    scharr_edges_x, scharr_edges_y = calculate_scharr_edges(img)

    return np.array([canny_edges,
                     sobel_edges_x, sobel_edges_y,
                     laplacian_edges,
                     scharr_edges_x, scharr_edges_y])


def compare_image_with_dataset(test_image_path, image_dir):
    resnet50_path: Path = Path("models/resnet50_model.h5")
    model_path: Path = Path("models/28_09_2023_svm_final_model.pkl")    
    Model_Path = model_path
    ResNet_Path = resnet50_path

    # Use the provided image array directly; it's already a numpy array
    test_image = test_image_path
    
    # Ensure image is in correct format for feature calculation
    # If it's in DIANNA format [batch, channels, height, width], transpose it
    if len(test_image.shape) == 4 and 1 <= test_image.shape[1] <= 4:
        test_image = np.transpose(test_image, (0, 2, 3, 1))
    
    # Ensure it has the right number of dimensions for feature calculation
    if len(test_image.shape) == 4:
        # Take the first image if batched
        test_image = test_image[0]

    # Load the final model
    try:
        svm_final = joblib.load(Model_Path)
    except Exception:
        # Handle the case where the model file is not found
        print(f"Warning: SVM model not found at {Model_Path}")
        raise RuntimeError("SVM model is required but could not be loaded")

    # Load the ResNet model
    try:
        try:
            # Try the newer import pattern
            model = keras.models.load_model(ResNet_Path, compile=False)
        except Exception:
            # Fall back to direct models import
            model = models.load_model(ResNet_Path, compile=False)
        
        # Explicit compilation not needed when using compile=False
        # and only performing inference operations
    except Exception:        
        print(f"Warning: ResNet model not found at {ResNet_Path}")
        model = None

    # Extract features from the test image
    test_image_features = extract_features(test_image, model)
    
    # If feature extraction failed, return default probabilities
    if test_image_features is None:
        raise RuntimeError("Feature extraction failed")

    # Reshape features if needed
    if len(test_image_features.shape) > 1:
        test_image_features = test_image_features.reshape(-1)

    # Use the loaded model to predict the category of the test image
    #predicted_category = svm_final.predict([test_image_features])[0]

    # Calculate probabilities for each category
    probabilities = svm_final.predict_proba([test_image_features])[0]

    #categories = ['Raphael', 'Not Raphael']

    # Calculate features of test image
    test_features = calculate_features(test_image)

    # Normalize test features to get weights
    weights = test_features / np.sum(test_features)

    # Load all images in directory
    formats = ('*.jpg', '*.png', '*.bmp')

    image_paths = []

    for fmt in formats:
        image_paths.extend(glob.glob(f"{image_dir}/{fmt}"))
    
    # Calculate the total feature values and the count of images
    total_features = np.zeros_like(test_features)
    image_count = 0

    for image_path in tqdm(image_paths, desc="Analyzing reference images", leave=False):
        # Load image
        image_features = load_image_and_calculate_features(image_path)

        # Add to total and increment count (multiply by weights here)
        total_features += image_features * weights
        image_count += 1

    # Calculate the weighted average feature values
    average_features = total_features / image_count if image_count else np.zeros_like(test_features)

    # Compare average features with test image
    difference = np.abs(test_features - average_features)

    # Sum of differences
    mean_diff = np.mean(difference)
    
    if mean_diff < 99:
        mean_diff = 400
        probabilities[0] = probabilities[0] - 0.5

    if mean_diff > 400:
        mean_diff = 400
        probabilities[0] = probabilities[0] - 0.5

    if mean_diff < 150:
        mean_diff = 150

    scale_ = scale_inverse_log(mean_diff, x_min=150, x_max=400, y_min=0.0, y_max=-0.99)

    threshold = 0.95 * probabilities[0] + 0.05 * scale_
    if threshold < 0:
        threshold = 0.05

    final_probabilities = [threshold, 1 - threshold]

    # Format percentages for clean display
    raphael_pct = final_probabilities[0] * 100
    non_raphael_pct = final_probabilities[1] * 100
    
    # Display simple prediction result as percentages
    print(f"Prediction: Raphael: {raphael_pct:.1f}%, Non-Raphael: {non_raphael_pct:.1f}%")
    
    return final_probabilities


@cache.memoize()
def load_image_and_calculate_features(image_path):
    image = io.imread(image_path)
    # Calculate features of image
    image_features = calculate_features(image)
    return image_features