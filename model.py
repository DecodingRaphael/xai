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

# Suppress tensorflow warnings and only show error messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=UserWarning)

class Model:
    def __init__(self):
        self.resnet_model = models.load_model("models/resnet50_model.h5", compile=False)

    def extract_features(self, img):
        preprocess_input = keras.applications.resnet50.preprocess_input(img)
        return self.resnet_model.predict(preprocess_input, verbose=0)

    def run_on_batch(self, input):
        """
        Run the model on a batch of input images.

        Args:
            input: Input images in the format [batch, height, width, channels]
                or a single image as [height, width, channels]
                or a file path

        Returns:
            A numpy array of probabilities for each image in the batch with
            shape [batch_size, num_classes]
        """

        # Handle file path inputs
        if isinstance(input, (str, Path)):
            img = io.imread(str(input))
            img, _ = preprocess_image(img, normalize=True, ensure_rgb=True)
            # For file paths, return direct probabilities to match test expectations
            return compare_image_with_dataset(img, 'data/Not Raphael/')

        # Create a copy to avoid modifying the original
        input_copy = input.copy()

        # For single images, add batch dimension
        if len(input_copy.shape) == 2 or (len(input_copy.shape) == 3 and input_copy.shape[2] in [1, 3, 4]):
            input_copy = np.expand_dims(input_copy, axis=0)

        # Ensure we have at least one image in the batch
        if input_copy.shape[0] == 0:
            raise ValueError("Empty batch provided")

        # Process each image in the batch using our standardized preprocessing
        processed_batch = []
        for i in range(input_copy.shape[0]):
            # Get one image and ensure it's in RGB format (ResNet needs RGB)
            img, _ = preprocess_image(input_copy[i], normalize=True, ensure_rgb=True)
            processed_batch.append(img)

        # Stack back into a batch
        processed_batch = np.stack(processed_batch)

        # Process each image in the batch
        results = []

        for img in tqdm(processed_batch, desc="Analyzing images", leave=False):
            # Compare with the dataset - use the correct path to Not Raphael folder
            predictions = compare_image_with_dataset(img, 'data/Not Raphael/')
            results.append(predictions)

        # Convert results to numpy array
        results = np.array(results)

        # Ensure the output is 2D with shape [batch_size, num_classes]
        if len(results.shape) == 1:
            results = results.reshape(1, -1)

        return results



cache = Cache('my_cache_directory')

def scale_inverse_log(x, x_min, x_max, y_min, y_max):
    # Check input boundaries
    if x < x_min or x > x_max:
        return "Input x must be within the range [x_min, x_max]"

    # Calculate inverse log of x
    inv_log_x = -1 / math.log(x + 1)

    # Calculate inverse log of x_min and x_max
    inv_log_x_min = -1 / math.log(x_min + 1)
    inv_log_x_max = -1 / math.log(x_max + 1)

    # Scale the inverse logarithmic value to the target range [y_min, y_max]
    y = y_min + (inv_log_x - inv_log_x_min) * (y_max - y_min) / (inv_log_x_max - inv_log_x_min)

    return y


def preprocess_image(img, normalize=True, ensure_rgb=False):
    """
    Standardized image preprocessing function.

    Args:
        img: Input image in various formats
        normalize: Whether to normalize to [0,1] range
        ensure_rgb: Whether to convert grayscale to RGB

    Returns:
        Processed image in the desired format
    """
    # Handle both file paths and numpy arrays
    if isinstance(img, (str, Path)):
        img = io.imread(str(img))

    # Handle batched images - take the first one if single image needed
    if len(img.shape) == 4:
        # For feature calculation, use single image
        single_img = img[0]
    else:
        single_img = img

    # Convert to grayscale if needed for edge detection
    if len(single_img.shape) == 3 and single_img.shape[2] > 1:
        gray = color.rgb2gray(single_img)
    else:
        # Handle grayscale with extra dimensions or already 2D
        gray = np.squeeze(single_img)

    # Ensure we have RGB if requested (for ResNet)
    if ensure_rgb:
        if len(single_img.shape) == 2:
            # Add channel dimension if missing
            single_img = np.expand_dims(single_img, axis=-1)

        if single_img.shape[-1] == 1:
            # Convert single channel to RGB
            single_img = np.repeat(single_img, 3, axis=-1)

    # Normalize if requested
    if normalize and single_img.max() > 1.0:
        single_img = single_img / 255.0

    return single_img, gray

def extract_features(img_path, model):
    """
    Extract features from an image using the provided model.

    Args:
        img_path: Path to an image or an image array
        model: The model to use for feature extraction

    Returns:
        Feature vector extracted from the image
    """
    # Get the processed image
    img, _ = preprocess_image(img_path, normalize=False, ensure_rgb=True)

    # Add batch dimension if missing
    if len(img.shape) == 3:
        img = np.expand_dims(img, axis=0)

    # Prepare for ResNet50
    if img.dtype == np.uint8:
        # Already in [0,255] range, no change needed
        pass
    elif img.max() <= 1.0:
        # Convert from [0,1] to [0,255] for preprocessing
        img = (img * 255).astype(np.uint8)

    # Use the model's preprocessing if available
    if model is not None:
        try:
            img = keras.applications.resnet50.preprocess_input(img)
            features = model.predict(img, verbose=0)
            return features
        except Exception as e:
            print(f"Feature extraction error: {str(e)}")
            return None
    return None

# Function to calculate edge features using Canny edge detector
def calculate_canny_edges(img):
    _, gray = preprocess_image(img)
    edges = feature.canny(gray, sigma=1.0)

    # Once the edge features are computed, the standard deviation is calculated for
    # every individual edge feature obtained from an image. The standard deviation serves as
    # an effective metric to quantify the variability and intensity of edge features in the image.
    return np.std(edges)


# Function to calculate edge features using Sobel operator
def calculate_sobel_edges(img):
    _, gray = preprocess_image(img)
    sobelx = filters.sobel_h(gray)
    sobely = filters.sobel_v(gray)
    return np.std(sobelx), np.std(sobely)


# Function to calculate edge features using Laplacian operator
def calculate_laplacian_edges(img):
    _, gray = preprocess_image(img)
    laplacian = filters.laplace(gray)
    return np.std(laplacian)


# Function to calculate edge features using Scharr operator
def calculate_scharr_edges(img):
    _, gray = preprocess_image(img)
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


def compare_image_with_dataset(test_image, image_dir):
    """
    Compare an image with a dataset of reference images to determine if it's a Raphael.

    Args:
        test_image: The image to test
        image_dir: Directory containing reference (non-Raphael) images

    Returns:
        List of probabilities [Raphael, Non-Raphael]
    """

    resnet_path = Path("models/resnet50_model.h5")
    svm_path = Path("models/28_09_2023_svm_final_model.pkl")

    # Load the final model
    svm_final = joblib.load(svm_path)

    # Load the saved model
    resnet_model = keras.models.load_model(resnet_path, compile=False)

    # Extract features from the test image
    test_image_features = extract_features(test_image, resnet_model)

    # Flatten features if needed
    if len(test_image_features.shape) > 1:
        test_image_features =  test_image_features.reshape(-1)

    # Use the loaded model to predict the category of the test image
    predicted_category = svm_final.predict([test_image_features])[0]

    # Calculate probabilities for each category
    probabilities = svm_final.predict_proba([test_image_features])[0]

    categories = ['Raphael', 'Not Raphael']

    # Calculate features of test image
    test_features = calculate_features(test_image)

    # Normalize test features to get weights
    weights = test_features / np.sum(test_features)

    # Load all images in directory
    formats = ('*.jpg', '*.png', '*.bmp')  # Add or remove formats as needed

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

    # Apply adjustment algorithm
    if mean_diff < 99:
        mean_diff = 400
        probabilities[0] -= 0.5

    if mean_diff > 400:
        mean_diff = 400
        probabilities[0] -= 0.5

    if mean_diff < 150:
        mean_diff = 150

    scale_ = scale_inverse_log(mean_diff, x_min=150, x_max=400, y_min=0.0, y_max=-0.99)

    threshold = 0.95 * probabilities[0] + 0.05 * scale_
    if threshold < 0:
        threshold = 0.05

    final_probabilities = [threshold, 1 - threshold]

    # Display prediction
    raphael_pct = final_probabilities[0] * 100
    non_raphael_pct = final_probabilities[1] * 100
    print(f"Prediction: Raphael: {raphael_pct:.1f}%, Non-Raphael: {non_raphael_pct:.1f}%")

    return final_probabilities


@cache.memoize()
def load_image_and_calculate_features(image_path):
    image = io.imread(image_path)
    # Calculate features of image
    image_features = calculate_features(image)
    return image_features