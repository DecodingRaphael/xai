from pathlib import Path
import joblib
import tensorflow as tf
# Fix the keras import
try:
    # Try the newer import pattern
    from tensorflow import keras
except ImportError:
    # Fall back to direct keras import
    import keras
import numpy as np
import glob
import pandas as pd
import math
from diskcache import Cache
from skimage import io, color, feature, filters

class Model:
    def __init__(self):
        self.resnet_model = None
        try:
            # Load the ResNet model if it exists
            self.resnet_model = keras.models.load_model("models/resnet50_model.h5")
        except:
            # Model will rely on image features if ResNet is not available
            pass

    def extract_features(self, img):
        if self.resnet_model is not None:
            # Preprocess input for ResNet model
            img_preprocessed = keras.applications.resnet50.preprocess_input(img)
            return self.resnet_model.predict(img_preprocessed)
        return None

    def run_on_batch(self, x):
        # Ensure x is properly formatted (handle different input shapes)
        if isinstance(x, (str, Path)):
            # If x is a file path, load the image
            x = io.imread(str(x))
        
        print(f"Input shape to run_on_batch: {x.shape}")
        
        # If input has 5 dimensions (from masks), reshape it
        if len(x.shape) == 5:
            print("Handling 5D input")
            # Reshape to 4D by combining batch dimensions
            x = x.reshape(-1, *x.shape[2:])
        
        # Handle DIANNA format [batch, channels, height, width]
        if len(x.shape) == 4:
            print("Handling 4D input")
            if x.shape[1] <= 4:  # channels in second dimension
                print("Transposing from DIANNA format")
                x = np.transpose(x, (0, 2, 3, 1))
        
        # Ensure we have a batch dimension
        if len(x.shape) == 3:
            print("Adding batch dimension")
            x = np.expand_dims(x, axis=0)
        
        # If we have a single channel, convert to RGB
        if x.shape[-1] == 1:
            print("Converting single channel to RGB")
            x = np.repeat(x, 3, axis=-1)
        
        print(f"Final shape before prediction: {x.shape}")
        
        # Get raw predictions
        predictions = compare_image_with_dataset(x, '../data/Not Rapheal/')
        print(f"Raw predictions: {predictions}")
        
        # Ensure predictions are in the format DIANNA expects: [batch_size, num_classes]
        if len(predictions) == 2 and not isinstance(predictions[0], (list, np.ndarray)):
            # If we have a single prediction, reshape it to [1, num_classes]
            predictions = np.array(predictions).reshape(1, -1)
        
        print(f"Formatted predictions shape: {predictions.shape}")
        return predictions


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
            features = model.predict(img)
            return features
        except:
            # If model prediction fails, return None
            return None
    return None


# Function to calculate edge features using Canny edge detector
def calculate_canny_edges(img):
    # Convert to grayscale if the image is in color
    if len(img.shape) > 2 and img.shape[2] > 1:
        gray = color.rgb2gray(img)
    else:
        gray = img
    
    # Apply Canny edge detection using scikit-image
    edges = feature.canny(gray, sigma=1.0)
    
    # Return standard deviation of edge image
    return np.std(edges)


# Function to calculate edge features using Sobel operator
def calculate_sobel_edges(img):
    # Convert to grayscale if the image is in color
    if len(img.shape) > 2 and img.shape[2] > 1:
        gray = color.rgb2gray(img)
    else:
        gray = img
        
    # Apply Sobel filter using scikit-image
    sobelx = filters.sobel_h(gray)
    sobely = filters.sobel_v(gray)
    
    return np.std(sobelx), np.std(sobely)


# Function to calculate edge features using Laplacian operator
def calculate_laplacian_edges(img):
    # Convert to grayscale if the image is in color
    if len(img.shape) > 2 and img.shape[2] > 1:
        gray = color.rgb2gray(img)
    else:
        gray = img
        
    # Apply Laplacian filter using scikit-image
    laplacian = filters.laplace(gray)
    
    return np.std(laplacian)


# Function to calculate edge features using Scharr operator
def calculate_scharr_edges(img):
    # Convert to grayscale if the image is in color
    if len(img.shape) > 2 and img.shape[2] > 1:
        gray = color.rgb2gray(img)
    else:
        gray = img
        
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

    # Use the provided image array directly
    test_image = test_image_path  # This is already a numpy array
    
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
    except:
        # Handle the case where the model file is not found
        print(f"Warning: SVM model not found at {Model_Path}")
        # Return default probabilities
        return [0.5, 0.5]

    # Load the ResNet model
    try:
        try:
            # Try the newer import pattern
            model = tf.keras.models.load_model(ResNet_Path)
        except:
            # Fall back to direct keras import
            model = keras.models.load_model(ResNet_Path)
    except:
        # Handle the case where the model file is not found
        print(f"Warning: ResNet model not found at {ResNet_Path}")
        model = None

    # Extract features from the test image
    test_image_features = extract_features(test_image, model)
    
    # If feature extraction failed, return default probabilities
    if test_image_features is None:
        return [0.5, 0.5]

    # Reshape features if needed
    if len(test_image_features.shape) > 1:
        test_image_features = test_image_features.reshape(-1)

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
    formats = ('*.jpg', '*.png', '*.bmp')

    image_paths = []

    for fmt in formats:
        image_paths.extend(glob.glob(f"{image_dir}/{fmt}"))

    # Calculate the total feature values and the count of images
    total_features = np.zeros_like(test_features)
    image_count = 0

    for image_path in image_paths:
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

    # adjusted values based on update in original code
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

    print(test_image_path)
    print(pd.DataFrame([['probabilities'] + list(probabilities), ['final'] + list(final_probabilities)],
                       columns=['type'] + categories))
    return final_probabilities


@cache.memoize()
def load_image_and_calculate_features(image_path):    
    image = io.imread(image_path)    
    # Calculate features of image
    image_features = calculate_features(image)
    return image_features