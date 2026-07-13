import cv2
import numpy as np
import tensorflow as tf
from scipy.stats import entropy


def add_salt_and_pepper(image, prob=0.05):
    # Generate random noise mask
    rand = tf.random.uniform(shape=tf.shape(image), minval=0, maxval=1)
    
    # Salt (1.0)
    salt_mask = rand < (prob / 2)
    # Pepper (0.0)
    pepper_mask = rand > (1 - prob / 2)
    
    # Apply noise
    image = tf.where(salt_mask, 1.0, image)
    image = tf.where(pepper_mask, 0.0, image)

    return image

def apply_watershed(image_array):
    img_uint8 = (image_array * 255).astype(np.uint8)
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Return as float tensor for consistency
    return thresh / 255.0

def extract_texture_features(image_np):
    """
    Extracts explicit, human-readable surface texture features (variance and entropy)
    using Gabor filters to evaluate surface homogeneity and defects.
    """
    # Fix: If the incoming data is a TensorFlow Tensor, convert it to a NumPy array first
    if hasattr(image_np, "numpy"):
        image_np = image_np.numpy()
    elif isinstance(image_np, tf.Tensor):
        image_np = image_np.numpy()

    # 1. Scale the input image to 0-255 uint8 format safely
    img_uint8 = (image_np * 255).astype(np.uint8)
    
    # 2. Check channels and convert to grayscale if necessary
    if len(img_uint8.shape) == 2 or img_uint8.shape[-1] == 1:
        gray = img_uint8
    else:
        gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
        
    # 3. Define Gabor filter parameters
    ksize = 31
    sigma = 4.0
    theta = np.pi / 4  
    lambd = 10.0
    gamma = 0.5
    psi = 0
    
    # 4. Build and apply the Gabor filter
    gabor_kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
    filtered_img = cv2.filter2D(gray, cv2.CV_32F, gabor_kernel)
    
    # 5. Calculate Texture Variance
    texture_variance = np.var(filtered_img)
    
    # 6. Calculate Entropy 
    hist, _ = np.histogram(filtered_img.flatten(), bins=256, range=(filtered_img.min(), filtered_img.max()))
    prob_dist = hist / np.sum(hist)
    prob_dist = prob_dist[prob_dist > 0]
    texture_entropy = -np.sum(prob_dist * np.log2(prob_dist))
    
    # 7. Return features as a flat array ready for concatenation
    return np.array([texture_variance, texture_entropy], dtype=np.float32)