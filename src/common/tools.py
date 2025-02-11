import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage import io, filters, morphology, measure, exposure, segmentation, feature
from scipy import ndimage
import matplotlib.pyplot as plt


def crop_image(img: cv2.UMat) -> cv2.UMat:
    # Crop out the header (bottom), and axes (sides)
    height, width = img.shape
    crop_bottom = int(height * 0.85)  # Remove bottom 15%
    crop_left = int(width * 0.1)      # Remove left 10%
    crop_right = int(width * 0.9)     # Remove right 10%
    return img[:crop_bottom, crop_left:crop_right]

def read_image(image_path) -> cv2.UMat:
    # Read image in grayscale
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return None
    return img

def save_binary_image(binary: cv2.UMat, file_name):
    # Save binary image for verification
    plt.figure(figsize=(10, 6))
    plt.imshow(binary, cmap='gray')
    plt.title('Binary - ' + file_name)
    plt.savefig(file_name + '_binary_basic.png')
    plt.close()

def plot_widths(widths):
    # Create a simple plot
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(widths)), widths)
    plt.title('Fiber Widths (Basic)')
    plt.xlabel('Image Number')
    plt.ylabel('Average Width (pixels)')
    plt.savefig('fiber_widths_plot_basic.png')
    plt.close()

def filter_noise(image: cv2.UMat, h=10, templateWindowSize=7, searchWindowSize=21):
    """
    Apply non-local means denoising to the image.
    advanced denoising method that compares small patches 
    across the entire image to find similar areas, then averages them to 
    remove noise while preserving edges and details.

    h: Filter strength (higher h → stronger denoising).
    templateWindowSize: Size of the patch used to compute weights.    
    searchWindowSize: Area around each patch that is searched for similar patches - kernel size
    """
    return cv2.fastNlMeansDenoising(image, None, h, templateWindowSize, searchWindowSize)

def contrast_enhancement(image: cv2.UMat, clip_limit=1.0, tile_grid_size=(4,4)):
    """
    Apply contrast enhancement using CLAHE (Contrast Limited Adaptive Histogram Equalization).
    clip_limit: Threshold for contrast limiting.
    tile_grid_size: Size of grid for histogram equalization.
    """
    return cv2.createCLAHE(clip_limit, tile_grid_size).apply(image)

def contrast_enhancement_adapthist(blurred_image: cv2.UMat):
    return exposure.equalize_adapthist(blurred_image, clip_limit=0.03)