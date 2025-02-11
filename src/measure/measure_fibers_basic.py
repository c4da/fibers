import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage import io, filters, morphology, measure, exposure
from scipy import ndimage

import common.tools as common
import common.preprocessing as pre

def filter_small_widths(dist_transform: cv2.UMat, min_width=2) -> cv2.UMat:
    # Calculate fiber widths (diameter = 2 * distance from edge)
    fiber_widths = dist_transform[dist_transform > min_width] * 2  # Filter out small values
    if len(fiber_widths) == 0:
        return None
    return fiber_widths

def process_image_basic(image_path):
    """
    Reads the image, applies a basic threshold, morphological cleanup, 
    computes distance transform, and returns average fiber width.
    """
    file_name = image_path.name
    
    img = pre.run(image_path)
    
    # Apply a basic Otsu threshold
    # _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive_block_size = 35
    adaptive_offset=0.02
    binary = cv2.adaptiveThreshold(np.uint8(img*255), 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                         cv2.THRESH_BINARY, adaptive_block_size, adaptive_offset*255) 
    
    # Morphological operations to clean up the binary image
    kernel = np.ones((3,3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Remove small objects (noise) from the binary image
    min_fiber_length_pixels = 5000
    binary = morphology.remove_small_objects(binary, min_size=min_fiber_length_pixels)

    # 3. Skeletonization
    # skeleton = morphology.skeletonize(binary_image_cleaned)
    # 4. Remove spurious pixels
    # cleaned_skeleton = morphology.binary_erosion(skeleton, selem=np.ones((3, 3)))
    # cleaned_skeleton = morphology.skeletonize(cleaned_skeleton)
    
    # Save binary image for verification
    common.save_binary_image(binary, file_name)
    
    # Get distance transform
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

    fiber_widths = filter_small_widths(dist_transform)
    
    # Return average fiber width in pixels
    return np.mean(fiber_widths)

def main():
    # Example usage: process all .tif files in a 'resources' folder
    parent_path = Path(__file__).resolve().parent.parent.parent
    print(f"Parent path: {parent_path}")
    resources_path = Path(parent_path/"resources/drive-download-20250122T154226Z-001")
    print(f"Resources path: {resources_path}")
    image_files = list(resources_path.glob("*.tif"))
    
    if not image_files:
        print("No .tif files found!")
        return
    
    widths = []
    for img_path in image_files:
        print(f"Processing {img_path.name}...")
        width = process_image_basic(img_path)
        if width is not None:
            widths.append(width)
            print(f"Average width in pixels: {width:.2f}")
    
    if widths:
        overall_avg = np.mean(widths)
        print(f"\nOverall average width across all images: {overall_avg:.2f} pixels")
        
        common.plot_widths(widths)
    else:
        print("No valid measurements obtained with basic filtering!")

if __name__ == "__main__":
    main()