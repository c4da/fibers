import cv2
import numpy as np
from skimage import morphology, filters, feature
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.filters import frangi, hessian
from skimage.feature import hessian_matrix, hessian_matrix_eigvals

def frangi_filter(image, scale_range=(1, 10), scale_step=2):
    """Apply Frangi filter to enhance fiber-like structures"""
    return frangi(image, 
                 scale_range=scale_range,
                 scale_step=scale_step,
                 black_ridges=False)

def trace_fiber(binary, start_point, visited):
    """Trace a single fiber using ridge following"""
    current = start_point
    fiber_points = [current]
    directions = [(0,1), (1,1), (1,0), (1,-1), (0,-1), (-1,-1), (-1,0), (-1,1)]
    
    while True:
        visited[current[0], current[1]] = True
        next_point = None
        max_val = 0
        
        # Look in all 8 directions
        for dy, dx in directions:
            y, x = current[0] + dy, current[1] + dx
            if (0 <= y < binary.shape[0] and 
                0 <= x < binary.shape[1] and 
                binary[y, x] > 0 and 
                not visited[y, x]):
                if binary[y, x] > max_val:
                    max_val = binary[y, x]
                    next_point = (y, x)
        
        if next_point is None:
            break
            
        fiber_points.append(next_point)
        current = next_point
        
    return fiber_points

def count_crossings(ridge_img, fiber_points, radius=5):
    """Count the number of fiber crossings along a fiber path"""
    crossings = 0
    checked_points = set()
    
    for y, x in fiber_points:
        # Create a circular mask around the current point
        y_min, y_max = max(0, y-radius), min(ridge_img.shape[0], y+radius+1)
        x_min, x_max = max(0, x-radius), min(ridge_img.shape[1], x+radius+1)
        
        # Count ridge pixels in the neighborhood
        for ny in range(y_min, y_max):
            for nx in range(x_min, x_max):
                if (ny, nx) not in checked_points and ridge_img[ny, nx] > 0:
                    # Don't count points that are part of the current fiber
                    if (ny, nx) not in fiber_points:
                        crossings += 1
                checked_points.add((ny, nx))
    
    return crossings

def get_fiber_width(dist_transform, points, continuity_threshold=0.7):
    """Calculate fiber width along its length with continuity check"""
    widths = []
    prev_width = None
    continuous_segments = 0
    total_segments = 0
    
    for y, x in points:
        width = dist_transform[y, x] * 2
        
        if prev_width is not None:
            # Check if current width is similar to previous width
            if abs(width - prev_width) / prev_width < 0.3:  # Allow 30% variation
                continuous_segments += 1
            total_segments += 1
        
        widths.append(width)
        prev_width = width
    
    # Calculate continuity ratio
    continuity_ratio = continuous_segments / total_segments if total_segments > 0 else 0
    
    # Only return average width if continuity criteria is met
    if continuity_ratio >= continuity_threshold:
        return np.mean(widths)
    return 0

def process_image(image_path):
    file_name = image_path.name
    
    # Read the image
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return None
    
    # Crop out the header (bottom), and axes (sides)
    height, width = img.shape
    crop_bottom = int(height * 0.85)  # Remove bottom 15%
    crop_left = int(width * 0.1)      # Remove left 10%
    crop_right = int(width * 0.9)     # Remove right 10%
    
    img = img[:crop_bottom, crop_left:crop_right]
    
    # Enhance fiber-like structures using Frangi filter
    fiber_enhanced = frangi_filter(img)
    
    # Normalize to 0-255 range
    fiber_enhanced = ((fiber_enhanced - fiber_enhanced.min()) * 
                     (255.0 / (fiber_enhanced.max() - fiber_enhanced.min())))
    fiber_enhanced = fiber_enhanced.astype(np.uint8)
    
    # Threshold the enhanced image
    _, binary = cv2.threshold(fiber_enhanced, 0, 255, 
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Clean up using morphological operations
    kernel = np.ones((3,3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Save binary image for verification
    plt.figure(figsize=(10, 6))
    plt.imshow(binary, cmap='gray')
    plt.title('Binary - ' + file_name)
    plt.savefig(file_name + 'binary.png')
    plt.close()
    
    # Get distance transform for width measurement
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    
    # Find fiber centerlines using Hessian matrix eigenvalues
    Hxx, Hxy, Hyy = hessian_matrix(dist_transform, sigma=1)
    H = np.array([Hxx, Hxy, Hyy])  # Combine Hessian components
    eigvals = hessian_matrix_eigvals(H)
    
    # Threshold the eigenvalues to get ridges
    ridge_img = np.zeros_like(binary)
    ridge_img[eigvals[1] < -0.001] = 255  # Second eigenvalue indicates ridges
    
    # Save ridge image and debug info
    plt.figure(figsize=(10, 6))
    plt.imshow(ridge_img, cmap='gray')
    plt.title('Ridges - ' + file_name)
    plt.savefig(file_name + 'skeleton.png')
    plt.close()
    
    # Create a colored visualization image
    vis_img = cv2.cvtColor(binary.copy(), cv2.COLOR_GRAY2BGR)
    
    # Trace fibers and measure widths
    visited = np.zeros_like(binary, dtype=bool)
    fiber_data = []  # Store (width, crossings, points) tuples
    min_fiber_length = 20  # Minimum length to consider a valid fiber
    max_crossings = 10    # Maximum number of crossings to consider a fiber in foreground
    
    total_fibers = 0
    long_enough_fibers = 0
    few_crossing_fibers = 0
    continuous_fibers = 0
    
    # Find starting points (unvisited ridge pixels)
    ridge_points = np.where(ridge_img > 0)
    for y, x in zip(*ridge_points):
        if not visited[y, x]:
            total_fibers += 1
            fiber_points = trace_fiber(ridge_img, (y, x), visited)
            
            if len(fiber_points) >= min_fiber_length:  # Filter short segments
                long_enough_fibers += 1
                # Convert fiber_points list to set for faster lookup
                fiber_points_set = set((y, x) for y, x in fiber_points)
                crossings = count_crossings(ridge_img, fiber_points_set)
                
                if crossings <= max_crossings:  # Only consider fibers with few crossings
                    few_crossing_fibers += 1
                    width = get_fiber_width(dist_transform, fiber_points)
                    if width > 0:
                        continuous_fibers += 1
                        fiber_data.append((width, crossings, fiber_points))
                        
                        # Draw the fiber on visualization image
                        color = (0, 255, 0) if crossings <= 3 else (0, 255, 255)  # Green for very clear fibers, yellow for others
                        for py, px in fiber_points:
                            vis_img[py, px] = color
    
    # Print debug information
    print(f"Total fibers found: {total_fibers}")
    print(f"Fibers long enough: {long_enough_fibers}")
    print(f"Fibers with few crossings: {few_crossing_fibers}")
    print(f"Continuous fibers: {continuous_fibers}")
    
    if not fiber_data:
        return None
    
    # Sort fibers by number of crossings (prioritize fibers with fewer crossings)
    fiber_data.sort(key=lambda x: x[1])  # Sort by number of crossings
    
    # Save visualization image
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
    plt.title('Analyzed Fibers - ' + file_name)
    plt.savefig(file_name + 'analyzed.png')
    plt.close()
    
    # Take only the top 50% of fibers with fewest crossings
    n_fibers = len(fiber_data)
    selected_fibers = fiber_data[:n_fibers//2]
    
    # Extract widths from selected fibers
    widths = [width for width, _, _ in selected_fibers]
    
    if not widths:
        return None
    
    # Filter out outliers using IQR method
    widths = np.array(widths)
    Q1 = np.percentile(widths, 25)
    Q3 = np.percentile(widths, 75)
    IQR = Q3 - Q1
    valid_widths = widths[
        (widths >= Q1 - 1.5 * IQR) & 
        (widths <= Q3 + 1.5 * IQR)
    ]
    
    if len(valid_widths) == 0:
        return None
        
    # Return average width in pixels
    return np.mean(valid_widths)

def main():
    # Path to resources folder
    resources_path = Path("resources/drive-download-20250122T154226Z-001")
    
    # Get all .tif files
    image_files = list(resources_path.glob("*.tif"))
    
    if not image_files:
        print("No .tif files found!")
        return
    
    widths = []
    
    # Process each image
    for img_path in image_files:
        print(f"Processing {img_path.name}...")
        width = process_image(img_path)
        if width is not None:
            widths.append(width)
            print(f"Average width in pixels: {width:.2f}")
    
    if widths:
        overall_avg = np.mean(widths)
        print(f"\nOverall average width across all images: {overall_avg:.2f} pixels")
        
        # Create a simple plot
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(widths)), widths)
        plt.title('Fiber Widths Across Images')
        plt.xlabel('Image Number')
        plt.ylabel('Average Width (pixels)')
        plt.savefig('fiber_widths_plot.png')
        plt.close()
    else:
        print("No valid measurements obtained!")

if __name__ == "__main__":
    main()
