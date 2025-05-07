import cv2
import numpy as np
import pickle
import argparse
import matplotlib.pyplot as plt
import os

def compute_disparity_map(left_img, right_img, calibration_data, max_disparity=128, block_size=5):
    """
    Compute disparity map from stereo images.
    
    Args:
        left_img: Left stereo image (BGR or grayscale)
        right_img: Right stereo image (BGR or grayscale)
        calibration_data: Dictionary with stereo calibration parameters
        max_disparity: Maximum disparity value
        block_size: Block size for stereo matching
        
    Returns:
        Disparity map and normalized disparity for visualization
    """
    # Convert to grayscale if needed
    if len(left_img.shape) == 3:
        left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    else:
        left_gray = left_img
        right_gray = right_img
    
    # Get calibration parameters
    stereo_params = calibration_data['stereo']
    
    # Create rectification maps
    left_map1, left_map2 = cv2.initUndistortRectifyMap(
        stereo_params['left_camera_matrix'],
        stereo_params['left_dist_coeffs'],
        stereo_params['R1'],
        stereo_params['P1'],
        stereo_params['image_size'],
        cv2.CV_16SC2)
    
    right_map1, right_map2 = cv2.initUndistortRectifyMap(
        stereo_params['right_camera_matrix'],
        stereo_params['right_dist_coeffs'],
        stereo_params['R2'],
        stereo_params['P2'],
        stereo_params['image_size'],
        cv2.CV_16SC2)
    
    # Apply rectification
    left_rectified = cv2.remap(left_gray, left_map1, left_map2, cv2.INTER_LINEAR)
    right_rectified = cv2.remap(right_gray, right_map1, right_map2, cv2.INTER_LINEAR)
    
    # Optional: Apply preprocessing to improve matching
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    left_enhanced = clahe.apply(left_rectified)
    right_enhanced = clahe.apply(right_rectified)
    
    # Create stereo matcher
    # Semi-Global Block Matching
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=max_disparity,
        blockSize=block_size,
        P1=8 * 3 * block_size**2,
        P2=32 * 3 * block_size**2,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=100,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    
    # Compute disparity
    disparity = stereo.compute(left_enhanced, right_enhanced)
    
    # Convert to float and scale
    disparity_float = disparity.astype(np.float32) / 16.0
    
    # Apply post-processing to remove noise
    # Median filter
    disparity_filtered = cv2.medianBlur(disparity_float.astype(np.float32), 5)
    
    # Bilateral filter for edge-preserving smoothing
    disparity_filtered = cv2.bilateralFilter(disparity_filtered, 9, 75, 75)
    
    # Create validity mask (exclude regions with invalid disparity)
    min_disparity = 1.0  # Minimum valid disparity value
    valid_mask = (disparity_filtered > min_disparity)
    
    # Apply mask
    disparity_filtered[~valid_mask] = 0
    
    # Normalize for visualization
    norm_disparity = cv2.normalize(disparity_filtered, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return disparity_filtered, norm_disparity, left_rectified, right_rectified

def disparity_to_depth(disparity, calibration_data):
    """
    Convert disparity map to depth map using stereo parameters.
    
    Args:
        disparity: Disparity map (float32)
        calibration_data: Dictionary with stereo calibration parameters
        
    Returns:
        Depth map in meters
    """
    # Get stereo parameters
    stereo_params = calibration_data['stereo']
    baseline = stereo_params['baseline'] / 1000.0  # Convert mm to meters
    focal_length = stereo_params['left_camera_matrix'][0, 0]  # Focal length in pixels
    
    # Create valid mask (exclude regions with invalid disparity)
    valid_mask = (disparity > 0)
    
    # Initialize depth map
    depth = np.zeros_like(disparity, dtype=np.float32)
    
    # Calculate depth: depth = (baseline * focal_length) / disparity
    depth[valid_mask] = (baseline * focal_length) / disparity[valid_mask]
    
    # Set invalid regions to zero or a large value
    depth[~valid_mask] = 0
    
    # Filter out unrealistic values (too close or too far)
    min_depth = 0.1  # 10 cm
    max_depth = 40.0  # 40 meters
    
    depth[depth < min_depth] = 0
    depth[depth > max_depth] = max_depth
    
    return depth

def filter_depth_map(depth_map, kernel_size=5, fill_holes=True):
    """
    Apply filters to improve depth map quality.
    
    Args:
        depth_map: Input depth map
        kernel_size: Size of filter kernels
        fill_holes: Whether to fill holes in the depth map
    
    Returns:
        Filtered depth map
    """
    # Create a copy of the depth map
    filtered_depth = depth_map.copy()
    
    # Create a mask for valid depth values
    valid_mask = (filtered_depth > 0)
    
    # Apply median filter to remove salt-and-pepper noise
    # Only apply to valid regions
    valid_depth = filtered_depth.copy()
    valid_depth[~valid_mask] = 0
    median_filtered = cv2.medianBlur(valid_depth, kernel_size)
    
    # Restore zeros where the original was zero
    median_filtered[~valid_mask] = 0
    
    # Apply bilateral filter for edge-preserving smoothing
    # First normalize to 0-1 range for better filter performance
    max_depth = np.max(median_filtered)
    if max_depth > 0:
        normalized = median_filtered / max_depth
        bilateral_filtered = cv2.bilateralFilter(normalized, kernel_size, 0.05, 5.0)
        bilateral_filtered = bilateral_filtered * max_depth
    else:
        bilateral_filtered = median_filtered
    
    # Fill holes if requested
    if fill_holes:
        # Only fill small holes (isolated zero regions surrounded by valid depth)
        mask = (bilateral_filtered == 0).astype(np.uint8)
        
        # Identify small holes using morphological operations
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        small_holes = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Inpaint the depth map only at small hole locations
        hole_mask = (small_holes > 0).astype(np.uint8) * 255
        if np.any(hole_mask):
            inpainted = cv2.inpaint(
                bilateral_filtered.astype(np.float32), 
                hole_mask, 
                kernel_size, 
                cv2.INPAINT_NS
            )
        else:
            inpainted = bilateral_filtered
    else:
        inpainted = bilateral_filtered
    
    return inpainted

def main():
    parser = argparse.ArgumentParser(description='Stereo Depth Estimation')
    
    # Input arguments
    parser.add_argument('--left_img', required=True, help='Path to left image')
    parser.add_argument('--right_img', required=True, help='Path to right image')
    parser.add_argument('--calib_file', required=True, help='Path to calibration file')
    parser.add_argument('--output_dir', default='./depth_results', help='Output directory')
    parser.add_argument('--max_disparity', type=int, default=128, help='Maximum disparity value')
    parser.add_argument('--block_size', type=int, default=5, help='Block size for stereo matching')
    parser.add_argument('--filter', action='store_true', help='Apply additional depth filtering')
    parser.add_argument('--visualize', action='store_true', help='Visualize results')
    
    args = parser.parse_args()
    
    # Load calibration data
    with open(args.calib_file, 'rb') as f:
        calibration_data = pickle.load(f)
    
    # Load images
    left_img = cv2.imread(args.left_img)
    right_img = cv2.imread(args.right_img)
    
    if left_img is None or right_img is None:
        print(f"Error: Could not read input images")
        return
    
    # Compute disparity map
    print("Computing disparity map...")
    disparity, norm_disparity, left_rect, right_rect = compute_disparity_map(
        left_img, right_img, calibration_data, args.max_disparity, args.block_size)
    
    # Convert disparity to depth
    print("Converting disparity to depth...")
    depth_map = disparity_to_depth(disparity, calibration_data)
    
    # Apply additional filtering if requested
    if args.filter:
        print("Applying depth filtering...")
        depth_map = filter_depth_map(depth_map)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save rectified images
    cv2.imwrite(os.path.join(args.output_dir, 'left_rectified.png'), left_rect)
    cv2.imwrite(os.path.join(args.output_dir, 'right_rectified.png'), right_rect)
    
    # Save disparity map
    cv2.imwrite(os.path.join(args.output_dir, 'disparity_map.png'), norm_disparity)
    
    # Save depth map as numpy array
    np.save(os.path.join(args.output_dir, 'depth_map.npy'), depth_map)
    
    # Save depth map visualization
    # Normalize for visualization (0-255)
    depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(os.path.join(args.output_dir, 'depth_map.png'), depth_normalized)
    
    # Save depth colormap
    depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(args.output_dir, 'depth_map_color.png'), depth_colormap)
    
    print(f"Results saved to {args.output_dir}")
    
    # Visualize results
    if args.visualize:
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        plt.title('Left Image')
        plt.imshow(cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.subplot(2, 2, 2)
        plt.title('Right Image')
        plt.imshow(cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.subplot(2, 2, 3)
        plt.title('Disparity Map')
        plt.imshow(norm_disparity, cmap='gray')
        plt.axis('off')
        
        plt.subplot(2, 2, 4)
        plt.title('Depth Map')
        plt.imshow(cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'visualization.png'))
        plt.show()

if __name__ == "__main__":
    main()