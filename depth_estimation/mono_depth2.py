#!/usr/bin/env python3
"""
Monocular Depth Estimation using MiDaS

This script implements depth estimation from a single image using:
1. MiDaS deep learning model
2. Ground plane geometry
3. Hybrid approach with Kalman filtering
"""

import cv2
import torch
import numpy as np
import math
import os
import glob
import argparse
import pickle
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

def load_midas_model(model_type="DPT_Large"):
    """
    Load MiDaS depth estimation model with model selection.
    
    Args:
        model_type: Type of MiDaS model to load. Options:
                   - "MiDaS_small": Faster but less accurate
                   - "DPT_Large": Best quality, slowest
                   - "DPT_Hybrid": Balanced quality and speed
    
    Returns:
        model: Loaded MiDaS model
        transform: Appropriate transform for the model
        device: Computation device
    """
    # Check for valid model type
    valid_models = ["MiDaS_small", "DPT_Large", "DPT_Hybrid"]
    if model_type not in valid_models:
        print(f"Warning: Invalid model_type '{model_type}'. Using 'DPT_Large' instead.")
        model_type = "DPT_Large"
    
    print(f"Loading MiDaS model: {model_type}")
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Try to import MiDaS directly
        try:
            import midas.transforms
            from midas.model_loader import load_model
            print("MiDaS already installed")
            
            # Load model
            model_path = None  # Use default path
            model, transform = load_model(model_type, model_path, optimize=True)
            
        except ImportError:
            # If MiDaS isn't installed as a module, try PyTorch Hub
            print("MiDaS module not found, trying PyTorch Hub...")
            
            # For DPT models
            if model_type in ["DPT_Large", "DPT_Hybrid"]:
                model = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
                transform = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True).dpt_transform
            # For MiDaS small
            elif model_type == "MiDaS_small":
                model = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
                transform = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True).small_transform
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            print(f"Successfully loaded MiDaS {model_type} via PyTorch Hub")
    
    except Exception as e:
        print(f"Error loading MiDaS model: {e}")
        print("Using fallback placeholder implementation for testing")
        
        # Create a placeholder model for testing/debugging
        class DummyModel:
            def __init__(self):
                self.device = device
            
            def to(self, device):
                return self
                
            def eval(self):
                return self
                
            def __call__(self, x):
                # Return random tensor of appropriate shape for testing
                b, c, h, w = x.shape
                return torch.rand((b, 1, h, w), device=device)
        
        class DummyTransform:
            def __call__(self, img):
                # Assumes img is a numpy array of shape (H, W, 3)
                # Convert to tensor of shape (1, 3, H, W)
                tensor = torch.from_numpy(img.transpose((2, 0, 1))).float() / 255.0
                return tensor.unsqueeze(0)
        
        model = DummyModel()
        transform = DummyTransform()
        
        print("WARNING: Using dummy model for testing! No real depth estimation will be performed.")
    
    # Move model to device and set to evaluation mode
    model.to(device)
    model.eval()
    
    return model, transform, device

def estimate_depth(image_path, model, transform, auto_calibrate=True, camera_height=1.4, pitch_degrees=12.0, gopro_model="HERO11"):
    """
    Estimate depth with improved scaling to absolute metrics using ground plane calibration.
    
    Args:
        image_path: Path to the input image
        model: MiDaS model
        transform: MiDaS transform
        auto_calibrate: Whether to auto-calibrate the depth map using ground plane geometry
        camera_height: Height of the camera from the ground in meters
        pitch_degrees: Downward pitch of the camera in degrees
        gopro_model: GoPro camera model for FOV estimation
        
    Returns:
        img: Original image
        depth_map_meters: Depth map in meters
        reference_pixel: Reference point used for calibration (if auto_calibrate=True)
        reference_distance: Reference distance in meters (if auto_calibrate=True)
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image from {image_path}")

    # Convert to RGB for MiDaS
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Prepare input batch
    input_batch = transform(img_rgb).to(next(model.parameters()).device)

    # Generate depth prediction
    with torch.no_grad():
        prediction = model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze().cpu().numpy()

    # Ensure positive depth values
    prediction = np.maximum(prediction, 0.1)  # Minimum threshold

    if auto_calibrate:
        # Get camera FOV information
        camera_fov = get_fov_from_camera_params(gopro_model)
        vertical_fov = camera_fov["vertical"]

        # Determine reference point for ground plane (visible road start)
        image_height = img.shape[0]
        image_width = img.shape[1]

        # Use a point approximately 3/4 down the image for the visible road
        ground_ref_v = int(image_height * 0.75)
        ground_ref_u = int(image_width / 2)  # Center horizontally

        # Calculate the reference distance using ground plane geometry
        reference_distance = calculate_ground_distance(
            ground_ref_v, image_height, camera_height, pitch_degrees, vertical_fov
        )

        # Get depth at the reference point
        relative_depth_at_reference = prediction[ground_ref_v, ground_ref_u]

        # Calculate scaling factor
        if relative_depth_at_reference > 0.1:
            depth_scale = reference_distance / relative_depth_at_reference
        else:
            # Try to find a better reference point nearby
            window_size = 25
            y_min = max(0, ground_ref_v - window_size)
            y_max = min(image_height, ground_ref_v + window_size + 1)
            x_min = max(0, ground_ref_u - window_size)
            x_max = min(image_width, ground_ref_u + window_size + 1)

            window = prediction[y_min:y_max, x_min:x_max]
            max_depth_idx = np.unravel_index(window.argmax(), window.shape)

            # Convert to image coordinates
            better_v = y_min + max_depth_idx[0]
            better_u = x_min + max_depth_idx[1]
            better_depth = prediction[better_v, better_u]

            # Calculate new reference distance
            new_reference_distance = calculate_ground_distance(
                better_v, image_height, camera_height, pitch_degrees, vertical_fov
            )

            depth_scale = new_reference_distance / better_depth
            
            # Update reference point
            ground_ref_u, ground_ref_v = better_u, better_v
            reference_distance = new_reference_distance

        # Apply scaling - convert to meters
        depth_map_meters = prediction * depth_scale

        # Apply a bilateral filter to reduce noise while preserving edges
        depth_map_meters = cv2.bilateralFilter(depth_map_meters.astype(np.float32),
                                              d=7, sigmaColor=0.1, sigmaSpace=5.0)

        reference_pixel = (ground_ref_u, ground_ref_v)
        return img, depth_map_meters, reference_pixel, reference_distance
    else:
        # Return the raw depth map without calibration
        return img, prediction, None, None

def get_depth_at_pixel(depth_map, x, y):
    """Get depth value at specific pixel coordinates."""
    if 0 <= y < depth_map.shape[0] and 0 <= x < depth_map.shape[1]:
        return depth_map[y, x]
    else:
        raise ValueError(f"Pixel coordinates ({x}, {y}) are outside the image bounds: {depth_map.shape[1]}x{depth_map.shape[0]}")

def estimate_distance_from_depth(depth_value, scale=1.0):
    """Convert depth value to distance in meters."""
    return depth_value * scale

def calculate_depth_confidence(x, y, depth_map, window_size=5):
    """
    Calculate confidence in the depth estimate based on depth variation in the local neighborhood.
    Lower variation indicates higher confidence.

    Args:
        x, y: Pixel coordinates
        depth_map: Depth map
        window_size: Size of the neighborhood window to analyze

    Returns:
        confidence: 0-1 value representing confidence (1 = highest)
    """
    # Create a small window around the point
    y_min = max(0, y-window_size)
    y_max = min(depth_map.shape[0], y+window_size+1)
    x_min = max(0, x-window_size)
    x_max = min(depth_map.shape[1], x+window_size+1)

    local_region = depth_map[y_min:y_max, x_min:x_max]

    # Calculate coefficient of variation (std/mean) as a measure of consistency
    mean_depth = np.mean(local_region)
    if mean_depth > 0:
        std_depth = np.std(local_region)
        coeff_variation = std_depth / mean_depth

        # Convert to confidence (0-1 scale)
        # Lower variation = higher confidence
        confidence = max(0, min(1, 1 - coeff_variation))
    else:
        confidence = 0

    return confidence

def calculate_ground_distance(v, image_height, camera_height, pitch_deg, v_fov_deg):
    """
    Calculate the distance to a point on the ground plane using perspective geometry.

    Args:
        v: Vertical pixel coordinate (from top of image)
        image_height: Height of the image in pixels
        camera_height: Height of the camera from the ground in meters
        pitch_deg: Downward pitch of the camera in degrees
        v_fov_deg: Vertical field of view in degrees

    Returns:
        distance: Distance to the ground point in meters
    """
    # Calculate the angle for each pixel
    deg_per_pixel = v_fov_deg / image_height

    # Get center of image
    center_v = image_height / 2

    # Calculate angle from optical axis (negative for points below center)
    pixel_angle = (center_v - v) * deg_per_pixel

    # Total angle from horizontal
    total_angle_rad = math.radians(pitch_deg - pixel_angle)

    # Calculate distance using trigonometry (adjacent = opposite / tan(angle))
    if total_angle_rad > 0:  # Make sure we're looking downward
        distance = camera_height / math.tan(total_angle_rad)
        return distance
    else:
        return float('inf')  # Point is above horizon

def get_fov_from_camera_params(gopro_model):
    """Get the diagonal, horizontal, and vertical FOV for a camera model."""
    # Default FOV values for different GoPro models
    camera_fov = {
        "HERO8": {"diagonal": 80, "horizontal": 69.5, "vertical": 49.8},
        "HERO9": {"diagonal": 84, "horizontal": 73.6, "vertical": 53.4},
        "HERO10": {"diagonal": 84, "horizontal": 73.6, "vertical": 53.4},
        "HERO7": {"diagonal": 78, "horizontal": 66.9, "vertical": 45.8},
        "HERO11": {"diagonal": 80, "horizontal": 69.5, "vertical": 49.8},  # Using HERO8 values as default
        "DEFAULT": {"diagonal": 80, "horizontal": 69.5, "vertical": 49.8}
    }

    return camera_fov.get(gopro_model.upper(), camera_fov["DEFAULT"])

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
    median_filtered = cv2.medianBlur(valid_depth.astype(np.float32), kernel_size)
    
    # Restore zeros where the original was zero
    median_filtered[~valid_mask] = 0
    
    # Apply bilateral filter for edge-preserving smoothing
    # First normalize to 0-1 range for better filter performance
    max_depth = np.max(median_filtered)
    if max_depth > 0:
        normalized = median_filtered / max_depth
        bilateral_filtered = cv2.bilateralFilter(normalized.astype(np.float32), kernel_size, 0.05, 5.0)
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

def save_depth_heatmap(depth_map, output_path, reference_pixel=None, reference_distance=None):
    """
    Save a heatmap visualization of the depth map.
    
    Args:
        depth_map: Depth map array
        output_path: Path to save the visualization
        reference_pixel: Optional reference pixel (x,y) to mark
        reference_distance: Optional reference distance to display
    """
    plt.figure(figsize=(10, 6))
    plt.imshow(depth_map, cmap='plasma')
    plt.colorbar(label='Depth (meters)')
    
    if reference_pixel is not None:
        plt.scatter(reference_pixel[0], reference_pixel[1], c='white', s=50)
        if reference_distance is not None:
            plt.annotate(f"{reference_distance:.1f}m", 
                         (reference_pixel[0]+10, reference_pixel[1]-10),
                         color='white', fontsize=8)
    
    plt.title('Depth Map')
    plt.axis('off')
    plt.tight_layout()
    
    # Make sure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

# New functions for geometric and hybrid depth estimation

def estimate_depth_with_geometry(img_path, model, transform, device, mono_params=None):
    """
    Estimate depth using MiDaS with geometric ground-plane correction.
    
    Args:
        img_path: Path to input image
        model: MiDaS model
        transform: MiDaS transform
        device: Computation device (cuda or cpu)
        mono_params: Dictionary with camera parameters (height, tilt, FOV)
        
    Returns:
        depth_map: Estimated depth map in meters
        reference_point: Reference point used for calibration
        reference_distance: Reference distance in meters
    """
    # Load image
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image from {img_path}")
    
    # Convert to RGB for MiDaS
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Get image dimensions
    image_height, image_width = img.shape[:2]
    
    # Generate MiDaS depth prediction
    input_batch = transform(img_rgb).to(device)
    with torch.no_grad():
        prediction = model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze().cpu().numpy()
    
    # Ensure positive values
    midas_depth = np.maximum(prediction, 0.1)
    
    # If no calibration parameters, return raw depth map
    if mono_params is None:
        return midas_depth, None, None
    
    # Extract parameters for ground plane geometry
    camera_height = mono_params.get('camera_height', 1.4)  # meters
    tilt_angle = mono_params.get('tilt_angle', 12.0)  # degrees
    
    # Get vertical FOV (derive from horizontal if needed)
    if 'fov_vertical' in mono_params:
        vertical_fov = mono_params['fov_vertical']
    elif 'fov_horizontal' in mono_params:
        # Approximate vertical FOV from horizontal and aspect ratio
        aspect_ratio = image_height / image_width
        vertical_fov = mono_params['fov_horizontal'] * aspect_ratio
    else:
        # Default vertical FOV for GoPro HERO11
        vertical_fov = 49.8
    
    # Calculate reference point (approximately 3/4 down the image)
    ref_y = int(image_height * 0.75)
    ref_x = int(image_width / 2)
    
    # Calculate expected distance using ground plane geometry
    reference_distance = calculate_ground_distance(
        ref_y, image_height, camera_height, tilt_angle, vertical_fov
    )
    
    # Get MiDaS depth at reference point
    relative_depth_at_reference = midas_depth[ref_y, ref_x]
    
    # Calculate scaling factor
    if relative_depth_at_reference > 0.1:
        depth_scale = reference_distance / relative_depth_at_reference
    else:
        # Try to find a better reference point nearby
        window_size = 25
        y_min = max(0, ref_y - window_size)
        y_max = min(image_height, ref_y + window_size + 1)
        x_min = max(0, ref_x - window_size)
        x_max = min(image_width, ref_x + window_size + 1)

        window = midas_depth[y_min:y_max, x_min:x_max]
        max_depth_idx = np.unravel_index(window.argmax(), window.shape)

        # Convert to image coordinates
        better_y = y_min + max_depth_idx[0]
        better_x = x_min + max_depth_idx[1]
        better_depth = midas_depth[better_y, better_x]

        # Calculate new reference distance
        new_reference_distance = calculate_ground_distance(
            better_y, image_height, camera_height, tilt_angle, vertical_fov
        )

        depth_scale = new_reference_distance / better_depth
        
        # Update reference point
        ref_x, ref_y = better_x, better_y
        reference_distance = new_reference_distance

    # Apply scaling - convert to meters
    depth_map_meters = midas_depth * depth_scale
    
    # Create and return reference point
    reference_point = (ref_x, ref_y)
    
    return depth_map_meters, reference_point, reference_distance

def apply_geometric_correction(depth_map, mono_params, weighted=True):
    """
    Apply geometric correction to depth map using ground plane geometry.
    
    Args:
        depth_map: Input depth map (scaled to meters)
        mono_params: Dictionary with camera parameters
        weighted: Whether to apply weighted blending based on Y position
        
    Returns:
        corrected_depth: Geometrically corrected depth map
    """
    h, w = depth_map.shape
    camera_height = mono_params.get('camera_height', 1.4)
    tilt_angle = mono_params.get('tilt_angle', 12.0)
    
    # Get vertical FOV (derive from horizontal if needed)
    if 'fov_vertical' in mono_params:
        vertical_fov = mono_params['fov_vertical']
    elif 'fov_horizontal' in mono_params:
        aspect_ratio = h / w
        vertical_fov = mono_params['fov_horizontal'] * aspect_ratio
    else:
        vertical_fov = 49.8  # Default for GoPro HERO11
    
    # Create geometric depth map
    geometric_depth = np.zeros_like(depth_map)
    for v in range(h):
        geo_distance = calculate_ground_distance(v, h, camera_height, tilt_angle, vertical_fov)
        geometric_depth[v, :] = geo_distance
    
    # Handle invalid values
    geometric_depth[geometric_depth < 0] = 100.0
    geometric_depth[geometric_depth == float('inf')] = 100.0
    
    if weighted:
        # Create weight map based on vertical position
        # Weight transitions from MiDaS-dominant at top to geometry-dominant at bottom
        weight_map = np.zeros_like(depth_map)
        for v in range(h):
            # Normalized vertical position (0 at top, 1 at bottom)
            norm_v = v / h
            # More weight to geometry for lower pixels (likely ground)
            geometry_weight = min(1.0, norm_v * 2.0)  # Increase weight faster
            weight_map[v, :] = geometry_weight
        
        # Apply weighted blending
        corrected_depth = (1.0 - weight_map) * depth_map + weight_map * geometric_depth
    else:
        # Simple average
        corrected_depth = (depth_map + geometric_depth) / 2.0
    
    return corrected_depth

def apply_kalman_filter(original_depth, geometric_depth, process_noise=0.01, measurement_noise=0.1):
    """
    Apply a simple Kalman filter to blend the original and geometric depth maps.
    
    Args:
        original_depth: Original depth map from MiDaS
        geometric_depth: Depth map from geometric calculations
        process_noise: Process noise parameter (Q)
        measurement_noise: Measurement noise parameter (R)
        
    Returns:
        filtered_depth: Kalman-filtered depth map
    """
    # Initialize Kalman filter state with original depth
    state = original_depth.copy()
    
    # Process covariance (uncertainty in the model) - higher for less trust in model
    P = np.ones_like(original_depth) * process_noise
    
    # Measurement noise (uncertainty in measurements) - higher for less trust in measurements
    R = measurement_noise
    
    # Prediction step (in 1D Kalman, prediction is same as previous state)
    # x_pred = x_prev (state is already the prediction in this simple case)
    # P_pred = P_prev + Q
    P = P + process_noise
    
    # Calculate Kalman gain
    # K = P_pred / (P_pred + R)
    K = P / (P + R)
    
    # Update step
    # x_new = x_pred + K * (measurement - x_pred)
    # P_new = (1 - K) * P_pred
    innovation = geometric_depth - state
    state = state + K * innovation
    P = (1 - K) * P
    
    return state

def visualize_confidence_map(depth_map, confidence_map, output_path=None):
    """Visualize depth map and confidence map side by side."""
    plt.figure(figsize=(12, 5))
    
    # Plot depth map
    plt.subplot(1, 2, 1)
    plt.imshow(depth_map, cmap='plasma')
    plt.colorbar(label='Depth (meters)')
    plt.title('Hybrid Depth Map')
    plt.axis('off')
    
    # Plot confidence map
    plt.subplot(1, 2, 2)
    plt.imshow(confidence_map, cmap='viridis')
    plt.colorbar(label='Confidence (0-1)')
    plt.title('Confidence Map')
    plt.axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Monocular Depth Estimation using MiDaS')
    
    # Input arguments
    parser.add_argument('--img', required=True, help='Path to input image')
    parser.add_argument('--model_type', default='DPT_Large', 
                       choices=['DPT_Large', 'DPT_Hybrid', 'MiDaS_small'],
                       help='MiDaS model type')
    parser.add_argument('--calib_file', help='Path to monocular calibration file (for scaling)')
    parser.add_argument('--camera_height', type=float, help='Camera height in meters (overrides calibration)')
    parser.add_argument('--tilt_angle', type=float, help='Camera tilt angle in degrees (overrides calibration)')
    parser.add_argument('--output_dir', default='./depth_results', help='Output directory')
    parser.add_argument('--filter', action='store_true', help='Apply additional depth filtering')
    parser.add_argument('--visualize', action='store_true', help='Visualize results')
    parser.add_argument('--method', choices=['standard', 'geometric', 'hybrid', 'kalman'], 
                       default='standard', help='Depth estimation method')
    parser.add_argument('--process_noise', type=float, default=0.01,
                       help='Process noise parameter for Kalman filter (lower = trust model more)')
    parser.add_argument('--measurement_noise', type=float, default=0.1,
                       help='Measurement noise parameter for Kalman filter (lower = trust geometry more)')
    
    args = parser.parse_args()
    
    # Load image
    img = cv2.imread(args.img)
    
    if img is None:
        print(f"Error: Could not read input image: {args.img}")
        return
    
    # Load monocular parameters if provided
    mono_params = None
    if args.calib_file:
        try:
            with open(args.calib_file, 'rb') as f:
                calib_data = pickle.load(f)
                if 'mono' in calib_data:
                    mono_params = calib_data['mono']
                else:
                    mono_params = calib_data  # Assume it's already mono_params
                
                print(f"Loaded calibration parameters:")
                print(f"  Camera height: {mono_params['camera_height']:.2f} m")
                print(f"  Tilt angle: {mono_params['tilt_angle']:.2f} degrees")
                print(f"  Field of view: {mono_params['fov_horizontal']:.2f} degrees")
                
                # Override with command line arguments if provided
                if args.camera_height:
                    print(f"Overriding camera height with command line value: {args.camera_height:.2f} m")
                    mono_params['camera_height'] = args.camera_height
                
                if args.tilt_angle:
                    print(f"Overriding tilt angle with command line value: {args.tilt_angle:.2f} degrees")
                    mono_params['tilt_angle'] = args.tilt_angle
                
        except Exception as e:
            print(f"Warning: Failed to load calibration file: {e}")
    
    # If no calibration file but command line params provided, create mono_params
    if mono_params is None and args.camera_height and args.tilt_angle:
        mono_params = {
            'camera_height': args.camera_height,
            'tilt_angle': args.tilt_angle,
            # Approximate params for GoPro HERO11 Black
            'camera_matrix': np.array([
                [1000.0, 0.0, img.shape[1]/2],
                [0.0, 1000.0, img.shape[0]/2],
                [0.0, 0.0, 1.0]
            ]),
            'fov_horizontal': 120.0,  # Default wide FOV for GoPro
            'image_size': (img.shape[1], img.shape[0])
        }
        print(f"Using command line parameters:")
        print(f"  Camera height: {mono_params['camera_height']:.2f} m")
        print(f"  Tilt angle: {mono_params['tilt_angle']:.2f} degrees")
    
# Check if we can proceed with geometric methods
    if (args.method in ['geometric', 'hybrid', 'kalman']) and mono_params is None:
        print("Warning: Geometric methods require camera calibration parameters.")
        print("Please provide a calibration file or specify camera_height and tilt_angle.")
        print("Falling back to standard MiDaS method.")
        args.method = 'standard'
    
    # Load MiDaS model
    model, transform, device = load_midas_model(args.model_type)
    
    if args.method == 'standard':
        print("Using standard MiDaS depth estimation")
        # Get depth map using the original estimate_depth function
        img, depth_map, reference_pixel, reference_distance = estimate_depth(
            args.img, model, transform, auto_calibrate=False
        )
        
        # Apply scaling if mono_params provided
        if mono_params:
            print("Applying calibration scaling")
            # Get a reference point on the ground plane
            h, w = img.shape[:2]
            ref_y = int(h * 0.75)  # 3/4 down the image
            ref_x = int(w / 2)     # Center horizontally
            
            # Calculate expected distance using camera parameters
            if 'fov_vertical' in mono_params:
                v_fov = mono_params['fov_vertical']
            elif 'fov_horizontal' in mono_params:
                # Approximate vertical FOV from horizontal
                aspect_ratio = h / w
                v_fov = mono_params['fov_horizontal'] * aspect_ratio
            else:
                v_fov = 49.8  # Default value
            
            ref_distance = calculate_ground_distance(
                ref_y, h, mono_params['camera_height'], mono_params['tilt_angle'], v_fov
            )
            
            # Get depth at reference point
            ref_depth = depth_map[ref_y, ref_x]
            
            # Calculate scaling factor
            scale = ref_distance / ref_depth
            print(f"Applied scale factor: {scale:.2f}")
            
            # Apply scaling
            depth_map = depth_map * scale
    
    elif args.method == 'geometric':
        print("Using MiDaS with geometric calibration")
        # Get depth map with geometric calibration
        depth_map, reference_point, reference_distance = estimate_depth_with_geometry(
            args.img, model, transform, device, mono_params
        )
        
        if reference_point:
            print(f"Reference point: {reference_point}")
            print(f"Reference distance: {reference_distance:.2f}m")
    
    elif args.method == 'hybrid':
        print("Using hybrid depth estimation (MiDaS + ground geometry)")
        # Get depth map with geometric calibration
        depth_map, reference_point, reference_distance = estimate_depth_with_geometry(
            args.img, model, transform, device, mono_params
        )
        
        # Apply geometric correction with weighted blending
        depth_map = apply_geometric_correction(depth_map, mono_params, weighted=True)
        
        if reference_point:
            print(f"Reference point: {reference_point}")
            print(f"Reference distance: {reference_distance:.2f}m")
    
    elif args.method == 'kalman':
        print("Using Kalman filter to combine MiDaS and geometry")
        # Get MiDaS depth with basic scaling
        midas_depth, reference_point, reference_distance = estimate_depth_with_geometry(
            args.img, model, transform, device, mono_params
        )
        
        # Generate purely geometric depth map
        h, w = img.shape[:2]
        geometric_depth = np.zeros((h, w), dtype=np.float32)
        
        # Calculate vertical FOV
        if 'fov_vertical' in mono_params:
            v_fov = mono_params['fov_vertical']
        elif 'fov_horizontal' in mono_params:
            aspect_ratio = h / w
            v_fov = mono_params['fov_horizontal'] * aspect_ratio
        else:
            v_fov = 49.8  # Default value
            
        # Generate geometric depth map
        for v in range(h):
            geo_distance = calculate_ground_distance(
                v, h, mono_params['camera_height'], mono_params['tilt_angle'], v_fov
            )
            geometric_depth[v, :] = geo_distance
        
        # Handle invalid values
        geometric_depth[geometric_depth < 0] = 100.0
        geometric_depth[geometric_depth == float('inf')] = 100.0
        
        # Apply Kalman filter
        depth_map = apply_kalman_filter(
            midas_depth, 
            geometric_depth, 
            process_noise=args.process_noise,
            measurement_noise=args.measurement_noise
        )
        
        if reference_point:
            print(f"Reference point: {reference_point}")
            print(f"Reference distance: {reference_distance:.2f}m")
    
    # Apply additional filtering if requested
    if args.filter and (depth_map is not None):
        print("Applying depth filtering...")
        depth_map = filter_depth_map(depth_map)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save depth map as numpy array
    np.save(os.path.join(args.output_dir, 'depth_map.npy'), depth_map)
    
    # Save depth map visualization
    depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(os.path.join(args.output_dir, 'depth_map.png'), depth_normalized)
    
    # Save depth colormap
    depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(args.output_dir, 'depth_map_color.png'), depth_colormap)
    
    # For geometric methods, also save the geometric-only depth map
    if args.method in ['hybrid', 'kalman']:
        # Create a purely geometric depth map based on ground plane for comparison
        h, w = img.shape[:2]
        geometric_depth = np.zeros((h, w), dtype=np.float32)
        
        # Calculate vertical FOV
        if 'fov_vertical' in mono_params:
            v_fov = mono_params['fov_vertical']
        elif 'fov_horizontal' in mono_params:
            aspect_ratio = h / w
            v_fov = mono_params['fov_horizontal'] * aspect_ratio
        else:
            v_fov = 49.8  # Default value
            
        # Generate geometric depth map
        for v in range(h):
            geo_distance = calculate_ground_distance(
                v, h, mono_params['camera_height'], mono_params['tilt_angle'], v_fov
            )
            geometric_depth[v, :] = geo_distance
        
        # Handle invalid values
        geometric_depth[geometric_depth < 0] = 100.0
        geometric_depth[geometric_depth == float('inf')] = 100.0
        
        # Save geometric depth map
        geo_normalized = cv2.normalize(geometric_depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imwrite(os.path.join(args.output_dir, 'geometric_depth.png'), geo_normalized)
        
        # Save geometric depth colormap
        geo_colormap = cv2.applyColorMap(geo_normalized, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(args.output_dir, 'geometric_depth_color.png'), geo_colormap)
        
        # Create a simple confidence map for visualization
        confidence_map = np.zeros_like(depth_map)
        for v in range(h):
            # Higher confidence for lower parts of the image (likely ground)
            relative_height = v / h  # 0 at top, 1 at bottom
            confidence_map[v, :] = min(1.0, relative_height * 2.0)  # Confidence increases toward bottom
            
        # Save confidence map
        conf_normalized = (confidence_map * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(args.output_dir, 'confidence_map.png'), conf_normalized)
        
        # Save confidence heatmap
        conf_colormap = cv2.applyColorMap(conf_normalized, cv2.COLORMAP_VIRIDIS)
        cv2.imwrite(os.path.join(args.output_dir, 'confidence_map_color.png'), conf_colormap)
    
    print(f"Results saved to {args.output_dir}")
    
    # Visualize results
    if args.visualize:
        if args.method in ['hybrid', 'kalman']:
            # Create a more detailed visualization with original, geometric and final depth
            plt.figure(figsize=(15, 8))
            
            plt.subplot(2, 2, 1)
            plt.title('Input Image')
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            
            plt.subplot(2, 2, 2)
            plt.title('MiDaS Depth')
            if 'midas_depth' in locals():
                midas_normalized = cv2.normalize(midas_depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                midas_colormap = cv2.applyColorMap(midas_normalized, cv2.COLORMAP_JET)
                plt.imshow(cv2.cvtColor(midas_colormap, cv2.COLOR_BGR2RGB))
            else:
                plt.imshow(cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            
            plt.subplot(2, 2, 3)
            plt.title('Geometric Depth')
            plt.imshow(cv2.cvtColor(geo_colormap, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            
            plt.subplot(2, 2, 4)
            plt.title(f'Final Depth ({args.method})')
            plt.imshow(cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, 'visualization_comparison.png'))
            
            # Create a side-by-side depth & confidence visualization
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.title(f'Depth Map ({args.method})')
            plt.imshow(cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.title('Confidence Map')
            plt.imshow(cv2.cvtColor(conf_colormap, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, 'visualization_confidence.png'))
            
            plt.show()
        else:
            # Standard visualization
            plt.figure(figsize=(12, 6))
            
            plt.subplot(1, 2, 1)
            plt.title('Input Image')
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.title(f'Depth Map ({args.method})')
            plt.imshow(cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, 'visualization.png'))
            plt.show()

if __name__ == "__main__":
    main()