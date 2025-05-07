import cv2
import numpy as np
import pickle
import argparse
import matplotlib.pyplot as plt
import os
import torch

def load_midas_model(model_type="DPT_Large"):
    """
    Load MiDaS model for monocular depth estimation.
    
    Args:
        model_type: MiDaS model type ('DPT_Large', 'DPT_Hybrid', or 'MiDaS_small')
        
    Returns:
        MiDaS model and transform
    """
    print(f"Loading MiDaS model: {model_type}")
    
    try:
        # Import MiDaS modules
        from midas.model_loader import load_model
        
    except ImportError:
        # If not installed, download via pip
        import subprocess
        print("MiDaS not found, installing required dependencies...")
        subprocess.check_call(["pip", "install", "timm"])
        subprocess.check_call(["pip", "install", "git+https://github.com/isl-org/MiDaS.git"])
        
        # Now import should work
        from midas.model_loader import load_model
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model_path = None  # Use default path
    model, transform = load_model(model_type, model_path, optimize=True)
    
    # Move model to device and set to evaluation mode
    model.to(device)
    model.eval()
    
    return model, transform, device

def estimate_midas_depth(img, model, transform, device, mono_params=None):
    """
    Estimate depth using MiDaS model.
    
    Args:
        img: Input image (BGR)
        model: MiDaS model
        transform: MiDaS transform
        device: Computation device (cuda or cpu)
        mono_params: Monocular parameters for scaling (optional)
        
    Returns:
        Relative depth map and scaled depth map (if mono_params provided)
    """
    # Convert from BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    
    # Apply transform
    input_batch = transform(img_rgb).to(device)
    
    # Make prediction
    with torch.no_grad():
        prediction = model(input_batch)
        
        # MiDaS outputs disparity, not depth
        # Convert to numpy array and resize to original image dimensions
        output = prediction.squeeze().cpu().numpy()
        
        # Resize to original image dimensions
        output = cv2.resize(output, (img.shape[1], img.shape[0]), 
                           interpolation=cv2.INTER_CUBIC)
    
    # MiDaS returns inverse depth, so we need to invert it
    # First normalize the output
    output_min = output.min()
    output_max = output.max()
    output = (output - output_min) / (output_max - output_min)
    
    # Inverse the normalized output to get relative depth
    # (closer objects have smaller values, farther objects have larger values)
    relative_depth = 1.0 - output
    
    # Scale to meters if mono_params provided
    if mono_params:
        # Use camera height and tilt to scale depth
        camera_height = mono_params['camera_height']  # in meters
        tilt_angle = mono_params['tilt_angle']  # in degrees
        
        # Convert tilt angle to radians
        tilt_rad = np.deg2rad(tilt_angle)
        
        # Get image dimensions
        height, width = img.shape[:2]
        
        # Get principal point and focal length
        cx = mono_params['camera_matrix'][0, 2]
        cy = mono_params['camera_matrix'][1, 2]
        fx = mono_params['camera_matrix'][0, 0]
        
        # Create coordinate grid (normalized coordinates)
        x_coords, y_coords = np.meshgrid(
            (np.arange(width) - cx) / fx,
            (np.arange(height) - cy) / fx
        )
        
        # Calculate angle for each pixel (relative to camera's optical axis)
        angles = np.arctan2(y_coords, np.sqrt(1.0 + x_coords**2)) + tilt_rad
        
        # Calculate ground distance for each pixel
        # Using the camera height and angle, apply basic trigonometry
        # distance = camera_height / tan(angle)
        # Avoid division by zero or negative values
        valid_mask = np.sin(angles) > 0.05  # About 3 degrees threshold
        
        # Initialize absolute depth map
        absolute_depth = np.zeros_like(relative_depth)
        
        # Calculate scaling factor based on ground plane geometry
        ground_distances = np.zeros_like(angles)
        ground_distances[valid_mask] = camera_height / np.sin(angles[valid_mask])
        
        # Apply scaling to get metric depth
        # Scale the relative depth to match the ground truth at the ground plane
        # This is a simple approximation that works reasonably well
        scale_factor = np.median(ground_distances[valid_mask]) / np.median(relative_depth[valid_mask])
        absolute_depth = relative_depth * scale_factor
        
        # Apply reasonable limits
        max_depth = 40.0  # meters
        absolute_depth[absolute_depth > max_depth] = max_depth
        absolute_depth[absolute_depth < 0] = 0
        
        return relative_depth, absolute_depth
    
    return relative_depth, None

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
                
                # Override with command line arguments if provided
                if args.camera_height:
                    mono_params['camera_height'] = args.camera_height
                if args.tilt_angle:
                    mono_params['tilt_angle'] = args.tilt_angle
                
                print(f"Loaded calibration parameters:")
                print(f"  Camera height: {mono_params['camera_height']:.2f} m")
                print(f"  Tilt angle: {mono_params['tilt_angle']:.2f} degrees")
                print(f"  Field of view: {mono_params['fov_horizontal']:.2f} degrees")
                
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
    
    # Load MiDaS model
    model, transform, device = load_midas_model(args.model_type)
    
    # Estimate depth
    print("Estimating depth...")
    relative_depth, absolute_depth = estimate_midas_depth(img, model, transform, device, mono_params)
    
    # Apply additional filtering if requested
    if args.filter and (relative_depth is not None):
        print("Applying depth filtering...")
        if absolute_depth is not None:
            absolute_depth = filter_depth_map(absolute_depth)
        relative_depth = filter_depth_map(relative_depth)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save relative depth map
    np.save(os.path.join(args.output_dir, 'relative_depth.npy'), relative_depth)
    
    # Save relative depth map visualization
    relative_depth_normalized = cv2.normalize(relative_depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(os.path.join(args.output_dir, 'relative_depth.png'), relative_depth_normalized)
    
    # Save relative depth colormap
    relative_depth_colormap = cv2.applyColorMap(relative_depth_normalized, cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(args.output_dir, 'relative_depth_color.png'), relative_depth_colormap)
    
    # Save absolute depth map if available
    if absolute_depth is not None:
        # Save raw depth as numpy array
        np.save(os.path.join(args.output_dir, 'absolute_depth.npy'), absolute_depth)
        
        # Save normalized absolute depth map
        abs_depth_normalized = cv2.normalize(absolute_depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imwrite(os.path.join(args.output_dir, 'absolute_depth.png'), abs_depth_normalized)
        
        # Save absolute depth colormap
        abs_depth_colormap = cv2.applyColorMap(abs_depth_normalized, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(args.output_dir, 'absolute_depth_color.png'), abs_depth_colormap)
        
        # Use absolute depth for visualization
        depth_for_vis = absolute_depth
        depth_colormap = abs_depth_colormap
    else:
        # Use relative depth for visualization
        depth_for_vis = relative_depth
        depth_colormap = relative_depth_colormap
    
    print(f"Results saved to {args.output_dir}")
    
    # Visualize results
    if args.visualize:
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.title('Input Image')
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.title('Depth Map')
        plt.imshow(cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'visualization.png'))
        plt.show()

if __name__ == "__main__":
    main()