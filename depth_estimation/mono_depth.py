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
        import torch
        # Try to import MiDaS directly
        try:
            import midas.transforms
            from midas.model_loader import load_model
            print("MiDaS already installed")
            
            # Load model
            model_path = None  # Use default path
            model, transform = load_model(model_type, model_path, optimize=True)
            
            # Move model to device and set to evaluation mode
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.eval()
            
            return model, transform, device
            
        except ImportError:
            # If MiDaS isn't installed as a module, try PyTorch Hub
            print("MiDaS module not found, trying PyTorch Hub...")
            
            # Use torch.hub.load instead
            try:
                # For DPT models
                if model_type in ["DPT_Large", "DPT_Hybrid"]:
                    model = torch.hub.load("intel-isl/MiDaS", model_type)
                    transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform
                # For MiDaS small
                elif model_type == "MiDaS_small":
                    model = torch.hub.load("intel-isl/MiDaS", model_type)
                    transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
                else:
                    raise ValueError(f"Unknown model type: {model_type}")
                
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model.to(device)
                model.eval()
                
                print(f"Successfully loaded MiDaS {model_type} via PyTorch Hub")
                return model, transform, device
                
            except Exception as e:
                print(f"PyTorch Hub loading failed: {e}")
                raise
            
    except Exception as e:
        print(f"Error loading MiDaS model: {e}")
        print("Falling back to placeholder implementation for testing")
        
        # Create a placeholder model for testing/debugging
        import torch
        import numpy as np
        
        class DummyModel:
            def __init__(self):
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            def to(self, device):
                return self
                
            def eval(self):
                return self
                
            def __call__(self, x):
                # Return random tensor of appropriate shape for testing
                b, c, h, w = x.shape
                return torch.rand((b, 1, h, w), device=self.device)
        
        class DummyTransform:
            def __call__(self, img):
                # Assumes img is a numpy array of shape (H, W, 3)
                # Convert to tensor of shape (1, 3, H, W)
                tensor = torch.from_numpy(img.transpose((2, 0, 1))).float() / 255.0
                return tensor.unsqueeze(0)
        
        model = DummyModel()
        transform = DummyTransform()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print("WARNING: Using dummy model for testing! No real depth estimation will be performed.")
        return model, transform, device
    
def estimate_depth_with_geometry(img, model, transform, mono_params=None):
    """
    Estimate depth using MiDaS with geometric ground-plane correction.
    
    Args:
        img: Input image (BGR)
        model: MiDaS model
        transform: MiDaS transform
        mono_params: Dictionary with camera parameters (height, tilt, FOV)
        
    Returns:
        depth_map: Estimated depth map in meters
        reference_point: Reference point used for calibration
        reference_distance: Reference distance in meters
    """
    # Convert to RGB for MiDaS
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Get image dimensions
    image_height, image_width = img.shape[:2]
    
    # Generate MiDaS depth prediction
    input_batch = transform(img_rgb).to(next(model.parameters()).device)
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
    try:
        # Convert from BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        
        # Apply transform
        input_batch = transform(img_rgb).to(device)
        
        # Make prediction
        with torch.no_grad():
            prediction = model(input_batch)
            
            # Ensure prediction has the right shape
            if len(prediction.shape) == 4:
                prediction = prediction.squeeze(1)
                
            # Resize to original dimensions
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        # Convert to numpy
        output = prediction.cpu().numpy()
        
    except Exception as e:
        print(f"Error during depth estimation: {e}")
        print("Using fallback random depth generation for testing")
        
        # Generate random depth map for testing
        output = np.random.rand(*img.shape[:2])
    
    # Convert to float32
    relative_depth = output.astype(np.float32)
    
    # Normalize depth values (higher value = closer)
    # MiDaS might output inverse depth where higher values are closer
    depth_min = relative_depth.min()
    depth_max = relative_depth.max()
    
    if depth_max - depth_min > 0:
        relative_depth = (relative_depth - depth_min) / (depth_max - depth_min)
    else:
        relative_depth = np.zeros_like(relative_depth)
    
    # Scale to meters if mono_params provided
    if mono_params:
        # Use camera height and tilt to scale depth
        camera_height = mono_params['camera_height']  # in meters
        tilt_angle = mono_params['tilt_angle']  # in degrees
        
        # Convert tilt angle to radians
        tilt_rad = np.deg2rad(tilt_angle)
        
        # Get image dimensions
        height, width = img.shape[:2]
        
        # Create coordinate grid
        y_coords, x_coords = np.mgrid[0:height, 0:width]
        
        # Normalize coordinates to be centered at principal point
        cx = mono_params['camera_matrix'][0, 2]
        cy = mono_params['camera_matrix'][1, 2]
        fx = mono_params['camera_matrix'][0, 0]
        
        x_normalized = (x_coords - cx) / fx
        y_normalized = (y_coords - cy) / fx
        
        # Calculate angle for each pixel
        angles = np.arctan2(y_normalized, 1.0) + tilt_rad
        
        # Scale factor is proportional to camera height / sin(angle)
        # Avoid division by zero with small epsilon
        epsilon = 1e-6
        scale_factors = camera_height / np.maximum(np.sin(angles), epsilon)
        
        # Use a simplified scaling approach
        # Map from relative depth to real-world depth
        # We invert relative_depth here since MiDaS predicts inverse depth
        # (1 - relative_depth) gives us proper depth ordering
        inverse_depth = 1.0 - relative_depth
        
        # Apply scale based on ground plane assumption
        absolute_depth = inverse_depth * scale_factors
        
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
    parser.add_argument('--method', choices=['midas', 'geometric', 'hybrid', 'kalman'], 
                       default='midas', help='Depth estimation method')
    
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
        args.method = 'midas'
    
    # Load MiDaS model
    model, transform, device = load_midas_model(args.model_type)
    
    if args.method == 'midas':
        print("Using standard MiDaS depth estimation")
        # Get depth map
        _, depth_map = estimate_depth(img, model, transform, auto_calibrate=False)[:2]
        
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
            img, model, transform, mono_params
        )
        
        if reference_point:
            print(f"Reference point: {reference_point}")
            print(f"Reference distance: {reference_distance:.2f}m")
    
    elif args.method == 'hybrid':
        print("Using hybrid depth estimation (MiDaS + ground geometry)")
        # Get depth map with geometric calibration
        depth_map, reference_point, reference_distance = estimate_depth_with_geometry(
            img, model, transform, mono_params
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
            img, model, transform, mono_params
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
        depth_map = apply_kalman_filter(midas_depth, geometric_depth)
    
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
    
    print(f"Results saved to {args.output_dir}")
    
    # Visualize results
    if args.visualize:
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