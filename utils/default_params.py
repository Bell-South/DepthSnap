import numpy as np
import pickle
import os
import argparse

def generate_default_params(resolution=(5312, 3552), preset='wide'):
    """
    Generate default camera parameters for GoPro HERO11 Black.
    
    Args:
        resolution: Camera resolution (width, height)
        preset: GoPro field of view preset ('wide', 'linear', 'narrow')
        
    Returns:
        Tuple of (camera_params, mono_params)
    """
    width, height = resolution
    
    # Default parameters based on typical GoPro HERO11 Black values
    # These are approximates and should be adjusted based on your specific camera
    
    # Focal length factor based on preset
    focal_factor = {
        'wide': 0.75,    # Wide FOV (~150°)
        'linear': 1.0,   # Linear FOV (~90°)
        'narrow': 1.5    # Narrow FOV (~65°)
    }
    
    # Distortion factor based on preset
    distortion_factor = {
        'wide': 1.0,      # Strong distortion
        'linear': 0.5,    # Medium distortion
        'narrow': 0.25    # Low distortion
    }
    
    # Calculate focal length based on resolution and preset
    fx = width * focal_factor[preset]
    fy = fx  # Same focal length for both axes
    
    # Principal point at center
    cx = width / 2
    cy = height / 2
    
    # Create camera matrix
    camera_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    
    # Distortion coefficients for wide-angle lens
    # k1, k2, p1, p2, k3
    k1 = -0.22 * distortion_factor[preset]  # Barrel distortion
    k2 = 0.05 * distortion_factor[preset]   # Higher order distortion
    p1 = 0  # Assuming no tangential distortion
    p2 = 0  # Assuming no tangential distortion
    k3 = 0  # Higher order distortion (usually small)
    
    dist_coeffs = np.array([[k1, k2, p1, p2, k3]])
    
    # Calculate horizontal FOV
    fov_horizontal = 2 * np.arctan(width / (2 * fx)) * 180 / np.pi
    
    # Camera parameters
    camera_params = {
        'camera_matrix': camera_matrix,
        'dist_coeffs': dist_coeffs,
        'image_size': resolution,
        'error': 0.0,  # No error because these are default parameters
        'preset': preset
    }
    
    # Monocular parameters for depth estimation
    mono_params = {
        'camera_matrix': camera_matrix,
        'dist_coeffs': dist_coeffs,
        'camera_height': 1.2,  # Default camera height in meters (adjust as needed)
        'tilt_angle': 15.0,    # Default tilt angle in degrees (adjust as needed)
        'fov_horizontal': fov_horizontal,
        'image_size': resolution,
        'preset': preset
    }
    
    # Create a combined parameter set
    combined_params = {
        'mono': mono_params,
        'calibration_error': 0.0  # Default error value
    }
    
    return camera_params, mono_params, combined_params

def generate_stereo_params(left_params, right_params=None, baseline=120.0):
    """
    Generate stereo parameters from individual camera parameters.
    
    Args:
        left_params: Left camera parameters
        right_params: Right camera parameters (if None, use left_params)
        baseline: Distance between cameras in mm
        
    Returns:
        Dictionary with stereo parameters
    """
    # If right_params not provided, use left_params
    if right_params is None:
        right_params = left_params
    
    # Extract camera matrices and distortion coefficients
    left_mtx = left_params['camera_matrix']
    left_dist = left_params['dist_coeffs']
    right_mtx = right_params['camera_matrix']
    right_dist = right_params['dist_coeffs']
    image_size = left_params['image_size']
    
    # Default rotation and translation
    # Assuming cameras are perfectly aligned horizontally
    R = np.eye(3)  # Identity rotation matrix
    T = np.array([[baseline], [0], [0]])  # Translation along X-axis
    
    # Calculate essential and fundamental matrices
    # Essential matrix: E = [t]x * R
    t_cross = np.array([
        [0, -T[2][0], T[1][0]],
        [T[2][0], 0, -T[0][0]],
        [-T[1][0], T[0][0], 0]
    ])
    E = np.dot(t_cross, R)
    
    # Fundamental matrix: F = inv(K2).T * E * inv(K1)
    inv_right_mtx = np.linalg.inv(right_mtx)
    inv_left_mtx = np.linalg.inv(left_mtx)
    F = np.dot(np.dot(inv_right_mtx.T, E), inv_left_mtx)
    
    # Calculate rectification parameters
    # In a real scenario, these would be computed with cv2.stereoRectify()
    # Here we'll use a simplification
    
    # Rectification rotation matrices (identity for simplicity)
    R1 = np.eye(3)
    R2 = np.eye(3)
    
    # Projection matrices
    P1 = np.zeros((3, 4))
    P1[:3, :3] = left_mtx
    
    P2 = np.zeros((3, 4))
    P2[:3, :3] = right_mtx
    P2[0, 3] = -baseline * right_mtx[0, 0]  # Translate second camera by baseline * fx
    
    # Disparity-to-depth mapping matrix
    Q = np.array([
        [1, 0, 0, -cx],
        [0, 1, 0, -cy],
        [0, 0, 0, fx],
        [0, 0, -1/baseline, 0]
    ])
    
    # Region of interest (full image)
    roi1 = (0, 0, image_size[0], image_size[1])
    roi2 = (0, 0, image_size[0], image_size[1])
    
    # Stereo parameters
    stereo_params = {
        'left_camera_matrix': left_mtx,
        'left_dist_coeffs': left_dist,
        'right_camera_matrix': right_mtx,
        'right_dist_coeffs': right_dist,
        'R': R,
        'T': T,
        'E': E,
        'F': F,
        'R1': R1,
        'R2': R2,
        'P1': P1,
        'P2': P2,
        'Q': Q,
        'roi1': roi1,
        'roi2': roi2,
        'baseline': baseline,
        'image_size': image_size,
        'calibration_error': 0.0  # Default error value
    }
    
    # Combined parameters
    combined_params = {
        'mono': left_params,
        'stereo': stereo_params,
        'calibration_error': 0.0  # Default error value
    }
    
    return combined_params

def print_camera_params(params, camera_name="GoPro HERO11 Black"):
    """
    Print camera parameters in a readable format.
    
    Args:
        params: Camera parameters dictionary
        camera_name: Name of the camera
    """
    print(f"\n{camera_name} Default Parameters:")
    
    # Handle different parameter formats
    if 'mono' in params:
        # Combined format
        mono = params['mono']
        image_size = mono['image_size']
        mtx = mono['camera_matrix']
        dist = mono['dist_coeffs']
        preset = mono.get('preset', 'unknown')
        fov = mono.get('fov_horizontal', 0.0)
    else:
        # Individual format
        image_size = params['image_size']
        mtx = params['camera_matrix']
        dist = params['dist_coeffs']
        preset = params.get('preset', 'unknown')
        # Calculate FOV if not provided
        if 'fov_horizontal' in params:
            fov = params['fov_horizontal']
        else:
            fx = mtx[0, 0]
            width = image_size[0]
            fov = 2 * np.arctan(width / (2 * fx)) * 180 / np.pi
    
    print(f"  Image Size (width x height): {image_size[0]} x {image_size[1]} pixels")
    print(f"  Preset: {preset}")
    
    # Camera matrix
    print("\nCamera Matrix:")
    print(f"  fx: {mtx[0, 0]:.2f} pixels")
    print(f"  fy: {mtx[1, 1]:.2f} pixels")
    print(f"  cx: {mtx[0, 2]:.2f} pixels")
    print(f"  cy: {mtx[1, 2]:.2f} pixels")
    
    # Distortion coefficients
    print("\nDistortion Coefficients:")
    print(f"  k1: {dist[0, 0]:.6f}")
    print(f"  k2: {dist[0, 1]:.6f}")
    print(f"  p1: {dist[0, 2]:.6f}")
    print(f"  p2: {dist[0, 3]:.6f}")
    print(f"  k3: {dist[0, 4]:.6f}")
    
    # Field of view
    print(f"\nEstimated Horizontal Field of View: {fov:.2f} degrees")
    
    # Print stereo parameters if available
    if 'stereo' in params:
        stereo = params['stereo']
        print("\nStereo Parameters:")
        print(f"  Baseline: {stereo['baseline']:.2f} mm")

def main():
    parser = argparse.ArgumentParser(description='Generate Default GoPro HERO11 Black Parameters')
    
    # Input arguments
    parser.add_argument('--resolution', default='5312x3552', 
                       help='Camera resolution (WxH)')
    parser.add_argument('--output_dir', default='./calibration_results', 
                       help='Output directory for calibration files')
    parser.add_argument('--preset', choices=['wide', 'linear', 'narrow'], default='wide',
                       help='GoPro field of view preset')
    parser.add_argument('--baseline', type=float, default=120.0,
                       help='Stereo baseline in mm (for stereo params)')
    parser.add_argument('--camera_height', type=float, default=1.2,
                       help='Camera height from ground in meters')
    parser.add_argument('--tilt_angle', type=float, default=15.0,
                       help='Camera tilt angle in degrees')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse resolution
    width, height = map(int, args.resolution.split('x'))
    resolution = (width, height)
    
    # Generate default parameters
    camera_params, mono_params, combined_params = generate_default_params(resolution, args.preset)
    
    # Update mono parameters with command line values
    mono_params['camera_height'] = args.camera_height
    mono_params['tilt_angle'] = args.tilt_angle
    combined_params['mono']['camera_height'] = args.camera_height
    combined_params['mono']['tilt_angle'] = args.tilt_angle
    
    # Generate stereo parameters
    stereo_params = generate_stereo_params(camera_params, None, args.baseline)
    
    # Save parameters
    camera_file = os.path.join(args.output_dir, 'default_camera_calib.pkl')
    with open(camera_file, 'wb') as f:
        pickle.dump(camera_params, f)
    
    mono_file = os.path.join(args.output_dir, 'default_mono_params.pkl')
    with open(mono_file, 'wb') as f:
        pickle.dump(mono_params, f)
    
    combined_file = os.path.join(args.output_dir, 'default_combined_params.pkl')
    with open(combined_file, 'wb') as f:
        pickle.dump(combined_params, f)
    
    stereo_file = os.path.join(args.output_dir, 'default_stereo_params.pkl')
    with open(stereo_file, 'wb') as f:
        pickle.dump(stereo_params, f)
    
    # Print parameters
    print("Default parameters generated for GoPro HERO11 Black")
    print(f"Resolution: {width}x{height}")
    print(f"Preset: {args.preset}")
    print(f"Camera height: {args.camera_height} m")
    print(f"Tilt angle: {args.tilt_angle} degrees")
    print(f"Stereo baseline: {args.baseline} mm")
    
    print(f"\nParameters saved to {args.output_dir}:")
    print(f"  Camera parameters: {camera_file}")
    print(f"  Mono parameters: {mono_file}")
    print(f"  Combined parameters: {combined_file}")
    print(f"  Stereo parameters: {stereo_file}")
    
    # Print detailed camera parameters
    print_camera_params(camera_params, f"GoPro HERO11 Black ({args.preset})")

if __name__ == "__main__":
    main()