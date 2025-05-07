import numpy as np
import cv2
import os
import glob
import argparse
import pickle
import matplotlib.pyplot as plt

def calibrate_cameras(left_images, right_images=None, pattern_size=(7, 7), 
                    square_size=24.0, baseline=None, debug_mode=False):
    """
    Calibrate cameras using chessboard patterns.
    
    Args:
        left_images: List of left/mono camera image paths
        right_images: List of right camera image paths (for stereo calibration) or None
        pattern_size: Inner corners of the chessboard pattern (width, height)
        square_size: Size of chess square in mm
        baseline: Measured distance between cameras in mm (if known)
        debug_mode: If True, show more detailed debugging information
    
    Returns:
        Dictionary with calibration parameters
    """
    print(f"Starting camera calibration...")
    print(f"Using chessboard pattern with {pattern_size[0]}x{pattern_size[1]} internal corners")
    print(f"Chessboard square size: {square_size} mm")
    
    # Criteria for cornerSubPix
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # Prepare object points (0,0,0), (1,0,0), (2,0,0) ...
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size  # Scale to real-world units
    
    # Create lists to store object and image points
    objpoints = []  # 3D points in real world space
    left_imgpoints = []  # 2D points in left image plane
    right_imgpoints = []  # 2D points in right image plane
    
    # Process all left/mono images
    successful_left = 0
    
    for img_path in left_images:
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply enhanced detection methods from the working script
        # 1. Apply bilateral filter to smooth while preserving edges
        blurred = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # 2. Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # 3. Invert if needed (depending on which color is predominant)
        if np.sum(thresh == 0) < np.sum(thresh == 255):
            thresh = cv2.bitwise_not(thresh)
        
        # 4. Apply morphological operations to clean up noise
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Try to find the chessboard corners - first on threshold image
        ret, corners = cv2.findChessboardCorners(thresh, pattern_size, None)
        
        # If that fails, try with enhanced contrast
        if not ret:
            # CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # Try with enhanced image with special flags
            ret, corners = cv2.findChessboardCorners(
                enhanced, pattern_size, 
                cv2.CALIB_CB_ADAPTIVE_THRESH + 
                cv2.CALIB_CB_NORMALIZE_IMAGE +
                cv2.CALIB_CB_FAST_CHECK
            )
            
        # If corners found, add to data points
        if ret:
            successful_left += 1
            objpoints.append(objp)
            
            # Refine corner positions
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            left_imgpoints.append(corners2)
            
            # Draw and display if in debug mode
            if debug_mode:
                drawn_img = img.copy()
                cv2.drawChessboardCorners(drawn_img, pattern_size, corners2, ret)
                
                # Resize if too large
                max_display_dim = 1200
                h, w = drawn_img.shape[:2]
                scale = min(1.0, max_display_dim / max(h, w))
                if scale < 1.0:
                    display_size = (int(w * scale), int(h * scale))
                    drawn_img = cv2.resize(drawn_img, display_size)
                
                cv2.imshow(f'Chessboard Corners: {os.path.basename(img_path)}', drawn_img)
                key = cv2.waitKey(500)  # Show briefly
                if key == 27:  # ESC key
                    cv2.destroyAllWindows()
                    break
                    
            print(f"Found chessboard in {os.path.basename(img_path)}")
        else:
            print(f"Could not find chessboard in {os.path.basename(img_path)}")
    
    if debug_mode:
        cv2.destroyAllWindows()
    
    print(f"Successfully detected chessboard in {successful_left}/{len(left_images)} left/mono images")
    
    if successful_left < 5:
        raise ValueError(f"Too few successful chessboard detections ({successful_left}). Need at least 5 images.")
    
    # Get image dimensions from first successful image
    img = cv2.imread(left_images[0])
    img_size = (img.shape[1], img.shape[0])  # width, height
    
    # Process all right images (for stereo calibration)
    successful_right = 0
    stereo_pairs = min(len(objpoints), len(right_images)) if right_images else 0
    
    # Clear objpoints and rebuild paired points
    objpoints_stereo = []
    left_imgpoints_stereo = []
    
    if right_images:
        for i, img_path in enumerate(right_images[:stereo_pairs]):
            # Read image
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read image {img_path}")
                continue
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply enhanced detection methods from the working script
            # 1. Apply bilateral filter to smooth while preserving edges
            blurred = cv2.bilateralFilter(gray, 9, 75, 75)
            
            # 2. Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # 3. Invert if needed (depending on which color is predominant)
            if np.sum(thresh == 0) < np.sum(thresh == 255):
                thresh = cv2.bitwise_not(thresh)
            
            # 4. Apply morphological operations to clean up noise
            kernel = np.ones((3, 3), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            # Try to find the chessboard corners - first on threshold image
            ret, corners = cv2.findChessboardCorners(thresh, pattern_size, None)
            
            # If that fails, try with enhanced contrast
            if not ret:
                # CLAHE (Contrast Limited Adaptive Histogram Equalization)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(gray)
                
                # Try with enhanced image with special flags
                ret, corners = cv2.findChessboardCorners(
                    enhanced, pattern_size, 
                    cv2.CALIB_CB_ADAPTIVE_THRESH + 
                    cv2.CALIB_CB_NORMALIZE_IMAGE +
                    cv2.CALIB_CB_FAST_CHECK
                )
                
            # If corners found, add to data points
            if ret:
                successful_right += 1
                
                # For stereo calibration, we need matching pairs
                if i < len(objpoints):
                    objpoints_stereo.append(objpoints[i])
                    left_imgpoints_stereo.append(left_imgpoints[i])
                
                    # Refine corner positions
                    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    right_imgpoints.append(corners2)
                    
                    # Draw and display if in debug mode
                    if debug_mode:
                        drawn_img = img.copy()
                        cv2.drawChessboardCorners(drawn_img, pattern_size, corners2, ret)
                        
                        # Resize if too large
                        max_display_dim = 1200
                        h, w = drawn_img.shape[:2]
                        scale = min(1.0, max_display_dim / max(h, w))
                        if scale < 1.0:
                            display_size = (int(w * scale), int(h * scale))
                            drawn_img = cv2.resize(drawn_img, display_size)
                        
                        cv2.imshow(f'Chessboard Corners: {os.path.basename(img_path)}', drawn_img)
                        key = cv2.waitKey(500)  # Show briefly
                        if key == 27:  # ESC key
                            cv2.destroyAllWindows()
                            break
                            
                    print(f"Found chessboard in {os.path.basename(img_path)}")
                else:
                    print(f"Found chessboard but no matching left image for {os.path.basename(img_path)}")
            else:
                print(f"Could not find chessboard in {os.path.basename(img_path)}")
        
        if debug_mode:
            cv2.destroyAllWindows()
        
        print(f"Successfully detected chessboard in {successful_right}/{len(right_images)} right images")
        print(f"Found {len(right_imgpoints)} stereo pairs")
        
        if len(right_imgpoints) < 5:
            raise ValueError(f"Too few successful stereo pairs ({len(right_imgpoints)}). Need at least 5 pairs.")
    
    # Calibrate left/mono camera
    ret_left, mtx_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(
        objpoints, left_imgpoints, img_size, None, None)
    
    print(f"Left/Mono camera calibration RMS error: {ret_left}")
    
    # If no right images, return mono calibration only
    if right_images is None:
        # Calculate horizontal FOV
        fx = mtx_left[0, 0]
        fov_horizontal = 2 * np.arctan(img_size[0] / (2 * fx)) * 180 / np.pi
        
        # Default mono parameters (to be adjusted by user if needed)
        mono_params = {
            'camera_matrix': mtx_left,
            'dist_coeffs': dist_left,
            'camera_height': 1.2,  # Default placeholder in meters
            'tilt_angle': 15.0,    # Default placeholder in degrees
            'fov_horizontal': fov_horizontal,
            'image_size': img_size
        }
        
        return {
            'mono': mono_params,
            'calibration_error': ret_left
        }
    
    # Calibrate right camera
    ret_right, mtx_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(
        objpoints_stereo, right_imgpoints, img_size, None, None)
    
    print(f"Right camera calibration RMS error: {ret_right}")
    
    # Stereo calibration
    flags = cv2.CALIB_FIX_INTRINSIC  # Use intrinsics from individual calibrations
    ret_stereo, mtx_left, dist_left, mtx_right, dist_right, R, T, E, F = cv2.stereoCalibrate(
        objpoints_stereo, left_imgpoints_stereo, right_imgpoints,
        mtx_left, dist_left, mtx_right, dist_right, img_size,
        flags=flags)
    
    print(f"Stereo calibration RMS error: {ret_stereo}")
    
    # If baseline is provided, scale translation vector
    if baseline:
        # Calculate scale factor
        current_baseline = float(abs(T[0, 0]))  # Access specific element with proper indexing
        scale_factor = baseline / current_baseline
        
        print(f"Scaling translation: Current baseline={current_baseline:.2f}mm, Target={baseline:.2f}mm")
        # Scale translation vector
        T = T * scale_factor
    else:
        # Use calibrated baseline
        baseline = float(abs(T[0, 0]))  # Access specific element with proper indexing
    
    print(f"Final baseline: {baseline:.2f}mm")
    
    # Stereo rectification
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        mtx_left, dist_left, mtx_right, dist_right, img_size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.9)  # Alpha=0.9 for partial zoom
    
    # Calculate horizontal FOV
    fx = mtx_left[0, 0]
    fov_horizontal = 2 * np.arctan(img_size[0] / (2 * fx)) * 180 / np.pi
    
    # Default mono parameters (to be adjusted by user if needed)
    mono_params = {
        'camera_matrix': mtx_left,
        'dist_coeffs': dist_left,
        'camera_height': 1.2,  # Default placeholder in meters
        'tilt_angle': 15.0,    # Default placeholder in degrees
        'fov_horizontal': fov_horizontal,
        'image_size': img_size
    }
    
    # Stereo parameters
    stereo_params = {
        'left_camera_matrix': mtx_left,
        'left_dist_coeffs': dist_left,
        'right_camera_matrix': mtx_right,
        'right_dist_coeffs': dist_right,
        'R': R,
        'T': T,
        'E': E,
        'F': F,
        'R1': R1,
        'R2': R2,
        'P1': P1,
        'P2': P2,
        'Q': Q,
        'baseline': baseline,
        'image_size': img_size,
        'roi1': roi1,
        'roi2': roi2,
        'calibration_error': ret_stereo
    }
    
    # Return calibration data
    return {
        'mono': mono_params,
        'stereo': stereo_params,
        'calibration_error': ret_stereo
    }

def visualize_rectification(calibration_data, left_img_path, right_img_path, output_dir=None):
    """
    Visualize the rectification result for a pair of images
    
    Args:
        calibration_data: Calibration parameters
        left_img_path: Path to a left camera image
        right_img_path: Path to a right camera image
        output_dir: Directory to save visualization (optional)
    """
    # Read images
    left_img = cv2.imread(left_img_path)
    right_img = cv2.imread(right_img_path)
    
    if left_img is None or right_img is None:
        print(f"Error: Could not read images: {left_img_path} or {right_img_path}")
        return
    
    # Get rectification maps
    stereo_params = calibration_data['stereo']
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
    left_rectified = cv2.remap(left_img, left_map1, left_map2, cv2.INTER_LINEAR)
    right_rectified = cv2.remap(right_img, right_map1, right_map2, cv2.INTER_LINEAR)
    
    # Draw horizontal lines for visualization
    line_interval = 50
    for y in range(0, left_img.shape[0], line_interval):
        left_rectified = cv2.line(left_rectified, (0, y), (left_img.shape[1], y), (0, 255, 0), 1)
        right_rectified = cv2.line(right_rectified, (0, y), (right_img.shape[1], y), (0, 255, 0), 1)
    
    # Create a side-by-side display
    display_img = np.hstack((left_rectified, right_rectified))
    
    # Save if output directory provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'rectification_visualization.png')
        cv2.imwrite(output_path, display_img)
        print(f"Rectification visualization saved to {output_path}")
    
    # Resize if too large
    display_scale = 0.5 if display_img.shape[1] > 1920 else 1.0
    display_img = cv2.resize(display_img, (0, 0), fx=display_scale, fy=display_scale)
    
    # Display the image
    cv2.imshow('Stereo Rectification', display_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def print_calibration_parameters(calibration_data):
    """
    Print the most important calibration parameters in a human-readable format.
    
    Args:
        calibration_data: Dictionary with calibration parameters
    """
    print("\n=== CALIBRATION PARAMETERS ===")
    
    # Print monocular parameters
    print("\nMonocular Parameters:")
    mono = calibration_data['mono']
    
    img_size = mono['image_size']
    print(f"Image size: {img_size[0]} x {img_size[1]} pixels")
    
    # Camera matrix
    mtx = mono['camera_matrix']
    print("\nCamera Matrix:")
    print(f"  fx: {mtx[0, 0]:.2f} pixels")
    print(f"  fy: {mtx[1, 1]:.2f} pixels")
    print(f"  cx: {mtx[0, 2]:.2f} pixels")
    print(f"  cy: {mtx[1, 2]:.2f} pixels")
    
    # Distortion coefficients
    dist = mono['dist_coeffs']
    print("\nDistortion Coefficients:")
    print(f"  k1: {dist[0, 0]:.6f}")
    print(f"  k2: {dist[0, 1]:.6f}")
    print(f"  p1: {dist[0, 2]:.6f}")
    print(f"  p2: {dist[0, 3]:.6f}")
    print(f"  k3: {dist[0, 4]:.6f}")
    
    # Field of view
    print(f"\nHorizontal Field of View: {mono['fov_horizontal']:.2f} degrees")
    
    # Print stereo parameters if available
    if 'stereo' in calibration_data:
        stereo = calibration_data['stereo']
        print("\nStereo Parameters:")
        print(f"  Baseline: {stereo['baseline']:.2f} mm")
        print(f"  Calibration Error: {stereo['calibration_error']:.6f}")
        
        # Rotation matrix
        print("\nRotation Matrix (R):")
        print(stereo['R'])
        
        # Translation vector
        print("\nTranslation Vector (T):")
        print(stereo['T'])

def save_calibration(calibration_data, output_dir='.'):
    """
    Save calibration data to files.
    
    Args:
        calibration_data: Dictionary with calibration parameters
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save full calibration data
    calib_file = os.path.join(output_dir, 'camera_calibration.pkl')
    with open(calib_file, 'wb') as f:
        pickle.dump(calibration_data, f)
    
    # Save mono parameters separately
    mono_file = os.path.join(output_dir, 'mono_params.pkl')
    with open(mono_file, 'wb') as f:
        pickle.dump(calibration_data['mono'], f)
    
    # Save parameters in readable text format
    txt_file = os.path.join(output_dir, 'calibration_parameters.txt')
    with open(txt_file, 'w') as f:
        # Redirect print output to file
        import sys
        original_stdout = sys.stdout
        sys.stdout = f
        print_calibration_parameters(calibration_data)
        sys.stdout = original_stdout
    
    print(f"Calibration data saved to {output_dir}")
    print(f"  Full calibration: {calib_file}")
    print(f"  Mono parameters: {mono_file}")
    print(f"  Text summary: {txt_file}")

def extract_inner_edge_points(corners2, pattern_size):
    """
    Extracts the points along the inner edges (top, bottom, left, right rows/columns).
    
    Args:
        corners2: Refined corner points from cv2.cornerSubPix
        pattern_size: Chessboard pattern inner corners (width, height)
    
    Returns:
        Dictionary with edge points
    """
    return {
        'top_edge': corners2[0:pattern_size[0]].reshape(-1, 2),
        'bottom_edge': corners2[-pattern_size[0]:].reshape(-1, 2),
        'left_edge': corners2[::pattern_size[0]].reshape(-1, 2),
        'right_edge': corners2[pattern_size[0]-1::pattern_size[0]].reshape(-1, 2)
    }

def main():
    parser = argparse.ArgumentParser(description='GoPro HERO11 Black Camera Calibration')
    
    # Input arguments
    parser.add_argument('--left_imgs', required=True, 
                       help='Path pattern to left/mono camera images (e.g., "data/calibration/left/*.jpg")')
    parser.add_argument('--right_imgs', 
                       help='Path pattern to right camera images (for stereo) (e.g., "data/calibration/right/*.jpg")')
    parser.add_argument('--pattern_size', default='7x7', 
                       help='Chessboard pattern inner corners (width x height)')
    parser.add_argument('--square_size', type=float, default=24.0, 
                       help='Chessboard square size in mm')
    parser.add_argument('--baseline', type=float, 
                       help='Camera baseline in mm (if known)')
    parser.add_argument('--output_dir', default='./calibration_results', 
                       help='Output directory')
    parser.add_argument('--visualize', action='store_true', 
                       help='Visualize calibration results')
    parser.add_argument('--debug', action='store_true', 
                       help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Parse pattern_size
    pattern_width, pattern_height = map(int, args.pattern_size.split('x'))
    pattern_size = (pattern_width, pattern_height)
    
    # Get image paths using glob (do this here instead of letting shell expand)
    import glob
    left_images = sorted(glob.glob(args.left_imgs))
    right_images = sorted(glob.glob(args.right_imgs)) if args.right_imgs else None
    
    if not left_images:
        print(f"Error: No left/mono images found matching {args.left_imgs}")
        return
    
    print(f"Found {len(left_images)} left/mono images")
    if right_images:
        print(f"Found {len(right_images)} right images")
    
    try:
        # Perform calibration
        calibration_data = calibrate_cameras(
            left_images, right_images, pattern_size, 
            args.square_size, args.baseline, args.debug
        )
        
        # Save calibration data
        save_calibration(calibration_data, args.output_dir)
        
        # Print parameters
        print_calibration_parameters(calibration_data)
        
        # Visualize rectification if requested
        if args.visualize and right_images:
            # Use the first pair of images
            visualize_rectification(
                calibration_data, left_images[0], right_images[0], args.output_dir
            )
    
    except Exception as e:
        print(f"Error during calibration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()