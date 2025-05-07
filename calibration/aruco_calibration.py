import numpy as np
import cv2
import os
import glob
import argparse
import pickle
import matplotlib.pyplot as plt

def calibrate_with_aruco(images_path, marker_size_mm=50.0, dictionary_name='DICT_6X6_250', 
                        baseline=None, visualize=False):
    """
    Calibrate camera using ArUco markers.
    
    Args:
        images_path: Path to calibration images
        marker_size_mm: ArUco marker size in mm
        dictionary_name: Name of ArUco dictionary to use
        baseline: Measured distance between cameras in mm (for stereo)
        visualize: Whether to visualize detected markers
        
    Returns:
        Dictionary with calibration parameters or None if calibration fails
    """
    print(f"Calibrating camera using ArUco markers (size: {marker_size_mm}mm)")
    
    # Get ArUco dictionary
    ARUCO_DICT = {
        "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
        "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
        "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
        "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
        "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
        "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
        "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
        "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
        "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
        "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
        "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
        "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
        "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
        "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
        "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
        "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
        "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL
    }
    
    if dictionary_name not in ARUCO_DICT:
        print(f"Error: Dictionary {dictionary_name} not found")
        print(f"Available dictionaries: {', '.join(ARUCO_DICT.keys())}")
        return None
    
    # Set up ArUco detector
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[dictionary_name])
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    
    # Get list of images
    if os.path.isdir(images_path):
        # Check for left and right folders for stereo
        left_dir = os.path.join(images_path, 'left')
        right_dir = os.path.join(images_path, 'right')
        
        if os.path.isdir(left_dir) and os.path.isdir(right_dir):
            # Stereo calibration
            left_images = glob.glob(os.path.join(left_dir, '*.jpg')) + \
                         glob.glob(os.path.join(left_dir, '*.png')) + \
                         glob.glob(os.path.join(left_dir, '*.jpeg'))
            
            right_images = glob.glob(os.path.join(right_dir, '*.jpg')) + \
                          glob.glob(os.path.join(right_dir, '*.png')) + \
                          glob.glob(os.path.join(right_dir, '*.jpeg'))
            
            # Sort images to ensure consistent ordering
            left_images.sort()
            right_images.sort()
            
            if not left_images or not right_images:
                print(f"No images found in {left_dir} or {right_dir}")
                return None
            
            # Calibrate stereo cameras
            return calibrate_stereo_aruco(left_images, right_images, marker_size_mm, 
                                        dictionary_name, baseline, visualize)
        else:
            # Mono calibration
            images = glob.glob(os.path.join(images_path, '*.jpg')) + \
                    glob.glob(os.path.join(images_path, '*.png')) + \
                    glob.glob(os.path.join(images_path, '*.jpeg'))
    else:
        # Assume it's a glob pattern
        images = glob.glob(images_path)
    
    if not images:
        print(f"No images found at {images_path}")
        return None
    
    # Sort images to ensure consistent ordering
    images.sort()
    
    # Get image size from first image
    img = cv2.imread(images[0])
    if img is None:
        print(f"Could not read image {images[0]}")
        return None
    
    img_size = (img.shape[1], img.shape[0])  # width, height
    
    print(f"Found {len(images)} images. Processing...")
    
    # Arrays to store object points and image points
    obj_points = []  # 3D points in real world space
    img_points = []  # 2D points in image plane
    
    # Process each image
    successful_images = 0
    
    for idx, img_path in enumerate(images):
        img = cv2.imread(img_path)
        if img is None:
            print(f"  Skipping image {idx+1}/{len(images)}: Could not read image")
            continue
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Try to enhance image using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Detect ArUco markers
        corners, ids, rejected = detector.detectMarkers(enhanced)
        
        if ids is not None and len(ids) > 0:
            # ArUco markers detected
            successful_images += 1
            
            # Draw detected markers if visualization is enabled
            if visualize:
                detected_img = img.copy()
                cv2.aruco.drawDetectedMarkers(detected_img, corners, ids)
                
                # Resize for display if image is too large
                max_display_dim = 1200
                h, w = detected_img.shape[:2]
                scale = min(1.0, max_display_dim / max(h, w))
                
                if scale < 1.0:
                    display_size = (int(w * scale), int(h * scale))
                    detected_img = cv2.resize(detected_img, display_size)
                
                cv2.imshow(f'Detected ArUco Markers', detected_img)
                cv2.waitKey(1000)  # Show for 1 second
            
            # For each detected marker, add object and image points
            for i, corner in enumerate(corners):
                # Get marker ID
                marker_id = ids[i][0]
                
                # Define 3D coordinates for this marker
                # Using marker size, with the marker centered at the origin
                half_size = marker_size_mm / 2
                objp = np.array([
                    [-half_size, half_size, 0],    # Top-left
                    [half_size, half_size, 0],     # Top-right
                    [half_size, -half_size, 0],    # Bottom-right
                    [-half_size, -half_size, 0]    # Bottom-left
                ])
                
                # Add object points
                obj_points.append(objp)
                
                # Add image points (corners)
                img_points.append(corner.reshape(-1, 2))
            
            print(f"  Processed image {idx+1}/{len(images)}: {len(corners)} markers detected")
        else:
            print(f"  Processed image {idx+1}/{len(images)}: No markers detected")
    
    if visualize:
        cv2.destroyAllWindows()
    
    if successful_images == 0:
        print("No ArUco markers found in any images")
        return None
    
    print(f"Successfully detected markers in {successful_images}/{len(images)} images")
    
    # Perform camera calibration
    flags = 0
    flags |= cv2.CALIB_RATIONAL_MODEL  # Use rational polynomial model for better distortion modeling
    
    print("Calculating camera parameters...")
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, img_size, None, None, flags=flags
    )
    
    # Calculate reprojection error
    mean_error = 0
    for i in range(len(obj_points)):
        img_points2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(img_points[i], img_points2, cv2.NORM_L2) / len(img_points2)
        mean_error += error
    
    mean_error /= len(obj_points)
    print(f"Calibration complete. Mean reprojection error: {mean_error:.6f} pixels")
    
    # Calculate Field of View (FOV)
    fx = camera_matrix[0, 0]
    fov_horizontal = 2 * np.arctan(img_size[0] / (2 * fx)) * 180 / np.pi
    
    # Monocular parameters for depth estimation
    mono_params = {
        'camera_matrix': camera_matrix,
        'dist_coeffs': dist_coeffs,
        'camera_height': 1.2,  # Default camera height in meters (adjust as needed)
        'tilt_angle': 15.0,    # Default tilt angle in degrees (adjust as needed)
        'fov_horizontal': fov_horizontal,
        'image_size': img_size
    }
    
    # Calibration data
    calibration_data = {
        'mono': mono_params,
        'calibration_error': mean_error
    }
    
    return calibration_data

def calibrate_stereo_aruco(left_images, right_images, marker_size_mm=50.0, 
                        dictionary_name='DICT_6X6_250', baseline=None, visualize=False):
    """
    Calibrate stereo cameras using ArUco markers.
    
    Args:
        left_images: List of left camera image paths
        right_images: List of right camera image paths
        marker_size_mm: ArUco marker size in mm
        dictionary_name: Name of ArUco dictionary to use
        baseline: Measured distance between cameras in mm
        visualize: Whether to visualize detected markers
        
    Returns:
        Dictionary with calibration parameters or None if calibration fails
    """
    print(f"Calibrating stereo cameras using ArUco markers (size: {marker_size_mm}mm)")
    
    # Ensure equal number of images
    num_images = min(len(left_images), len(right_images))
    left_images = left_images[:num_images]
    right_images = right_images[:num_images]
    
    print(f"Using {num_images} image pairs for calibration")
    
    # Get ArUco dictionary
    ARUCO_DICT = {
        "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
        "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
        "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
        "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
        "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
        "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
        "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
        "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
        "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
        "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
        "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
        "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
        "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
        "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
        "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
        "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
        "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL
    }
    
    # Set up ArUco detector
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[dictionary_name])
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    
    # Get image size from first images
    left_img = cv2.imread(left_images[0])
    right_img = cv2.imread(right_images[0])
    
    if left_img is None or right_img is None:
        print(f"Could not read image pair: {left_images[0]} or {right_images[0]}")
        return None
    
    # Ensure both cameras have the same resolution
    if left_img.shape != right_img.shape:
        print(f"Warning: Left and right images have different resolutions")
        print(f"Left: {left_img.shape}, Right: {right_img.shape}")
        print("Calibration may be less accurate. Resizing images...")
        
        # Resize right images to match left
        for i in range(len(right_images)):
            right_img = cv2.imread(right_images[i])
            if right_img is not None:
                right_img = cv2.resize(right_img, (left_img.shape[1], left_img.shape[0]))
                cv2.imwrite(right_images[i], right_img)
    
    img_size = (left_img.shape[1], left_img.shape[0])  # width, height
    
    # Arrays to store object points and image points
    obj_points = []  # 3D points in real world space
    left_img_points = []  # 2D points in left image plane
    right_img_points = []  # 2D points in right image plane
    
    # Process each image pair
    successful_pairs = 0
    
    for idx in range(num_images):
        left_img_path = left_images[idx]
        right_img_path = right_images[idx]
        
        left_img = cv2.imread(left_img_path)
        right_img = cv2.imread(right_img_path)
        
        if left_img is None or right_img is None:
            print(f"  Skipping pair {idx+1}/{num_images}: Could not read images")
            continue
        
        # Convert to grayscale
        left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
        
        # Enhance images using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        left_enhanced = clahe.apply(left_gray)
        right_enhanced = clahe.apply(right_gray)
        
        # Detect ArUco markers in both images
        left_corners, left_ids, left_rejected = detector.detectMarkers(left_enhanced)
        right_corners, right_ids, right_rejected = detector.detectMarkers(right_enhanced)
        
        # Only use this pair if both images have markers
        if (left_ids is not None and len(left_ids) > 0 and 
            right_ids is not None and len(right_ids) > 0):
            
            # Find common markers
            common_ids = []
            left_idx = []
            right_idx = []
            
            for i, left_id in enumerate(left_ids):
                for j, right_id in enumerate(right_ids):
                    if left_id[0] == right_id[0]:
                        common_ids.append(left_id[0])
                        left_idx.append(i)
                        right_idx.append(j)
            
            # Only use this pair if there are common markers
            if len(common_ids) > 0:
                successful_pairs += 1
                
                # Draw detected markers if visualization is enabled
                if visualize:
                    left_detected = left_img.copy()
                    right_detected = right_img.copy()
                    
                    cv2.aruco.drawDetectedMarkers(left_detected, left_corners, left_ids)
                    cv2.aruco.drawDetectedMarkers(right_detected, right_corners, right_ids)
                    
                    # Resize for display if image is too large
                    max_display_dim = 1200
                    h, w = left_detected.shape[:2]
                    scale = min(1.0, max_display_dim / max(h, w))
                    
                    if scale < 1.0:
                        left_display = cv2.resize(left_detected, (int(w * scale), int(h * scale)))
                        right_display = cv2.resize(right_detected, (int(w * scale), int(h * scale)))
                    else:
                        left_display = left_detected
                        right_display = right_detected
                    
                    # Display side by side
                    display_img = np.hstack((left_display, right_display))
                    cv2.imshow(f'Detected ArUco Markers', display_img)
                    cv2.waitKey(1000)  # Show for 1 second
                
                # For each common marker, add object and image points
                for i, marker_id in enumerate(common_ids):
                    # Define 3D coordinates for this marker
                    half_size = marker_size_mm / 2
                    objp = np.array([
                        [-half_size, half_size, 0],    # Top-left
                        [half_size, half_size, 0],     # Top-right
                        [half_size, -half_size, 0],    # Bottom-right
                        [-half_size, -half_size, 0]    # Bottom-left
                    ])
                    
                    # Add object points
                    obj_points.append(objp)
                    
                    # Add image points (corners)
                    left_img_points.append(left_corners[left_idx[i]].reshape(-1, 2))
                    right_img_points.append(right_corners[right_idx[i]].reshape(-1, 2))
                
                print(f"  Processed pair {idx+1}/{num_images}: {len(common_ids)} common markers")
            else:
                print(f"  Processed pair {idx+1}/{num_images}: No common markers detected")
        else:
            print(f"  Processed pair {idx+1}/{num_images}: No markers detected in one or both images")
    
    if visualize:
        cv2.destroyAllWindows()
    
    if successful_pairs == 0:
        print("No common ArUco markers found in any image pairs")
        return None
    
    print(f"Successfully detected common markers in {successful_pairs}/{num_images} image pairs")
    
    # Perform calibration for each camera separately
    flags = 0
    flags |= cv2.CALIB_RATIONAL_MODEL  # Use rational polynomial model for better distortion modeling
    
    # Left camera calibration
    print("Calibrating left camera...")
    left_ret, left_mtx, left_dist, left_rvecs, left_tvecs = cv2.calibrateCamera(
        obj_points, left_img_points, img_size, None, None, flags=flags
    )
    
    # Right camera calibration
    print("Calibrating right camera...")
    right_ret, right_mtx, right_dist, right_rvecs, right_tvecs = cv2.calibrateCamera(
        obj_points, right_img_points, img_size, None, None, flags=flags
    )
    
    # Stereo calibration
    print("Performing stereo calibration...")
    flags = cv2.CALIB_FIX_INTRINSIC  # Use intrinsics from individual calibrations
    stereo_ret, left_mtx, left_dist, right_mtx, right_dist, R, T, E, F = cv2.stereoCalibrate(
        obj_points, left_img_points, right_img_points,
        left_mtx, left_dist, right_mtx, right_dist, img_size,
        flags=flags
    )
    
    # If baseline is provided, scale translation vector
    if baseline:
        # Calculate scale factor
        current_baseline = abs(T[0])
        scale_factor = baseline / current_baseline
        
        print(f"Scaling translation: Current baseline={current_baseline:.2f}mm, Target={baseline:.2f}mm")
        # Scale translation vector
        T = T * scale_factor
    else:
        # Use calibrated baseline
        baseline = abs(T[0])
    
    print(f"Final baseline: {baseline:.2f}mm")
    
    # Stereo rectification
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        left_mtx, left_dist, right_mtx, right_dist, img_size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.9  # Alpha=0.9 for partial zoom
    )
    
    # Calculate reprojection error
    mean_error = 0
    for i in range(len(obj_points)):
        left_img_points2, _ = cv2.projectPoints(obj_points[i], left_rvecs[i], left_tvecs[i], left_mtx, left_dist)
        error = cv2.norm(left_img_points[i], left_img_points2, cv2.NORM_L2) / len(left_img_points2)
        mean_error += error
    
    left_error = mean_error / len(obj_points)
    
    mean_error = 0
    for i in range(len(obj_points)):
        right_img_points2, _ = cv2.projectPoints(obj_points[i], right_rvecs[i], right_tvecs[i], right_mtx, right_dist)
        error = cv2.norm(right_img_points[i], right_img_points2, cv2.NORM_L2) / len(right_img_points2)
        mean_error += error
    
    right_error = mean_error / len(obj_points)
    
    print(f"Calibration complete.")
    print(f"Left camera reprojection error: {left_error:.6f} pixels")
    print(f"Right camera reprojection error: {right_error:.6f} pixels")
    print(f"Stereo calibration error: {stereo_ret:.6f}")
    
    # Calculate horizontal FOV
    fx = left_mtx[0, 0]
    fov_horizontal = 2 * np.arctan(img_size[0] / (2 * fx)) * 180 / np.pi
    
    # Monocular parameters for depth estimation
    mono_params = {
        'camera_matrix': left_mtx,
        'dist_coeffs': left_dist,
        'camera_height': 1.2,  # Default camera height in meters (adjust as needed)
        'tilt_angle': 15.0,    # Default tilt angle in degrees (adjust as needed)
        'fov_horizontal': fov_horizontal,
        'image_size': img_size
    }
    
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
        'baseline': baseline,
        'image_size': img_size,
        'roi1': roi1,
        'roi2': roi2,
        'calibration_error': stereo_ret
    }
    
    # Return calibration data
    return {
        'mono': mono_params,
        'stereo': stereo_params,
        'calibration_error': stereo_ret
    }

def generate_aruco_marker(dictionary_name="DICT_6X6_250", marker_id=0, size=200, save_path="aruco_marker.png"):
    """
    Generate an ArUco marker image for printing.
    
    Args:
        dictionary_name: Name of ArUco dictionary to use
        marker_id: ID of the marker to generate
        size: Size of the output marker image in pixels
        save_path: Path to save the marker image
    """
    # Get ArUco dictionary
    ARUCO_DICT = {
        "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
        "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
        "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
        "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
        "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
        "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
        "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
        "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
        "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
        "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
        "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
        "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
        "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
        "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
        "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
        "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
        "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL
    }
    
    if dictionary_name not in ARUCO_DICT:
        print(f"Error: Dictionary {dictionary_name} not found")
        print(f"Available dictionaries: {', '.join(ARUCO_DICT.keys())}")
        return
    
    # Set up ArUco dictionary
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[dictionary_name])
    
    # Generate marker image
    marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, size)
    
    # Add border for better printing
    border_size = size // 10
    bordered_img = cv2.copyMakeBorder(marker_img, border_size, border_size, border_size, border_size, 
                                    cv2.BORDER_CONSTANT, value=255)
    
    # Add marker ID and size text
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"ID: {marker_id}, Dict: {dictionary_name}"
    cv2.putText(bordered_img, text, (border_size, size + border_size + border_size//2), 
               font, 0.6, 0, 1)
    
    # Save marker image
    cv2.imwrite(save_path, bordered_img)
    print(f"ArUco marker saved to {save_path}")

def generate_aruco_board(dictionary_name="DICT_6X6_250", markers_x=4, markers_y=5, 
                       marker_size=80, marker_spacing=20, save_path="aruco_board.png"):
    """
    Generate an ArUco marker board image for printing.
    
    Args:
        dictionary_name: Name of ArUco dictionary to use
        markers_x: Number of markers in X direction
        markers_y: Number of markers in Y direction
        marker_size: Size of each marker in pixels
        marker_spacing: Spacing between markers in pixels
        save_path: Path to save the board image
    """
    # Get ArUco dictionary
    ARUCO_DICT = {
        "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
        "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
        "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
        "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
        "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
        "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
        "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
        "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
        "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
        "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
        "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
        "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
        "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
        "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
        "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
        "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
        "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL
    }
    
    if dictionary_name not in ARUCO_DICT:
        print(f"Error: Dictionary {dictionary_name} not found")
        print(f"Available dictionaries: {', '.join(ARUCO_DICT.keys())}")
        return
    
    # Set up ArUco dictionary
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[dictionary_name])
    
    # Calculate board size
    board_width = markers_x * marker_size + (markers_x - 1) * marker_spacing + 2 * marker_spacing
    board_height = markers_y * marker_size + (markers_y - 1) * marker_spacing + 2 * marker_spacing
    
    # Create board image (white background)
    board_img = np.ones((board_height, board_width), dtype=np.uint8) * 255
    
    # Generate and place markers
    marker_id = 0
    for y in range(markers_y):
        for x in range(markers_x):
            # Generate marker
            marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)
            
            # Calculate position
            pos_x = x * (marker_size + marker_spacing) + marker_spacing
            pos_y = y * (marker_size + marker_spacing) + marker_spacing
            
            # Place marker on board
            board_img[pos_y:pos_y+marker_size, pos_x:pos_x+marker_size] = marker_img
            
            # Add marker ID (small text below marker)
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = f"{marker_id}"
            text_size = cv2.getTextSize(text, font, 0.4, 1)[0]
            text_x = pos_x + (marker_size - text_size[0]) // 2
            text_y = pos_y + marker_size + 15
            cv2.putText(board_img, text, (text_x, text_y), font, 0.4, 0, 1)
            
            marker_id += 1
    
    # Add board info text
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"ArUco Board: {markers_x}x{markers_y}, Dict: {dictionary_name}"
    cv2.putText(board_img, text, (marker_spacing, board_height - marker_spacing//2), 
               font, 0.6, 0, 1)
    
    # Save board image
    cv2.imwrite(save_path, board_img)
    print(f"ArUco board saved to {save_path}")
    
    # Return used marker IDs
    return marker_id

def print_calibration_params(calibration_data):
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
    fov = mono['fov_horizontal']
    print(f"\nHorizontal Field of View: {fov:.2f} degrees")
    
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
    calib_file = os.path.join(output_dir, 'aruco_calibration.pkl')
    with open(calib_file, 'wb') as f:
        pickle.dump(calibration_data, f)
    
    # Save mono parameters separately
    mono_file = os.path.join(output_dir, 'aruco_mono_params.pkl')
    with open(mono_file, 'wb') as f:
        pickle.dump(calibration_data['mono'], f)
    
    # Save stereo parameters if available
    if 'stereo' in calibration_data:
        stereo_file = os.path.join(output_dir, 'aruco_stereo_params.pkl')
        with open(stereo_file, 'wb') as f:
            pickle.dump(calibration_data['stereo'], f)
    
    # Save parameters in readable text format
    txt_file = os.path.join(output_dir, 'aruco_calibration_parameters.txt')
    with open(txt_file, 'w') as f:
        # Redirect print output to file
        import sys
        original_stdout = sys.stdout
        sys.stdout = f
        print_calibration_params(calibration_data)
        sys.stdout = original_stdout
    
    print(f"Calibration data saved to {output_dir}")
    print(f"  Full calibration: {calib_file}")
    print(f"  Mono parameters: {mono_file}")
    if 'stereo' in calibration_data:
        print(f"  Stereo parameters: {stereo_file}")
    print(f"  Text summary: {txt_file}")

def main():
    parser = argparse.ArgumentParser(description='GoPro HERO11 Black Camera Calibration with ArUco Markers')
    
    # Input arguments
    parser.add_argument('--images', help='Path to calibration images')
    parser.add_argument('--marker_size', type=float, default=50.0, 
                       help='ArUco marker size in mm')
    parser.add_argument('--dictionary', default='DICT_6X6_250', 
                       help='ArUco dictionary to use')
    parser.add_argument('--baseline', type=float, help='Camera baseline in mm (if known)')
    parser.add_argument('--output_dir', default='./calibration_results', 
                       help='Output directory for calibration files')
    parser.add_argument('--visualize', action='store_true', 
                       help='Visualize detected markers')
    
    # Arguments for generating markers
    parser.add_argument('--generate_marker', action='store_true', 
                       help='Generate a single ArUco marker for printing')
    parser.add_argument('--generate_board', action='store_true', 
                       help='Generate an ArUco board for printing')
    parser.add_argument('--marker_id', type=int, default=0, 
                       help='ID of marker to generate')
    parser.add_argument('--marker_output', default='aruco_marker.png', 
                       help='Output file for generated marker')
    parser.add_argument('--markers_x', type=int, default=4, 
                       help='Number of markers in X direction for board')
    parser.add_argument('--markers_y', type=int, default=5, 
                       help='Number of markers in Y direction for board')
    parser.add_argument('--board_output', default='aruco_board.png', 
                       help='Output file for generated board')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Execute requested action
    if args.generate_marker:
        # Generate a single ArUco marker
        marker_path = os.path.join(args.output_dir, args.marker_output)
        generate_aruco_marker(
            args.dictionary, args.marker_id, 400, marker_path
        )
        print(f"To use this marker for calibration:")
        print(f"1. Print the marker at a known size (e.g., 50mm x 50mm)")
        print(f"2. Ensure it's printed on a rigid, flat surface")
        print(f"3. Take 20-30 photos of the marker from different angles")
        print(f"4. Run the calibration with --marker_size set to the physical size in mm")
        
    elif args.generate_board:
        # Generate an ArUco board
        board_path = os.path.join(args.output_dir, args.board_output)
        num_markers = generate_aruco_board(
            args.dictionary, args.markers_x, args.markers_y, 
            100, 20, board_path  # Default marker size and spacing in pixels
        )
        print(f"Generated ArUco board with {num_markers} markers")
        print(f"To use this board for calibration:")
        print(f"1. Print the board at a known size")
        print(f"2. Ensure it's printed on a rigid, flat surface")
        print(f"3. Take 20-30 photos of the board from different angles")
        print(f"4. Run the calibration with --marker_size set to the physical marker size in mm")
        
    elif args.images:
        # Perform camera calibration with ArUco markers
        print(f"Calibrating camera using ArUco markers from {args.images}")
        print(f"Marker size: {args.marker_size} mm")
        print(f"Dictionary: {args.dictionary}")
        
        calibration_data = calibrate_with_aruco(
            args.images, args.marker_size, args.dictionary, 
            args.baseline, args.visualize
        )
        
        if calibration_data:
            # Save calibration data
            save_calibration(calibration_data, args.output_dir)
            
            # Print parameters
            print_calibration_params(calibration_data)
        else:
            print("Calibration failed. Please check the images and markers.")
            print("Tips for successful calibration:")
            print("1. Use a rigid, flat surface for the markers")
            print("2. Take images from different angles and distances")
            print("3. Ensure good lighting with minimal glare")
            print("4. Use the correct dictionary and marker size")
            print("5. Try using a calibration board with multiple markers")
            
    else:
        # No action specified, show help
        parser.print_help()
        print("\nExamples:")
        print("  # Generate an ArUco marker")
        print(f"  python {__file__} --generate_marker --marker_id 0 --output_dir outputs")
        print()
        print("  # Generate an ArUco board")
        print(f"  python {__file__} --generate_board --markers_x 5 --markers_y 7 --output_dir outputs")
        print()
        print("  # Calibrate using ArUco markers")
        print(f"  python {__file__} --images data/calibration/ --marker_size 50.0 --output_dir outputs --visualize")

if __name__ == "__main__":
    main()