import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def detect_chessboard_advanced(image, pattern_size=(7, 7), show_steps=False):
    """
    Advanced chessboard detection for challenging lighting conditions.
    
    Args:
        image: Input image (numpy array) or path to image file
        pattern_size: Chessboard pattern inner corners (width, height)
        show_steps: Whether to display intermediate processing steps
    
    Returns:
        Tuple of (success, corners, visualization_image)
    """
    # Read the image if a path is provided
    if isinstance(image, str):
        original = cv2.imread(image)
        if original is None:
            print(f"Could not read image from {image}")
            return False, None, None
    else:
        original = image.copy()
    
    # Create a copy for visualization
    img = original.copy()
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Display original grayscale
    if show_steps:
        plt.figure(figsize=(10, 8))
        plt.imshow(gray, cmap='gray')
        plt.title("Original Grayscale Image")
        plt.axis('off')
        plt.show()
    
    # Try standard method first
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    
    # If standard method succeeds, refine corners
    if ret:
        print("Standard chessboard detection successful")
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        # Draw the corners
        cv2.drawChessboardCorners(img, pattern_size, corners2, ret)
        return True, corners2, img
    
    print("Standard detection failed. Trying enhanced methods...")
    
    # Method 1: Adaptive thresholding
    print("Trying adaptive thresholding...")
    
    # Apply bilateral filter to smooth the image while preserving edges
    blurred = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Apply adaptive thresholding to get a binary image
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # Display threshold result
    if show_steps:
        plt.figure(figsize=(10, 8))
        plt.imshow(thresh, cmap='gray')
        plt.title("Adaptive Threshold")
        plt.axis('off')
        plt.show()
    
    # Invert if needed (depending on which color is predominant)
    # Check which is more common: black or white pixels
    if np.sum(thresh == 0) < np.sum(thresh == 255):
        thresh = cv2.bitwise_not(thresh)
        if show_steps:
            plt.figure(figsize=(10, 8))
            plt.imshow(thresh, cmap='gray')
            plt.title("Inverted Threshold")
            plt.axis('off')
            plt.show()
    
    # Apply morphological operations to clean up noise
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    if show_steps:
        plt.figure(figsize=(10, 8))
        plt.imshow(thresh, cmap='gray')
        plt.title("After Morphological Operations")
        plt.axis('off')
        plt.show()
    
    # Try to find the chessboard corners on the adaptive threshold image
    ret, corners = cv2.findChessboardCorners(thresh, pattern_size, None)
    
    if ret:
        print("Adaptive threshold method successful")
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        # Draw the corners
        cv2.drawChessboardCorners(img, pattern_size, corners2, ret)
        return True, corners2, img
    
    # Method 2: CLAHE (Contrast Limited Adaptive Histogram Equalization)
    print("Trying CLAHE enhancement...")
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    if show_steps:
        plt.figure(figsize=(10, 8))
        plt.imshow(enhanced, cmap='gray')
        plt.title("CLAHE Enhanced")
        plt.axis('off')
        plt.show()
    
    # Try with enhanced image
    ret, corners = cv2.findChessboardCorners(enhanced, pattern_size, 
                                           flags=cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                                 cv2.CALIB_CB_NORMALIZE_IMAGE +
                                                 cv2.CALIB_CB_FAST_CHECK)
    
    if ret:
        print("CLAHE enhancement method successful")
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(enhanced, corners, (11, 11), (-1, -1), criteria)
        
        # Draw the corners
        cv2.drawChessboardCorners(img, pattern_size, corners2, ret)
        return True, corners2, img
    
    # Method 3: Edge detection
    print("Trying edge detection method...")
    
    # Use Canny edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    if show_steps:
        plt.figure(figsize=(10, 8))
        plt.imshow(edges, cmap='gray')
        plt.title("Canny Edges")
        plt.axis('off')
        plt.show()
    
    # Dilate edges to connect nearby edges
    dilated_edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    
    # Try with edge image
    ret, corners = cv2.findChessboardCorners(dilated_edges, pattern_size, None)
    
    if ret:
        print("Edge detection method successful")
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        # Draw the corners
        cv2.drawChessboardCorners(img, pattern_size, corners2, ret)
        return True, corners2, img
    
    # Method 4: Combination of techniques
    print("Trying combination method...")
    
    # Enhance contrast
    alpha = 1.5  # Contrast control
    beta = 10    # Brightness control
    contrast_enhanced = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
    
    # Apply adaptive threshold
    threshold_enhanced = cv2.adaptiveThreshold(
        contrast_enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    if show_steps:
        plt.figure(figsize=(10, 8))
        plt.imshow(threshold_enhanced, cmap='gray')
        plt.title("Contrast Enhanced + Threshold")
        plt.axis('off')
        plt.show()
    
    # Try with combined enhancement
    ret, corners = cv2.findChessboardCorners(threshold_enhanced, pattern_size, 
                                           flags=cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                                 cv2.CALIB_CB_NORMALIZE_IMAGE)
    
    if ret:
        print("Combination method successful")
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        # Draw the corners
        cv2.drawChessboardCorners(img, pattern_size, corners2, ret)
        return True, corners2, img
    
    # If all methods fail, suggest alternative pattern sizes to try
    print("All detection methods failed.")
    print("Trying alternative pattern sizes...")
    
    # Try different pattern sizes
    alternative_sizes = [
        (8, 6), (6, 8), (6, 6), (8, 8),
        (7, 6), (6, 7), (7, 8), (8, 7)
    ]
    
    for alt_size in alternative_sizes:
        if alt_size == pattern_size:
            continue
        
        print(f"Trying pattern size {alt_size}...")
        ret, corners = cv2.findChessboardCorners(gray, alt_size, None)
        
        if ret:
            print(f"Detection successful with pattern size {alt_size}")
            print(f"Consider using {alt_size} for calibration instead of {pattern_size}")
            
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # Draw the corners
            alt_img = original.copy()
            cv2.drawChessboardCorners(alt_img, alt_size, corners2, ret)
            
            if show_steps:
                plt.figure(figsize=(10, 8))
                plt.imshow(cv2.cvtColor(alt_img, cv2.COLOR_BGR2RGB))
                plt.title(f"Detected with alternative size: {alt_size}")
                plt.axis('off')
                plt.show()
            
            break
    
    print("Chessboard detection failed with all methods.")
    return False, None, None

def detect_aruco_markers(image, dictionary_name='DICT_6X6_250', show_steps=False):
    """
    Detect ArUco markers in an image.
    
    Args:
        image: Input image (numpy array) or path to image file
        dictionary_name: ArUco dictionary to use
        show_steps: Whether to display intermediate processing steps
    
    Returns:
        Tuple of (corners, ids, visualization_image)
    """
    # Read the image if a path is provided
    if isinstance(image, str):
        original = cv2.imread(image)
        if original is None:
            print(f"Could not read image from {image}")
            return None, None, None
    else:
        original = image.copy()
    
    # Create a copy for visualization
    img = original.copy()
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
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
        return None, None, None
    
    # Set up ArUco detector
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[dictionary_name])
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    
    # Standard detection
    corners, ids, rejected = detector.detectMarkers(gray)
    
    if ids is not None and len(ids) > 0:
        print(f"Detected {len(ids)} ArUco markers with standard method")
    else:
        print("Standard detection failed. Trying enhanced methods...")
        
        # Try with contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        if show_steps:
            plt.figure(figsize=(10, 8))
            plt.imshow(enhanced, cmap='gray')
            plt.title("CLAHE Enhanced")
            plt.axis('off')
            plt.show()
        
        corners, ids, rejected = detector.detectMarkers(enhanced)
        
        if ids is not None and len(ids) > 0:
            print(f"Detected {len(ids)} ArUco markers with CLAHE enhancement")
        else:
            # Try with adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            if show_steps:
                plt.figure(figsize=(10, 8))
                plt.imshow(thresh, cmap='gray')
                plt.title("Adaptive Threshold")
                plt.axis('off')
                plt.show()
            
            corners, ids, rejected = detector.detectMarkers(thresh)
            
            if ids is not None and len(ids) > 0:
                print(f"Detected {len(ids)} ArUco markers with adaptive thresholding")
            else:
                print("All detection methods failed")
                return None, None, None
    
    # Draw detected markers for visualization
    vis_img = img.copy()
    vis_img = cv2.aruco.drawDetectedMarkers(vis_img, corners, ids)
    
    if show_steps:
        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
        plt.title("Detected ArUco Markers")
        plt.axis('off')
        plt.show()
    
    return corners, ids, vis_img

def main():
    parser = argparse.ArgumentParser(description='Advanced Calibration Pattern Detection')
    
    parser.add_argument('--image', required=True, help='Path to image file')
    parser.add_argument('--pattern', choices=['chessboard', 'aruco'], default='chessboard',
                      help='Type of calibration pattern')
    parser.add_argument('--pattern_size', default='7x7', help='Pattern size (width x height)')
    parser.add_argument('--dictionary', default='DICT_6X6_250', help='ArUco dictionary')
    parser.add_argument('--output_dir', default='.', help='Output directory')
    parser.add_argument('--visualize', action='store_true', help='Visualize intermediate steps')
    
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.isfile(args.image):
        print(f"Error: Image file not found: {args.image}")
        return
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Parse pattern size for chessboard
    if args.pattern == 'chessboard':
        pattern_width, pattern_height = map(int, args.pattern_size.split('x'))
        pattern_size = (pattern_width, pattern_height)
        
        # Detect chessboard
        print(f"Detecting {pattern_size[0]}x{pattern_size[1]} chessboard pattern...")
        success, corners, vis_img = detect_chessboard_advanced(
            args.image, pattern_size, args.visualize)
        
        if success:
            # Save visualization image
            output_path = os.path.join(args.output_dir, 'detected_chessboard.jpg')
            cv2.imwrite(output_path, vis_img)
            print(f"Chessboard detection successful. Result saved to {output_path}")
            
            # Display result
            plt.figure(figsize=(10, 8))
            plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
            plt.title("Detected Chessboard")
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        else:
            print("\nDetection suggestions:")
            print("1. Check that the entire chessboard is visible with clear edges")
            print("2. Ensure good lighting with minimal glare")
            print("3. Take the photo straight-on to minimize perspective distortion")
            print("4. Make sure the chessboard has clearly visible contrasting squares")
            print("5. Try a different pattern size (e.g., 8x6 or 6x6)")
            print("6. Consider using an ArUco marker pattern instead")
    
    else:  # ArUco pattern
        # Detect ArUco markers
        print(f"Detecting ArUco markers with dictionary {args.dictionary}...")
        corners, ids, vis_img = detect_aruco_markers(
            args.image, args.dictionary, args.visualize)
        
        if corners is not None and ids is not None:
            # Save visualization image
            output_path = os.path.join(args.output_dir, 'detected_aruco.jpg')
            cv2.imwrite(output_path, vis_img)
            print(f"ArUco detection successful. Result saved to {output_path}")
            
            # Display result
            plt.figure(figsize=(10, 8))
            plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
            plt.title("Detected ArUco Markers")
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        else:
            print("\nDetection suggestions:")
            print("1. Check that the ArUco markers are visible with clear edges")
            print("2. Ensure good lighting with minimal glare")
            print("3. Make sure the markers have good contrast")
            print("4. Try a different ArUco dictionary")
            print("5. Make sure you're using the correct dictionary for your markers")

if __name__ == "__main__":
    main()