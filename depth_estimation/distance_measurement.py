import cv2
import numpy as np
import pickle
import argparse
import matplotlib.pyplot as plt
import os
import csv

def measure_distance_from_depth(depth_map, points, calibration_data=None):
    """
    Measure distances to specific points in the depth map.
    
    Args:
        depth_map: Depth map (in meters)
        points: List of (x, y) coordinates
        calibration_data: Calibration data for additional info (optional)
        
    Returns:
        List of (x, y, distance) tuples
    """
    results = []
    
    for x, y in points:
        # Ensure coordinates are within bounds
        if 0 <= y < depth_map.shape[0] and 0 <= x < depth_map.shape[1]:
            # Get depth value at the point
            distance = depth_map[y, x]
            
            # For robustness, consider average in a small region
            region_size = 5
            y_min = max(0, y - region_size // 2)
            y_max = min(depth_map.shape[0], y + region_size // 2 + 1)
            x_min = max(0, x - region_size // 2)
            x_max = min(depth_map.shape[1], x + region_size // 2 + 1)
            
            region = depth_map[y_min:y_max, x_min:x_max]
            
            # Filter out zeros (invalid depth values)
            valid_values = region[region > 0]
            
            if len(valid_values) > 0:
                # Use median for robustness
                robust_distance = np.median(valid_values)
                distance = robust_distance
            
            results.append((x, y, distance))
        else:
            results.append((x, y, -1))  # Invalid point
    
    return results

def measure_distance_interactive(image, depth_map):
    """
    Interactive tool for measuring distances in an image.
    
    Args:
        image: Original image
        depth_map: Depth map (in meters)
        
    Returns:
        List of measured points and distances
    """
    # Create a copy of the image for display
    display_img = image.copy()
    points = []
    
    # Mouse callback function
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Add point
            points.append((x, y))
            
            # Get distance at point
            distance = depth_map[y, x]
            
            # For robustness, consider region average
            region_size = 5
            y_min = max(0, y - region_size // 2)
            y_max = min(depth_map.shape[0], y + region_size // 2 + 1)
            x_min = max(0, x - region_size // 2)
            x_max = min(depth_map.shape[1], x + region_size // 2 + 1)
            
            region = depth_map[y_min:y_max, x_min:x_max]
            valid_values = region[region > 0]
            
            if len(valid_values) > 0:
                distance = np.median(valid_values)
            
            # Draw point and distance on image
            cv2.circle(display_img, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(display_img, f"{distance:.2f}m", (x + 10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Update display
            cv2.imshow('Measure Distance (Click to add points, ESC to finish)', display_img)
    
    # Create window and set callback
    cv2.imshow('Measure Distance (Click to add points, ESC to finish)', display_img)
    cv2.setMouseCallback('Measure Distance (Click to add points, ESC to finish)', mouse_callback)
    
    # Wait for ESC key
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            break
    
    cv2.destroyAllWindows()
    
    # Measure distances for all points
    measurements = measure_distance_from_depth(depth_map, points)
    
    return measurements, display_img

def measure_object_dimensions(depth_map, points, calibration_data=None):
    """
    Measure dimensions of objects in the scene based on selected points.
    
    Args:
        depth_map: Depth map (in meters)
        points: List of (x, y) pairs defining object boundaries
        calibration_data: Calibration data for additional info (optional)
        
    Returns:
        Dictionary with object dimensions
    """
    if len(points) < 2:
        print("Error: At least 2 points needed to measure dimensions")
        return None
    
    # Get 3D coordinates of points
    coordinates_3d = []
    for x, y in points:
        # Get depth
        depth = depth_map[y, x]
        
        # Skip invalid points
        if depth <= 0:
            continue
        
        # Convert to 3D coordinates
        if calibration_data:
            # Use calibration data to get more accurate 3D coordinates
            if 'mono' in calibration_data:
                camera_matrix = calibration_data['mono']['camera_matrix']
            elif 'camera_matrix' in calibration_data:
                camera_matrix = calibration_data['camera_matrix']
            else:
                # Fallback to approximation
                camera_matrix = np.array([
                    [1000.0, 0.0, depth_map.shape[1]/2],
                    [0.0, 1000.0, depth_map.shape[0]/2],
                    [0.0, 0.0, 1.0]
                ])
            
            # Get camera parameters
            fx = camera_matrix[0, 0]
            fy = camera_matrix[1, 1]
            cx = camera_matrix[0, 2]
            cy = camera_matrix[1, 2]
            
            # Convert pixel coordinates to 3D coordinates
            X = (x - cx) * depth / fx
            Y = (y - cy) * depth / fy
            Z = depth
            
            coordinates_3d.append((X, Y, Z))
        else:
            # Simple approximation without calibration
            fx = 1000.0  # Approximate focal length
            cx = depth_map.shape[1] / 2
            cy = depth_map.shape[0] / 2
            
            X = (x - cx) * depth / fx
            Y = (y - cy) * depth / fx
            Z = depth
            
            coordinates_3d.append((X, Y, Z))
    
    # Calculate dimensions
    dimensions = {}
    
    # If we have at least 2 valid points, calculate distances
    if len(coordinates_3d) >= 2:
        # Calculate euclidean distances between all pairs of points
        for i in range(len(coordinates_3d)):
            for j in range(i+1, len(coordinates_3d)):
                p1 = np.array(coordinates_3d[i])
                p2 = np.array(coordinates_3d[j])
                
                # Euclidean distance
                distance = np.linalg.norm(p2 - p1)
                dimensions[f"Distance {i+1}-{j+1}"] = distance
    
    return dimensions

def main():
    parser = argparse.ArgumentParser(description='Distance Measurement from Depth Maps')
    
    # Input arguments
    parser.add_argument('--image', required=True, help='Path to original image')
    parser.add_argument('--depth_map', required=True, help='Path to depth map (.npy file)')
    parser.add_argument('--calib_file', help='Path to calibration file (optional)')
    parser.add_argument('--output_dir', default='./measurement_results', help='Output directory')
    parser.add_argument('--interactive', action='store_true', help='Enable interactive measurement')
    parser.add_argument('--points', help='Comma-separated list of points: x1,y1,x2,y2,...')
    parser.add_argument('--measure_object', action='store_true', help='Measure object dimensions')
    
    args = parser.parse_args()
    
    # Load original image
    image = cv2.imread(args.image)
    
    if image is None:
        print(f"Error: Could not read input image: {args.image}")
        return
    
    # Load depth map
    if args.depth_map.endswith('.npy'):
        depth_map = np.load(args.depth_map)
    else:
        # Try to load as image and convert
        depth_img = cv2.imread(args.depth_map, cv2.IMREAD_UNCHANGED)
        if depth_img is None:
            print(f"Error: Could not read depth map: {args.depth_map}")
            return
        
        # Normalize to 0-40 meters range (assuming 8-bit depth map)
        depth_map = depth_img.astype(float) * 40.0 / 255.0
    
    # Load calibration data if provided
    calibration_data = None
    if args.calib_file:
        try:
            with open(args.calib_file, 'rb') as f:
                calibration_data = pickle.load(f)
        except Exception as e:
            print(f"Warning: Failed to load calibration file: {e}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Measure distances
    if args.interactive:
        measurements, result_img = measure_distance_interactive(image, depth_map)
    elif args.points:
        # Parse points from command line
        coords = list(map(int, args.points.split(',')))
        points = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
        measurements = measure_distance_from_depth(depth_map, points, calibration_data)
        
        # Create visualization
        result_img = image.copy()
        for i, (x, y, distance) in enumerate(measurements):
            cv2.circle(result_img, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(result_img, f"{i+1}: {distance:.2f}m", (x + 10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        print("Error: Either --interactive or --points must be specified")
        return
    
    # Measure object dimensions if requested
    if args.measure_object and len(measurements) >= 2:
        print("\nMeasuring object dimensions...")
        points = [(x, y) for x, y, _ in measurements]
        dimensions = measure_object_dimensions(depth_map, points, calibration_data)
        
        if dimensions:
            # Print dimensions
            print("\nObject Dimensions:")
            for key, value in dimensions.items():
                print(f"  {key}: {value:.2f} meters")
            
            # Draw dimensions on image
            for i, ((p1_idx, p2_idx), distance) in enumerate(zip(
                [(i, j) for i in range(len(points)) for j in range(i+1, len(points))],
                dimensions.values()
            )):
                p1 = points[p1_idx]
                p2 = points[p2_idx]
                
                # Draw line between points
                cv2.line(result_img, p1, p2, (0, 255, 0), 2)
                
                # Calculate midpoint for text
                mid_x = (p1[0] + p2[0]) // 2
                mid_y = (p1[1] + p2[1]) // 2
                
                # Draw distance
                cv2.putText(result_img, f"{distance:.2f}m", (mid_x, mid_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Save results
    cv2.imwrite(os.path.join(args.output_dir, 'measurement_visualization.png'), result_img)
    
    # Print measurements
    print("\nDistance Measurements:")
    for i, (x, y, distance) in enumerate(measurements):
        print(f"Point {i+1} ({x}, {y}): {distance:.2f} meters")
    
    # Save measurements to CSV
    with open(os.path.join(args.output_dir, 'measurements.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Point', 'X', 'Y', 'Distance (m)'])
        for i, (x, y, distance) in enumerate(measurements):
            writer.writerow([i+1, x, y, distance])
    
    print(f"\nResults saved to {args.output_dir}")
    
    # Show result image
    cv2.imshow('Distance Measurements', result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()