import matplotlib.pyplot as plt
import cv2
import numpy as np

# Load the image - replace this path with your actual image path
image_path = "data/test_images/left.jpg"
img = cv2.imread(image_path)

if img is None:
    print(f"Error: Could not read image from {image_path}")
else:
    # Verify image dimensions
    height, width, channels = img.shape
    print(f"Image dimensions: {width}x{height} pixels")
    
    # Convert BGR to RGB for matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Reference point and distance
    ref_x, ref_y = 1333, 1500
    ref_distance = 1.89
    
    # Create figure
    plt.figure(figsize=(12, 10))
    plt.imshow(img_rgb)
    
    # Draw the reference point
    plt.scatter(ref_x, ref_y, color='red', s=200, marker='x', linewidths=4)
    
    # Add a circle around the point to make it more visible
    circle = plt.Circle((ref_x, ref_y), 100, color='red', fill=False, linewidth=3)
    plt.gca().add_patch(circle)
    
    # Add label with distance
    plt.annotate(f"Reference Point\nDistance: {ref_distance:.2f}m", 
                 (ref_x + 120, ref_y - 50), 
                 color='white', 
                 fontsize=16,
                 bbox=dict(facecolor='red', alpha=0.7, boxstyle='round'),
                 weight='bold')
    
    # Add a line showing where the point is in relation to the ground plane
    plt.plot([ref_x, ref_x], [ref_y, height], 'r--', linewidth=2)
    
    plt.title('Input Image with Reference Point (Kalman Filter Calibration)', fontsize=16)
    plt.axis('off')
    
    # Save the figure
    output_path = "reference_point_visualization.jpg"
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"Visualization saved to {output_path}")

# If you want to also show it on a depth map, you would add a second plot
# This part assumes you have a depth map saved as a NumPy array
try:
    depth_map_path = "results/depth_map.npy"
    depth_map = np.load(depth_map_path)
    
    plt.figure(figsize=(12, 10))
    plt.imshow(depth_map, cmap='plasma')
    
    # Draw the reference point
    plt.scatter(ref_x, ref_y, color='white', s=200, marker='x', linewidths=4)
    
    # Add a circle around the point to make it more visible
    circle = plt.Circle((ref_x, ref_y), 100, color='white', fill=False, linewidth=3)
    plt.gca().add_patch(circle)
    
    # Add label with distance
    plt.annotate(f"Reference Point\nDistance: {ref_distance:.2f}m", 
                 (ref_x + 120, ref_y - 50), 
                 color='white', 
                 fontsize=16,
                 bbox=dict(facecolor='black', alpha=0.7, boxstyle='round'),
                 weight='bold')
    
    plt.colorbar(label='Depth (meters)')
    plt.title('Depth Map with Reference Point', fontsize=16)
    plt.axis('off')
    
    # Save the figure
    output_path = "reference_point_depth_visualization.jpg"
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"Depth visualization saved to {output_path}")
except Exception as e:
    print(f"Could not create depth map visualization: {e}")