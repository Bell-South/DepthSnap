import numpy as np
import cv2
import argparse
import os

def create_chessboard(width, height, square_size, margin=50):
    """Create a high-contrast chessboard calibration pattern image
    
    Args:
        width: Number of squares in width direction
        height: Number of squares in height direction
        square_size: Size of each square in pixels
        margin: Margin around the pattern in pixels
        
    Returns:
        Image of chessboard pattern
    """
    # Calculate image size
    img_width = width * square_size + 2 * margin
    img_height = height * square_size + 2 * margin
    
    # Create white image
    img = np.ones((img_height, img_width), dtype=np.uint8) * 255
    
    # Draw squares
    for i in range(height):
        for j in range(width):
            if (i + j) % 2 == 0:  # Black squares only on even sum of indices
                y1 = i * square_size + margin
                y2 = (i + 1) * square_size + margin
                x1 = j * square_size + margin
                x2 = (j + 1) * square_size + margin
                img[y1:y2, x1:x2] = 0
    
    # Add markers for orientation
    radius = square_size // 8
    
    # Top-left
    cv2.circle(img, (margin // 2, margin // 2), radius, 0, -1)
    
    # Top-right
    cv2.circle(img, (img_width - margin // 2, margin // 2), radius, 0, -1)
    
    # Bottom-left
    cv2.circle(img, (margin // 2, img_height - margin // 2), radius, 0, -1)
    
    # Add text with dimensions
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"{width}x{height} squares, {width-1}x{height-1} internal corners"
    cv2.putText(img, text, (margin, img_height - margin // 4), font, 0.5, 0, 1)
    
    return img

def create_circles_grid(width, height, circle_size, spacing_factor=2.5, margin=50):
    """Create a circles grid calibration pattern
    
    Args:
        width: Number of circles in width direction
        height: Number of circles in height direction
        circle_size: Radius of each circle in pixels
        spacing_factor: Spacing between circles as a factor of circle_size
        margin: Margin around the pattern in pixels
        
    Returns:
        Image of circles grid pattern
    """
    spacing = int(circle_size * spacing_factor)
    
    # Calculate image size
    img_width = (width - 1) * spacing + 2 * margin
    img_height = (height - 1) * spacing + 2 * margin
    
    # Create white image
    img = np.ones((img_height, img_width), dtype=np.uint8) * 255
    
    # Draw circles
    for i in range(height):
        for j in range(width):
            center_x = j * spacing + margin
            center_y = i * spacing + margin
            cv2.circle(img, (center_x, center_y), circle_size, 0, -1)
    
    # Add markers for orientation
    radius = circle_size // 2
    
    # Top-left corner
    cv2.rectangle(img, (margin // 2 - radius, margin // 2 - radius), 
                 (margin // 2 + radius, margin // 2 + radius), 0, -1)
    
    # Bottom-right corner
    cv2.rectangle(img, (img_width - margin // 2 - radius, img_height - margin // 2 - radius), 
                 (img_width - margin // 2 + radius, img_height - margin // 2 + radius), 0, -1)
    
    # Add text with dimensions
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"{width}x{height} circles"
    cv2.putText(img, text, (margin, img_height - margin // 4), font, 0.5, 0, 1)
    
    return img

def create_asymmetric_circles_grid(width, height, circle_size, spacing_factor=2.5, margin=50):
    """Create an asymmetric circles grid pattern where every other row is shifted
    
    Args:
        width: Number of circles in width direction in even rows
        height: Number of rows
        circle_size: Radius of each circle in pixels
        spacing_factor: Spacing between circles as a factor of circle_size
        margin: Margin around the pattern in pixels
        
    Returns:
        Image of asymmetric circles grid pattern
    """
    spacing = int(circle_size * spacing_factor)
    
    # Calculate image size
    img_width = (width - 0.5) * spacing + 2 * margin  # Extra space for shifted rows
    img_height = (height - 1) * spacing + 2 * margin
    
    # Create white image
    img = np.ones((img_height, img_width), dtype=np.uint8) * 255
    
    # Draw circles with shifted rows
    for i in range(height):
        row_shift = spacing // 2 if i % 2 == 1 else 0  # Shift odd rows
        for j in range(width - (1 if i % 2 == 1 else 0)):  # One less circle in odd rows
            center_x = j * spacing + margin + row_shift
            center_y = i * spacing + margin
            cv2.circle(img, (center_x, center_y), circle_size, 0, -1)
    
    # Add markers for orientation
    radius = circle_size // 2
    
    # Top-left corner
    cv2.rectangle(img, (margin // 2 - radius, margin // 2 - radius), 
                 (margin // 2 + radius, margin // 2 + radius), 0, -1)
    
    # Add text with dimensions
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"{width}x{height} asymmetric circles grid"
    cv2.putText(img, text, (margin, img_height - margin // 4), font, 0.5, 0, 1)
    
    return img

def create_charuco_board(squares_x, squares_y, square_length, marker_length_ratio=0.8, margin=50):
    """Create a ChArUco board (chessboard with ArUco markers)
    
    Args:
        squares_x: Number of squares in X direction
        squares_y: Number of squares in Y direction
        square_length: Size of each square in pixels
        marker_length_ratio: Size of ArUco marker relative to square (0-1)
        margin: Margin around the pattern in pixels
        
    Returns:
        Image of ChArUco board
    """
    # Create ArUco dictionary
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    
    # Calculate marker length
    marker_length = int(square_length * marker_length_ratio)
    
    # Create CharUco board
    board = cv2.aruco.CharucoBoard((squares_x, squares_y), square_length, marker_length, dictionary)
    
    # Generate board image
    img_size = (squares_x * square_length + 2 * margin, 
               squares_y * square_length + 2 * margin)
    board_img = board.generateImage(img_size)
    
    # Add text with dimensions
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"ChArUco board {squares_x}x{squares_y}"
    cv2.putText(board_img, text, (margin, img_size[1] - margin // 4), font, 0.5, 0, 1)
    
    return board_img

def print_pattern_instructions(pattern_type, output_path):
    """Print instructions for using the generated pattern"""
    print(f"\nGenerated {pattern_type} pattern saved to: {output_path}")
    print("\nInstructions for use:")
    print("1. Print the pattern on a rigid, flat surface")
    print("2. Ensure the pattern is printed at the correct scale")
    print("3. For accurate calibration, avoid pattern deformation")
    
    if pattern_type == 'chessboard':
        print("\nFor chessboard calibration:")
        print("- Use the internal corners count (width-1 x height-1) in your calibration code")
        print("- Take 20-30 images of the pattern from different angles")
        print("- Ensure the entire pattern is visible in each image")
    elif 'circles' in pattern_type:
        print("\nFor circles grid calibration:")
        print("- Ensure consistent lighting to improve circle detection")
        print("- Avoid reflective surfaces")
    elif pattern_type == 'charuco':
        print("\nFor ChArUco calibration:")
        print("- ChArUco patterns work better in challenging lighting conditions")
        print("- Not all markers need to be visible in each image")
    
    print("\nCalibration quality tips:")
    print("- Cover different parts of the image frame with the pattern")
    print("- Include images with the pattern at different distances")
    print("- Include some images with the pattern at an angle to the camera")

def main():
    parser = argparse.ArgumentParser(description='Generate calibration patterns')
    parser.add_argument('--type', choices=['chessboard', 'circles', 'asymmetric_circles', 'charuco'],
                       default='chessboard', help='Type of pattern to generate')
    parser.add_argument('--width', type=int, default=9, help='Width of pattern (squares/circles)')
    parser.add_argument('--height', type=int, default=6, help='Height of pattern (squares/circles)')
    parser.add_argument('--size', type=int, default=80, help='Size of elements in pixels')
    parser.add_argument('--output', default='calibration_pattern.png', help='Output filename')
    parser.add_argument('--show', action='store_true', help='Display the pattern after generation')
    parser.add_argument('--dpi', type=int, default=300, help='DPI for printing information')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate the appropriate pattern
    if args.type == 'chessboard':
        img = create_chessboard(args.width, args.height, args.size)
        print(f"Generated chessboard pattern with {args.width}x{args.height} squares")
        print(f"This gives {args.width-1}x{args.height-1} internal corners for calibration")
        
        # Calculate physical size at specified DPI
        width_mm = args.width * args.size * 25.4 / args.dpi
        height_mm = args.height * args.size * 25.4 / args.dpi
        print(f"At {args.dpi} DPI, the pattern will be {width_mm:.1f}mm x {height_mm:.1f}mm")
        print(f"Each square will be {args.size * 25.4 / args.dpi:.1f}mm")
        
    elif args.type == 'circles':
        img = create_circles_grid(args.width, args.height, args.size // 4)
        print(f"Generated circles grid pattern with {args.width}x{args.height} circles")
        
        # Calculate physical size at specified DPI
        spacing = int(args.size // 4 * 2.5)
        width_mm = ((args.width - 1) * spacing) * 25.4 / args.dpi
        height_mm = ((args.height - 1) * spacing) * 25.4 / args.dpi
        print(f"At {args.dpi} DPI, the pattern will be {width_mm:.1f}mm x {height_mm:.1f}mm")
        
    elif args.type == 'asymmetric_circles':
        img = create_asymmetric_circles_grid(args.width, args.height, args.size // 4)
        print(f"Generated asymmetric circles grid pattern")
        print(f"Even rows have {args.width} circles, odd rows have {args.width-1} circles")
        
        # Calculate physical size at specified DPI
        spacing = int(args.size // 4 * 2.5)
        width_mm = ((args.width - 0.5) * spacing) * 25.4 / args.dpi
        height_mm = ((args.height - 1) * spacing) * 25.4 / args.dpi
        print(f"At {args.dpi} DPI, the pattern will be {width_mm:.1f}mm x {height_mm:.1f}mm")
        
    elif args.type == 'charuco':
        img = create_charuco_board(args.width, args.height, args.size)
        print(f"Generated ChArUco board with {args.width}x{args.height} squares")
        
        # Calculate physical size at specified DPI
        width_mm = args.width * args.size * 25.4 / args.dpi
        height_mm = args.height * args.size * 25.4 / args.dpi
        print(f"At {args.dpi} DPI, the pattern will be {width_mm:.1f}mm x {height_mm:.1f}mm")
        print(f"Each square will be {args.size * 25.4 / args.dpi:.1f}mm")
    
    # Save the image
    cv2.imwrite(args.output, img)
    
    # Print instructions
    print_pattern_instructions(args.type, args.output)
    
    # Display the image if requested
    if args.show:
        # Resize for display if too large
        max_display_dim = 1200
        h, w = img.shape[:2]
        scale = min(1.0, max_display_dim / max(h, w))
        
        if scale < 1.0:
            display_size = (int(w * scale), int(h * scale))
            display_img = cv2.resize(img, display_size)
        else:
            display_img = img
            
        cv2.imshow("Calibration Pattern", display_img)
        print("Press any key to close the window")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()