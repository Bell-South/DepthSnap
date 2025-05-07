# DepthSnap
A comprehensive toolkit for calibrating GoPro cameras and estimating distances using both stereo vision (two cameras) and monocular vision (single camera with MiDaS).

## Features

- **Robust Camera Calibration**: Calibrate GoPro cameras using chessboard patterns or ArUco markers
- **Advanced Detection**: Tools for challenging lighting conditions and camera characteristics
- **Dual Depth Estimation Methods**:
  - Stereo vision for dual-camera setups
  - Monocular depth estimation using MiDaS for single-camera setups
- **Distance Measurement**: Calculate real-world distances to objects
- **Visualization Tools**: Visualize calibration results and depth maps
- **High Accuracy**: Optimized for 95%+ accuracy in distance measurements

## Installation

```bash
# Clone the repository
git clone https://github.com/Bell-South/DepthSnap.git
cd DepthSnap

# Install dependencies
pip install -r requirements.txt
```

## Camera Setup Recommendations

### Stereo Setup
- Mount two GoPro HERO11 Black cameras side by side
- Recommended baseline (distance between cameras): 10-20 cm
- Use a rigid mount to prevent relative movement between cameras
- Ensure cameras are at the same height and perfectly aligned horizontally

### Monocular Setup
- Mount the camera at a known height from the ground (measure precisely)
- Set a fixed tilt angle (typically 10-15°)
- Measure and record the camera height and tilt angle for accurate scaling

## Workflow

### 1. Generate Calibration Pattern

```bash
python utils/pattern_generator.py --type chessboard --width 9 --height 9 --output calibration_pattern.png
```

Print the pattern on a rigid surface. A standard chessboard has 8x8 squares (7x7 inner corners).

### 2. Capture Calibration Images

Take 20-30 photos of the chessboard pattern from different angles and distances. For stereo calibration, take synchronized images from both cameras.

Organize images in folders:
```
data/calibration/
  ├── left01.jpg, left02.jpg, ...
  └── right01.jpg, right02.jpg, ...
```

### 3. Run Camera Calibration

```bash
python calibration/gopro_calibration.py --left_imgs data/calibration/left*.jpg --right_imgs data/calibration/right*.jpg --pattern_size 7x7 --square_size 24.0 --output_dir results
```

This will:
- Process the calibration images
- Calculate camera parameters
- Save calibration data to `results/camera_calibration.pkl`

### 4. Estimate Depth from Images

For stereo depth estimation:
```bash
python depth_estimation/stereo_depth.py --left_img data/test/left.jpg --right_img data/test/right.jpg --calib_file results/camera_calibration.pkl --output_dir results
```

For monocular depth estimation:
```bash
python depth_estimation/mono_depth.py --img data/test/image.jpg --calib_file results/mono_params.pkl --output_dir results
```

### 5. Measure Distances to Objects

```bash
python depth_estimation/distance_measurement.py --image data/test/image.jpg --depth_map results/depth_map.npy --interactive --output_dir results
```

This launches an interactive tool to measure distances by clicking on points in the image.

## Advanced Usage

### Camera Calibration with ArUco Markers

For challenging conditions where chessboard detection is difficult:

```bash
# Generate ArUco markers
python calibration/aruco_calibration.py --generate_marker --marker_id 0 --marker_output aruco_marker.png

# Calibrate using ArUco markers
python calibration/aruco_calibration.py --images data/calibration/aruco --marker_size 50.0 --output_dir results
```

### Advanced Chessboard Detection

For difficult lighting or brown chessboards:

```bash
python utils/advanced_detection.py --image data/calibration/left01.jpg
```

### Default Parameters

If calibration is not possible, generate default parameters:

```bash
python calibration/default_params.py --resolution 5312x3552 --preset wide --output_dir results
```

## Parameter Descriptions

The key parameters for the algorithms are:

### Stereo Vision Parameters
- **Baseline**: Distance between camera centers (mm)
- **Camera Matrix**: Intrinsic parameters (focal length, principal point)
- **Distortion Coefficients**: Lens distortion correction
- **Rotation & Translation**: Relative pose between cameras

### Monocular Vision Parameters
- **Camera Height**: Distance from ground to camera center (m)
- **Tilt Angle**: Camera inclination from horizontal (degrees)
- **Field of View**: Camera viewing angle (degrees)
- **Camera Matrix**: Intrinsic parameters

## Troubleshooting

- If chessboard detection fails, try the advanced detection algorithms
- For challenging lighting conditions, consider using ArUco markers
- Validate your calibration by measuring known distances
- Ensure GoPro settings (FOV, resolution) match between calibration and testing

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.