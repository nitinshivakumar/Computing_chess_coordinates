# Camera Calibration 

### Preliminary:
1. Define the projection matrix M and solve for intrinsic parameters (fx, fy, ox, oy) and extrinsic parameters (R, T) from world coordinates to image coordinates.
2. Implement functions to:
   - Obtain image coordinates of 32 corners from the image.
   - Calculate world coordinates of the 32 corners.
   - Determine intrinsic parameters (fx, fy, ox, oy) from image and world coordinates.
   - Extract extrinsic parameters (R, T) from image and world coordinates.

### Implementation Notes:
- Use OpenCV version 4.5.4 for this project.
- Code is being implemented in the "UB Geometry.py" file without modifying other files.
- Ensure correct output formats for "result task1.json" and "result task2.json".

