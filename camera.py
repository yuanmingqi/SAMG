import pyrealsense2 as rs
import numpy as np
import cv2

# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Get depth scale (distance per unit in depth image)
        depth_scale = pipeline.get_active_profile().get_device().first_depth_sensor().get_depth_scale()
        
        # Convert depth image to distance in meters
        depth_image_in_meters = depth_image * depth_scale
        print(depth_image_in_meters.shape)

        # Create a window and display the RGB and depth images
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        images = np.hstack((color_image, depth_colormap))

        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)

        # Press esc or 'q' to close the image window
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            cv2.imwrite("./rgb.png", color_image)
            # np.save("./samples/depth_data.npy", depth_image_in_meters)
            break

        # Print depth data for a specific pixel, for example, at the center of the image
        height, width = depth_image.shape
        center_x, center_y = width // 2, height // 2
        distance = depth_image_in_meters[center_y, center_x]
        print(f"Distance at center ({center_x}, {center_y}): {distance:.2f} meters")

finally:
    # Stop streaming
    pipeline.stop()

cv2.destroyAllWindows()
