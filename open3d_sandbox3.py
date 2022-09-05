import open3d as o3d
import numpy as np
import pyrealsense2 as rs
import time
import matplotlib.pyplot as plt

RECORDING_PATH = "./data_1_9_2022/rec1.bag"

# bag_reader = o3d.t.io.RSBagReader()
# bag_reader.open(RECORDING_PATH)
# while not bag_reader.is_eof():
#     rgbd_im = bag_reader.next_frame()
#     pcd = rgbd_im.depth
#     # pcd = o3d.t.io.read_image(rgbd_im.depth_path)
#     # color = o3d.t.io.read_image(rgbd_im.color_path)
#     # process im_rgbd.depth and im_rgbd.color
#     # Flip it, otherwise the pointcloud will be upside down.
#     pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
#     print(pcd)
#     axis_aligned_bounding_box = pcd.get_axis_aligned_bounding_box()
#     axis_aligned_bounding_box.color = (1, 0, 0)
#     oriented_bounding_box = pcd.get_oriented_bounding_box()
#     oriented_bounding_box.color = (0, 1, 0)
#     print(
#         "Displaying axis_aligned_bounding_box in red and oriented bounding box in green ..."
#     )
#     o3d.visualization.draw(
#         [pcd, axis_aligned_bounding_box, oriented_bounding_box])

# bag_reader.close()



# Set default configuration
pipeline = rs.pipeline()
config = rs.config()

# Enable depth and RGB -> resolution, data_format, FPS
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# # Set source
# config.enable_device_from_file(RECORDING_PATH)

# Start stream
try:
    pipeline.start(config)
    print('Streaming...')
except:
    print('ERROR: Could not start pipeline')

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize.
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz = np.random.rand(100, 3))
        # Add color and estimate normals for better visualization.
        pcd.paint_uniform_color([0.5, 0.5, 0.5])
        pcd.estimate_normals()
        pcd.orient_normals_consistent_tangent_plane(1)
        print("Displaying Open3D pointcloud made using numpy array ...")
        o3d.visualization.draw([pcd])

        time.sleep(1)

finally:
    # Stop streaming
    pipeline.stop()