# License: Apache 2.0. See LICENSE file in root directory.
# Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

"""
OpenCV and Numpy Point cloud Software Renderer

This sample is mostly for demonstration and educational purposes.
It really doesn't offer the quality or performance that can be
achieved with hardware acceleration.

Usage:
------
Mouse: 
    Drag with left button to rotate around pivot (thick small axes), 
    with right button to translate and the wheel to zoom.

Keyboard: 
    [p]     Pause
    [r]     Reset View
    [d]     Cycle through decimation values
    [z]     Toggle point scaling
    [c]     Toggle color source
    [s]     Save PNG (./out.png)
    [e]     Export points to ply (./out.ply)
    [q\ESC] Quit
"""

from statistics import mean
import numpy as np
import pyrealsense2 as rs
import open3d as o3d
import cv2 as cv
import time
import matplotlib.pyplot as plt
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from mpl_toolkits import mplot3d

## FUNCTIONS #################################################################################################################
def detect_peaks(image):
    """
    Takes an image and detect the peaks usingthe local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """

    # define an 8-connected neighborhood
    neighborhood = generate_binary_structure(2,2)

    #apply the local maximum filter; all pixel of maximal value 
    #in their neighborhood are set to 1
    local_max = maximum_filter(image, footprint=neighborhood)==image
    #local_max is a mask that contains the peaks we are 
    #looking for, but also the background.
    #In order to isolate the peaks we must remove the background from the mask.

    #we create the mask of the background
    background = (image==0)

    #a little technicality: we must erode the background in order to 
    #successfully subtract it form local_max, otherwise a line will 
    #appear along the background border (artifact of the local maximum filter)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

    #we obtain the final mask, containing only peaks, 
    #by removing the background from the local_max mask (xor operation)
    detected_peaks = local_max ^ eroded_background

    return detected_peaks

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, rs.format.z16, 30)
config.enable_stream(rs.stream.color, rs.format.bgr8, 30)

config.enable_device_from_file('./data_1_9_2022/rec1.bag')

# Start streaming
pipeline.start(config)

# Get stream profile and camera intrinsics
# profile = pipeline.get_active_profile()
# depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
# depth_intrinsics = depth_profile.get_intrinsics()
# w, h = depth_intrinsics.width, depth_intrinsics.height

# Processing blocks
pc = rs.pointcloud()
colorizer = rs.colorizer()
pcd = o3d.geometry.PointCloud()

# Grab all frames
try:
    i = 0
    firstStream = True
    previous_frame_n = 0
    firstLoop = True
    while firstStream:
        # Grab camera data
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        # Grab new intrinsics (may be changed by decimation)
        # depth_intrinsics = rs.video_stream_profile(
        #     depth_frame.profile).get_intrinsics()

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        depth_colormap = np.asanyarray(
            colorizer.colorize(depth_frame).get_data())

        points = pc.calculate(depth_frame)

        # Pointcloud data to arrays
        v, t = points.get_vertices(), points.get_texture_coordinates()
        verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz
        texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)  # uv
        print("Reading frame",frames.frame_number)
        if firstLoop:
            vert_list = np.empty((verts.shape[0],verts.shape[1],1),np.float32)
            vert_list[:,:,0] = verts
            vert_frame = np.empty((verts.shape[0],verts.shape[1],1),np.float32)
            firstLoop = False
        else:
            # vert_list[:,:,i] = verts
            vert_frame[:,:,0] = verts
            # Iterative redefining slows things down, better to replace with big initial matrix, then cut down to size
            vert_list = np.append(vert_list, vert_frame, 2)

        # Update loop variables
        i = i+1
        if frames.frame_number - previous_frame_n < 0 and i > 5:
            firstStream = False
        previous_frame_n = frames.frame_number
        # End firstStream early
        if i > 65:
            firstStream = False

finally:
    # Stop streaming
    pipeline.stop()

startIdx = 50
for j in range(startIdx, startIdx + i):

    # Grab a frame
    ptCloud = vert_list[:,:,j]

    # Remove zero elements
    # ptCloud = np.ma.masked_equal(ptCloud,0)
    con1 = ptCloud[:,0] != 0
    con2 = ptCloud[:,1] != 0
    con3 = ptCloud[:,2] != 0
    ptCloud = ptCloud[con1 & con2 & con3]

    # Remove points in distance
    z_limit = 1.0
    ptCloud = ptCloud[ptCloud[:,2] < z_limit]

    # fig = plt.figure()
    # ax = plt.axes(projection='3d')

    # ax.scatter3D(ptCloud[:,0], ptCloud[:,2], (-1)*ptCloud[:,1], c=ptCloud[:,2], cmap='Greens')
    # # ax.auto_scale_xyz(ptCloud[:,0], ptCloud[:,1], ptCloud[:,2])
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    # # ax.autoscale_view()

    # plt.show()

    # Make histogram of XZ (Z = depth, Y = height)
    x = ptCloud[:,0] # ptCloud's x
    y = ptCloud[:,2] # ptCloud's z

    x_min = np.min(x)
    x_max = np.max(x)
    
    y_min = np.min(y)
    y_max = np.max(y)
    
    x_bins = np.linspace(x_min, x_max, 25)
    y_bins = np.linspace(y_min, y_max, 4)

    # Get histogram array
    # plt.plot(x, y,"ob")
    hist, xEdges, yEdges = np.histogram2d(x, y, bins=[x_bins, y_bins])
    plt.subplot(1,3,1)
    plt.title('Histogram')
    plt.imshow(hist)

    # Convert float64 -> uint8
    maxVal = np.max(np.max(hist))
    hist = np.round_(hist * 255 / maxVal)
    hist = hist.astype('uint8')

    # Treshold image with TRIANGLE method for determining cutoff
    # mask = cv.adaptiveThreshold(hist,255,cv.ADAPTIVE_THRESH_MEAN_C,\
    #             cv.THRESH_BINARY,3,2)
    # img = cv.bitwise_and(hist, mask)
    _, img = cv.threshold(hist,cv.THRESH_TRIANGLE,255,cv.THRESH_TOZERO)

    plt.subplot(1,3,2)
    plt.title('Treshold')
    plt.imshow(img)

    # Find local peaks
    detected_peaks = detect_peaks(img)
    plt.subplot(1,3,3)
    plt.title('Peaks')
    plt.imshow(detected_peaks)

    plt.show()

    # Get peak-pixel's idxs
    peak_idxs = np.where(detected_peaks)

    # For every peak
    for idx_x, idx_y in zip(peak_idxs[0], peak_idxs[1]):
        # Find every point in pixel's region
        peak_xRange = xEdges[idx_x:idx_x+2]
        peak_yRange = yEdges[idx_y:idx_y+2]

        # Remember histogram y is ptCloud z
        con1 = (ptCloud[:,0] > peak_xRange[0]) & (ptCloud[:,0] < peak_xRange[1])
        con2 = (ptCloud[:,2] > peak_yRange[0]) & (ptCloud[:,2] < peak_yRange[1])
        peak_points = ptCloud[con1 & con2]

        # Get average height
        peakY = np.mean(peak_points[:,1])

        # Get peak's xz (lateral coordinates)
        peakX = sum(peak_xRange)/2
        peakZ = sum(peak_yRange)/2

        print("Peak coordinates: ",peakX,peakY,peakZ)
    





    # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize.
    # pcd.points = o3d.utility.Vector3dVector(ptCloud)

    # Add color and estimate normals for better visualization.
    # pcd.paint_uniform_color([0.5, 0.5, 0.5])

    # SLOW, FOR VISUALISATION
    # pcd.estimate_normals()

    #Show
    # print('Showing frame',j)
    # o3d.visualization.draw([pcd])