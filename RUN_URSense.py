# Script to run with UR5e and realsense camera attached
# Finds positions of plant-lowering loops from stereo depth images, communicates via TCP/IP
# 
# Author: Ales Rucigaj, ales.rucigaj@fe.uni-lj.si

# Libraries
import pyrealsense2 as rs
import numpy as np
import cv2 as cv
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from scipy.ndimage.filters import maximum_filter
import time
import socket
import os
import matplotlib.pyplot as plt
import open3d as o3d
import sys
import re

# CONSTANTS
TCP_HOST_IP = "192.168.65.122" # IP adress of PC
TCP_HOST_PORT = 53002 # Port to listen on (non-privileged ports are > 1023)n

N_OF_BURST_FRAMES = 11 # Integer, MUST BE ODD
MAX_DEPTH = 0.5 # Max depth of ptCloud in meters
MAX_WIDTH = 0.1 # Max width of ptCloud in meters (only during RS_burst_find_closest)
N_NEIGHBOR_POINTS = 25 # How many points required in neighborhood (RS_burst_find_closest implementation 3 and 5)
LATITUDE_BUFFER = 0.010 # Leniency of width-histogram cutoff
MIN_CLOUD_SIZE = 8000 # Minimum size of pointcloud after closeup and trimming
N_CLOSEST_POINTS = 51 # How many closest points to pick from, MUST BE ODD (RS_burst_find_closest implementation 2)
NEIGHBORHOOD_BOX_SIZE = 0.010 # Length of cube edge (RS_burst_find_closest implementation 3)

FROM_RECORDING = False # Streams frames from recording if True
RECORD_VIDEO = False # Turns on recording, incompatible with FROM_RECORDING
RECORDING_PATH = "./URSense_data/"
RECORDING_FILENAME = "rec_0020.bag"

### FUNCTIONS ############################################################################################################
def start_pipeline(pipeline, config, fromRecording):
    # Start streaming from camera to pipeline
    try:
        tic = time.time()

        pipe_profile = pipeline.start(config)
        # --- Configure device preset ------------------------------------------------------------------------------------
        if not fromRecording:
            depth_sensor = pipe_profile.get_device().first_depth_sensor()

            # List presets
            # preset_range = depth_sensor.get_option_range(rs.option.visual_preset)
            # for i in range(int(preset_range.max)+2):
            #     visulpreset = depth_sensor.get_option_value_description(rs.option.visual_preset,i)
            #     print(i,visulpreset)

            depth_sensor.set_option(rs.option.visual_preset, 5)
            print("INFO: Setting device preset")
        # ----------------------------------------------------------------------------------------------------------------

        # Throw away first 10 frames
        tmp = pipeline.wait_for_frames()
        while tmp.frame_number <= 10:
            tmp = pipeline.wait_for_frames()
        print('INFO: Streaming...')

        toc = time.time() - tic
        print("INFO: Starting pipeline lasted: %.3f" % (toc) + " seconds")
        return pipeline

    except Exception as e:
        print('ERROR: Could not start pipeline')
        print(e)
        sys.exit()

def grab_ptCloud_from_frame(pipeline):
    tic = time.time()

    pc = rs.pointcloud()
    frame = pipeline.wait_for_frames()
    # print(frame.frame_number)

    # Isolate depth frame
    depth_frame = frame.get_depth_frame()
    # Get ptCloud from depth
    points_object = pc.calculate(depth_frame)
    # Convert to numpy array
    v = points_object.get_vertices()
    ptCloud = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz

    # Display frame
    # depth_image = np.asanyarray(depth_frame.get_data())
    # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    # cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    # cv2.imshow('RealSense', depth_colormap)
    # cv2.waitKey(1)

    # Remove zero elements
    con1 = ptCloud[:,0] != 0
    con2 = ptCloud[:,1] != 0
    con3 = ptCloud[:,2] != 0
    ptCloud = ptCloud[con1 & con2 & con3]

    # Remove points in distance
    ptCloud = ptCloud[ptCloud[:,2] < MAX_DEPTH]

    toc = time.time() - tic
    print("INFO: Grabbing frame lasted: %.3f" % (toc) + " seconds")

    return ptCloud

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

def get_2D_hist(x,y):
    # Generate 2D histogram. NOTE: xyEdges is 1 element longer than hist
    x_min = np.min(x)
    x_max = np.max(x)
    
    y_min = np.min(y)
    y_max = np.max(y)
    
    x_bins = np.linspace(x_min, x_max, 25)
    y_bins = np.linspace(y_min, y_max, 2) # Only 1 hist bin, as multiple objects at the same latitude are not accounted for

    # Get histogram array
    hist, xEdges, yEdges = np.histogram2d(x, y, bins=[x_bins, y_bins])
    return hist, xEdges, yEdges

def zyxEul_to_rotMat(eulPose):
    # Returns transformation matrix assuming input is (x,y,z,xEul,yEul,zEul) for zyxEul=R(z)R(y)R(x)
    xEul = eulPose[3]
    yEul = eulPose[4]
    zEul = eulPose[5]

    T = [[np.cos(zEul)*np.cos(yEul),np.cos(zEul)*np.sin(yEul)*np.sin(xEul)-np.cos(xEul)*np.sin(zEul),np.sin(zEul)*np.sin(xEul)+np.cos(zEul)*np.cos(xEul)*np.sin(yEul),eulPose[0]],
         [np.cos(yEul)*np.sin(zEul),np.cos(zEul)*np.cos(xEul)+np.sin(zEul)*np.sin(yEul)*np.sin(xEul),np.cos(xEul)*np.sin(zEul)*np.sin(yEul)-np.cos(zEul)*np.sin(xEul),eulPose[1]],
         [-np.sin(yEul),np.cos(yEul)*np.sin(xEul),np.cos(yEul)*np.cos(xEul),eulPose[2]],
         [0,0,0,1]]

    return np.asanyarray(T)

def pointCloud_changeFrame(pointCloud, Trans_BK):
    # Transforms points in pointCloud from camera frame to horizontal camera frame, using base z as reference
    newPointCloud = np.zeros(np.shape(pointCloud))

    # Get alpha between -z(base) and y(camera)
    alpha = -np.arccos( -(Trans_BK[2,1]) )

    # Get angular functions
    ca = np.cos(alpha)
    sa = np.sin(alpha)

    # Transform                  
    newPointCloud[:,0] = pointCloud[:,0]
    newPointCloud[:,1] = pointCloud[:,1]*ca - pointCloud[:,2]*sa
    newPointCloud[:,2] = pointCloud[:,1]*sa + pointCloud[:,2]*ca

    return newPointCloud

def pointCloud_revertFrame(pointCloud, Trans_BK):
    # Transforms points in pointCloud back from horizontal camera frame to camera frame, using base z as reference
    newPointCloud = np.zeros(np.shape(pointCloud))

    # Get minus alpha between -z(base) and y(camera)
    alpha = np.arccos( -(Trans_BK[2,1]) )

    # Get angular functions
    ca = np.cos(alpha)
    sa = np.sin(alpha)

    # Transform
    newPointCloud[:,0] = pointCloud[:,0]
    newPointCloud[:,1] = pointCloud[:,1]*ca - pointCloud[:,2]*sa
    newPointCloud[:,2] = pointCloud[:,1]*sa + pointCloud[:,2]*ca

    return newPointCloud

def merge_peaksOfFrames(peaks_of_frames, peaks):
    # Adds 2d array to 3d array so that the smaller 1st dimention (=a[THIS,:,:]) is kept in case of inequality. Decides
    # which to shave off based on "x" (=a[:,1,:]) proximity.
    # Only works with a mismatch of 1
    # WARNING: Does not work if array values are substantially different

    try:
        # --- FILTER EDGE CASES ---
        # If n_of_peaks equal...
        if  np.shape(peaks)[0] == np.shape(peaks_of_frames)[0]:
            # Do nothing
            pass
        # If peaks_of_frames empty
        elif not peaks_of_frames.any():
            # Set peaks as peaks_of_frames
            peaksShape = np.shape(peaks) + (1,)
            tmp = np.zeros(peaksShape)
            tmp[:,:,0] = peaks
            return tmp
        # If found more peaks than established
        elif np.shape(peaks)[0] > np.shape(peaks_of_frames)[0]:
            # Compare x values to find if extra on left or right
            dLeft = abs(peaks[0,0] - peaks_of_frames[0,0,-1])
            dRight = abs(peaks[-1,0] - peaks_of_frames[-1,0,-1])
            # If left is further
            if dLeft > dRight:
                # Remove 1st (=left) element from new peaks
                peaks = peaks[1:len(peaks[:,0]),:]
            else:
                # Remove last (=right) element from new peaks
                peaks = peaks[0:-1,:]
        # If found fewer peaks than established
        else:
            # Compare x values to find if extra on left or right
            dLeft = abs(peaks[0,0] - peaks_of_frames[0,0,-1])
            dRight = abs(peaks[-1,0] - peaks_of_frames[-1,0,-1])
            # If left is further
            if dLeft > dRight:
                # Remove 1st (=left) element from all old peaks
                peaks_of_frames = peaks_of_frames[1:len(peaks_of_frames[:,0,0]),:,:]
            else:
                # Remove last (=right) element from all old peaks
                peaks_of_frames = peaks_of_frames[0:-1,:,:]
        # ---------------------

        # --- ADD NEW FRAME ---
        peaksShape = np.shape(peaks) + (1,)
        tmp = np.zeros(peaksShape)
        tmp[:,:,0] = peaks
        peaks_of_frames = np.dstack((peaks_of_frames,tmp))
        # ---------------------

    except Exception as e:
        print("ERROR: Could not merge arrays")
        print(e)
    
    return peaks_of_frames

### CLASSDEF ################################################################################################################
class vision:
    def __init__(self):
        self.tilted_camera = False # Flag for non-horizontal camera
        self.T_BK_eul = np.asanyarray([0,0,0,0,0,0]) # Camera to Base frame transform in zyx Euler
        self.peakArray = np.zeros((0,0)) # Array of detected approximate peak locations

    def getNextPeak(self):
        # Returns the right-most peak (last in array) and flags it as served by setting all values to -99.
        # Also returns lastPeakFlag: 1 if last else 0

        # Check if peakArray has elements
        if not self.peakArray.any():
            print("WARNING: Peak array is empty")
            return 0.0,0.0,0.0,1

        # Check if peak has already been served
        self.peakArray = self.peakArray[self.peakArray[:,0] != -99.0]

        # Check again if peakArray has elements
        if not self.peakArray.any():
            print("WARNING: Peak array is empty")
            return 0.0,0.0,0.0,1

        # Serve the last peak in array
        x = self.peakArray[-1,0]
        y = self.peakArray[-1,1]
        z = self.peakArray[-1,2]

        # Flag the peak as served
        self.peakArray[-1,0] = -99.0

        # Check if this was the last peak
        if self.peakArray[0,0] == -99.0:
            lastPeakFlag = 1
        else:
            lastPeakFlag = 0

        return x,y,z,lastPeakFlag

    def RS_burst_find_closest(self, pipeline, n_of_frames):
        # Get xyz of peak which is closest to center of image, in camera frame
        
        try:
            #Init array of frame peaks
            tops = np.zeros((n_of_frames,3))
            # Grab n_of_frames frames
            for frame_idx in range(n_of_frames):
                ptCloud = grab_ptCloud_from_frame(pipeline)

                tic = time.time()

                # Remove points more than 50 mm from origin laterally
                ptCloud = ptCloud[abs(ptCloud[:,0]) < MAX_WIDTH/2]

                # If camera tilted change frame to horizontal
                if self.tilted_camera:
                    T_BK_eul = np.asanyarray(self.T_BK_eul)
                    T_BK_rotMat = zyxEul_to_rotMat(T_BK_eul)
                    ptCloud = pointCloud_changeFrame(ptCloud, T_BK_rotMat)
                
                # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize.
                # pcd = o3d.geometry.PointCloud()
                # pcd.points = o3d.utility.Vector3dVector(ptCloud) # Full ptCloud | ptCloud_vis
                # #Show
                # print('Showing frame')
                # o3d.visualization.draw(pcd)
                
                # Find point
                point = np.asanyarray([0.0,0.0,0.0])
                # -- IMPLEMENTATION 1 ---------------------------------------------------------------------------------------------------
                # # Select point with smallest z
                # idxMin = np.argmin(ptCloud[:,2])
                # point = ptCloud[idxMin,:]
                # tops[frame_idx,:] = point
                # ----------------------------------------------------------------------------------------------------------------------

                # -- IMPLEMENTATION 2 ---------------------------------------------------------------------------------------------------
                # # Select the median from n closest points (smallest z)
                # minIdxs = ptCloud[:,2].argsort()[:N_CLOSEST_POINTS]
                # pointIdx = int(np.median(minIdxs))
                # point = ptCloud[pointIdx,:]
                # # Remember the point
                # tops[frame_idx,:] = point

                # # Visualisation
                # # Find valid points
                # points = ptCloud[minIdxs,:]
                # # Remove overlapping points
                # ptCloud_vis = np.delete(ptCloud, minIdxs, axis=0)
                # ----------------------------------------------------------------------------------------------------------------------

                # -- IMPLEMENTATION 3 ---------------------------------------------------------------------------------------------------
                # # Get list of closest points
                # minIdxs = ptCloud[:,2].argsort()

                # box_d = NEIGHBORHOOD_BOX_SIZE/2
                # # test_i = 0
                # for i in minIdxs:
                #     point_candidate = ptCloud[i,:]
                #     # Count points inside neighborhood
                #     n_points = 0
                #     for j in range(np.size(ptCloud[:,0])):
                #         con1 = (ptCloud[j,0] < (point_candidate[0] + box_d)) and (ptCloud[j,0] > (point_candidate[0] - box_d)) # x within range
                #         con2 = (ptCloud[j,1] < (point_candidate[1] + box_d)) and (ptCloud[j,1] > (point_candidate[1] - box_d)) # y within range
                #         con3 = (ptCloud[j,2] < (point_candidate[2] + box_d)) and (ptCloud[j,2] > (point_candidate[2] - box_d)) # z within range
                #         if con1 and con2 and con3:
                #             n_points = n_points + 1
                #     # print(n_points,test_i)
                #     # test_i = test_i + 1
                #     # If there is N_NEIGHBOR_POINTS around point
                #     if n_points >= N_NEIGHBOR_POINTS:
                #         # Remember the point
                #         point = point_candidate
                #         tops[frame_idx,:] = point
                #         break
                # ----------------------------------------------------------------------------------------------------------------------

                # -- IMPLEMENTATION 4 ---------------------------------------------------------------------------------------------------
                # # Get list of closest points
                # minIdxs = ptCloud[:,2].argsort()

                # box_d = NEIGHBORHOOD_BOX_SIZE/2
                # # test_i = 0
                # for i in minIdxs:
                #     point_candidate = ptCloud[i,:]
                #     # Count points inside neighborhood
                #     n_points = 0
                #     for j in range(np.size(ptCloud[:,0])):
                #         con1 = (ptCloud[j,0] < (point_candidate[0] + box_d/2)) and (ptCloud[j,0] > (point_candidate[0] - box_d/2)) # x within range
                #         con2 = (ptCloud[j,1] < (point_candidate[1] + box_d*5)) and (ptCloud[j,1] > (point_candidate[1] - box_d*5)) # y within range
                #         con3 = (ptCloud[j,2] < (point_candidate[2] + box_d)) and (ptCloud[j,2] > (point_candidate[2] - box_d)) # z within range
                #         if con1 and con2 and con3:
                #             n_points = n_points + 1
                #     # print(n_points,test_i)
                #     # test_i = test_i + 1
                #     # If there is N_NEIGHBOR_POINTS around point
                #     if n_points >= N_NEIGHBOR_POINTS:
                #         # Remember the point
                #         point = point_candidate
                #         tops[frame_idx,:] = point
                #         break
                # ----------------------------------------------------------------------------------------------------------------------

                # -- IMPLEMENTATION 5 ---------------------------------------------------------------------------------------------------
                # Get histogram along x (lateral)
                x = ptCloud[:,0]
                xBins = np.linspace(-0.14,0.14,641) # Approx FOV width (=0.28 m) across 640 px -> cca 0.5 mm resolution
                hist, xEdges = np.histogram(x, bins=xBins, density=False) #bins='auto'

                # Display
                # plt.title('Histogram')
                # plt.hist(xEdges[:-1], xEdges, weights=hist)
                # plt.show()

                # Trim anything outside 30% max density (minus a little bit)
                hist_norm = hist/max(hist)
                hook_idxs = np.where(hist_norm > 0.3)
                hook_idxs = hook_idxs[0]
                xRange = xEdges[hook_idxs[0]:(hook_idxs[-1]+2)]
                con1 = (ptCloud[:,0] > (min(xRange)-LATITUDE_BUFFER)) & (ptCloud[:,0] < (max(xRange)+LATITUDE_BUFFER))
                ptCloud = ptCloud[con1]

                # Safety measure
                if ptCloud.size < MIN_CLOUD_SIZE:
                    print('WARNING: Point cloud too small, bad vision')
                    return 0.0, 0.0, 0.0

                # Get histogram along z (depth)
                z = ptCloud[:,2]
                zMax = max(z)
                zMin = min(z)
                zRange = zMax - zMin
                zBinNum = int(zRange/0.003) # 3 mm resolution
                hist, zEdges = np.histogram(z, bins=zBinNum, density=False) # bins=30

                # Display
                # plt.title('Histogram')
                # plt.hist(zEdges[:-1], zEdges, weights=hist)
                # plt.show()

                # Pick the 1st point with sufficient neighbour density
                point_idxs = np.where(hist > N_NEIGHBOR_POINTS)
                point_idx = point_idxs[0][0]
                zRange = zEdges[point_idx:point_idx+2]
                print("INFO: Depth bin size: %.1f" % ((max(zRange)-min(zRange))*1000) + " millimeters")
                con1 = (ptCloud[:,2] > min(zRange)) & (ptCloud[:,2] < max(zRange))
                candidates = ptCloud[con1]
                p_x = np.median(candidates[:,0])
                p_y = np.median(candidates[:,1])
                p_z = np.median(candidates[:,2])
                point = [p_x,p_y,p_z]
                tops[frame_idx,:] = point
                # ----------------------------------------------------------------------------------------------------------------------

                # -- VISUALISATION -----------------------------------------------------------------------------------------------------
                # # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize.
                # pcd = o3d.geometry.PointCloud()
                # pcd.points = o3d.utility.Vector3dVector(ptCloud) # Full ptCloud | ptCloud_vis
                # # pcd_area = o3d.geometry.PointCloud()5
                # # pcd_area.points = o3d.utility.Vector3dVector(points) # Valid points (implementation 2)
                # pcd_point = o3d.geometry.TriangleMesh()
                # pcd_point = pcd_point.create_sphere(0.002)
                # pcd_point = pcd_point.translate(point, relative=False) # Selected point

                # # Add color for better visualization.
                # pcd.paint_uniform_color([0.5, 0.5, 0.5])
                # # pcd_area.paint_uniform_color([1, 0, 0])
                # pcd_point.paint_uniform_color([1, 1, 0])
                
                # #Show
                # print('Showing depth frame')
                # o3d.visualization.draw([pcd, pcd_point])
                # ----------------------------------------------------------------------------------------------------------------------

                toc = time.time() - tic
                print("INFO: Picking point computing lasted: %.3f" % (toc) + " seconds")
                
        finally:
            # Get median closest point
            med_x = np.median(tops[:,0])
            med_y = np.median(tops[:,1])
            med_z = np.median(tops[:,2])

            # Adjust point for picking HARDCODED RIGID TRANSFORM
            med_x = med_x - 0.010
            med_y = med_y - 0.000
            med_z = med_z + 0.010

            # Transform back to camera frame
            if self.tilted_camera:
                out = pointCloud_revertFrame(np.asanyarray([[med_x, med_y, med_z]]), T_BK_rotMat)
                med_x = out[0,0]
                med_y = out[0,1]
                med_z = out[0,2]

        # Return the point in camera's frame
        return med_x, med_y, med_z

    def RS_burst(self, pipeline, n_of_frames):
        # Get xyz of peak which is closest to center of image, in camera frame
        
        try:
            #Init array of frame peaks
            peaks_of_frames = np.zeros((n_of_frames,3))
            # Grab n_of_frames frames
            for frame_idx in range(n_of_frames):
                ptCloud = grab_ptCloud_from_frame(pipeline)

                tic = time.time()

                # If camera tilted change frame to horizontal
                if self.tilted_camera:
                    T_BK_eul = np.asanyarray(self.T_BK_eul)
                    T_BK_rotMat = zyxEul_to_rotMat(T_BK_eul)
                    ptCloud = pointCloud_changeFrame(ptCloud, T_BK_rotMat)

                # Make histogram of XZ plane (Z = depth, Y = height)
                x = ptCloud[:,0] # ptCloud's x
                y = ptCloud[:,2] # ptCloud's z
                hist, xEdges, yEdges = get_2D_hist(x,y)

                # Convert float64 -> uint8
                maxVal = np.max(np.max(hist))
                hist = np.round_(hist * 255 / maxVal)
                hist = hist.astype('uint8')

                # Treshold image with TRIANGLE method for determining cutoff
                # _, img = cv.threshold(hist,cv.THRESH_TRIANGLE,255,cv.THRESH_TOZERO)
                _, tresh = cv.threshold(hist,70,255,cv.THRESH_TOZERO)

                # Close image to connect loop chunks (morph operation)
                # kernel = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
                # closeing = cv.morphologyEx(tresh, cv.MORPH_CLOSE, kernel)

                # Find local peaks
                detected_peaks = detect_peaks(tresh)

                # Get peak-pixel's idxs
                peak_idxs = np.where(detected_peaks)

                # Return if no pixels were found
                if not peak_idxs:
                    print("WARNING: No peaks found on frame" + str(frame_idx))
                    return 0.0, 0.0, 0.0

                # # Display
                # plt.subplot(1,3,1)
                # plt.title('Histogram')
                # plt.imshow(hist)
                # plt.subplot(1,3,2)
                # plt.title('Treshold')
                # plt.imshow(tresh)
                # # plt.subplot(1,4,3)
                # # plt.title('Closing')
                # # plt.imshow(closeing)
                # plt.subplot(1,3,3)
                # plt.title('Peaks')
                # plt.imshow(detected_peaks)
                # plt.show()

                # Init array of peaks
                n_of_peaks = np.size(peak_idxs[0])
                peaks = np.empty((n_of_peaks,3))
                # For every peak
                i = 0
                for idx_x, idx_y in zip(peak_idxs[0], peak_idxs[1]):
                    # Find every point in pixel's region
                    peak_xRange = xEdges[idx_x:idx_x+2]
                    peak_yRange = yEdges[idx_y:idx_y+2]

                    # Remember histogram y is ptCloud z
                    con1 = (ptCloud[:,0] > peak_xRange[0]) & (ptCloud[:,0] < peak_xRange[1])
                    # con2 = (ptCloud[:,2] > peak_yRange[0]) & (ptCloud[:,2] < peak_yRange[1])
                    peak_points = ptCloud[con1] # & con2]

                    # Get median height
                    peakY = np.median(peak_points[:,1])
                    # Get peak's xz (lateral coordinates)
                    peakX = sum(peak_xRange)/2
                    peakZ = np.median(peak_points[:,2]) # sum(peak_yRange)/2 | median is better against outliers
                    
                    # Store peak in array
                    peaks[i,:] = [peakX, peakY, peakZ]
                    i = i + 1

                # Select peak with smallest abs(x) = in the center
                idxMin = np.argmin(abs(peaks[:,0]))
                frame_peak = peaks[idxMin,:]

                #Put this peak into array and then average its location across all frames
                peaks_of_frames[frame_idx,:] = frame_peak

        finally:
            # Get median peak
            med_x = np.median(peaks_of_frames[:,0])
            med_y = np.median(peaks_of_frames[:,1])
            med_z = np.median(peaks_of_frames[:,2])

            # Transform back to camera frame
            if self.tilted_camera:
                out = pointCloud_revertFrame(np.asanyarray([[med_x, med_y, med_z]]), T_BK_rotMat)
                med_x = out[0,0]
                med_y = out[0,1]
                med_z = out[0,2]

            toc = time.time() - tic
            print("INFO: Closeup point computing lasted: %.3f" % (toc) + " seconds")

            # Return the peak in camera's frame
            return med_x, med_y, med_z

    def RS_burstMulti(self, pipeline, n_of_frames):
        # Equivalent of RS_burst for multiple simultaneus loop manipulations
        
        try:
            #Init array of frame peaks
            peaks_of_frames = np.zeros((0,3,0))
            # Grab n_of_frames frames
            for frame_idx in range(n_of_frames):
                ptCloud = grab_ptCloud_from_frame(pipeline)

                tic = time.time()

                # If camera tilted change frame to horizontal
                if self.tilted_camera:
                    T_BK_eul = np.asanyarray(self.T_BK_eul)
                    T_BK_rotMat = zyxEul_to_rotMat(T_BK_eul)
                    ptCloud = pointCloud_changeFrame(ptCloud, T_BK_rotMat)

                # Make histogram of XZ plane (Z = depth, Y = height)
                x = ptCloud[:,0] # ptCloud's x
                y = ptCloud[:,2] # ptCloud's z
                hist, xEdges, yEdges = get_2D_hist(x,y)

                # Convert float64 -> uint8
                maxVal = np.max(np.max(hist))
                hist = np.round_(hist * 255 / maxVal)
                hist = hist.astype('uint8')

                # Treshold image with TRIANGLE method for determining cutoff
                # _, img = cv.threshold(hist,cv.THRESH_TRIANGLE,255,cv.THRESH_TOZERO)
                _, tresh = cv.threshold(hist,70,255,cv.THRESH_TOZERO)

                # Close image to connect loop chunks (morph operation)
                # kernel = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
                # closeing = cv.morphologyEx(tresh, cv.MORPH_CLOSE, kernel)

                # Find local peaks
                detected_peaks = detect_peaks(tresh)

                # Get peak-pixel's idxs
                peak_idxs = np.where(detected_peaks)

                # Return if no pixels were found
                if not peak_idxs:
                    print("WARNING: No peaks found on frame" + str(frame_idx))
                    return 0

                # # Display
                # plt.subplot(1,3,1)
                # plt.title('Histogram')
                # plt.imshow(hist)
                # plt.subplot(1,3,2)
                # plt.title('Treshold')
                # plt.imshow(tresh)
                # # plt.subplot(1,4,3)
                # # plt.title('Closing')
                # # plt.imshow(closeing)
                # plt.subplot(1,3,3)
                # plt.title('Peaks')
                # plt.imshow(detected_peaks)
                # plt.show()

                # Init array of peaks
                n_of_peaks = np.size(peak_idxs[0])
                peaks = np.empty((n_of_peaks,3))
                # For every peak
                i = 0
                for idx_x, idx_y in zip(peak_idxs[0], peak_idxs[1]):
                    # Find every point in pixel's region
                    peak_xRange = xEdges[idx_x:idx_x+2]
                    peak_yRange = yEdges[idx_y:idx_y+2]

                    # Remember histogram y is ptCloud z
                    con1 = (ptCloud[:,0] > peak_xRange[0]) & (ptCloud[:,0] < peak_xRange[1])
                    # con2 = (ptCloud[:,2] > peak_yRange[0]) & (ptCloud[:,2] < peak_yRange[1])
                    peak_points = ptCloud[con1] # & con2]

                    # Get median height
                    peakY = np.median(peak_points[:,1])
                    # Get peak's xz (lateral coordinates)
                    peakX = sum(peak_xRange)/2
                    peakZ = np.median(peak_points[:,2]) # sum(peak_yRange)/2 | median is better against outliers
                    
                    # Store peak in array
                    peaks[i,:] = [peakX, peakY, peakZ]
                    i = i + 1

                #Add 2d array of peaks to 3d array
                peaks_of_frames = merge_peaksOfFrames(peaks_of_frames, peaks)

        finally:
            # Get median peaks across frames
            n_of_peaks = np.size(peaks_of_frames[:,0,0])
            self.peakArray = np.zeros((n_of_peaks,3))

            self.peakArray[:,0] = np.median(peaks_of_frames[:,0,:], axis=1)
            self.peakArray[:,1] = np.median(peaks_of_frames[:,1,:], axis=1)
            self.peakArray[:,2] = np.median(peaks_of_frames[:,2,:], axis=1)

            # Transform back to camera frame
            if self.tilted_camera:
                self.peakArray = pointCloud_revertFrame(self.peakArray, T_BK_rotMat)

            toc = time.time() - tic
            print("INFO: Closeup points computing lasted: %.3f" % (toc) + " seconds")



### MAIN ############################################################################################################
def main():
    # Init the camera
    pipeline = rs.pipeline()
    config = rs.config()

    # Enable depth and RGB -> resolution, data_format, FPS
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30) # 848x480 for d435, 640x480 for l515
    # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30) # 848x480 for d435, 640x480 for l515
    # Enable recording to file
    if RECORD_VIDEO:
        config.enable_record_to_file(RECORDING_PATH + RECORDING_FILENAME)
    # Make sure destination exists
    if not os.path.exists(RECORDING_PATH):
        os.mkdir(RECORDING_PATH)

    # Choose alternate video source (TESTING)
    if FROM_RECORDING:
        config.enable_device_from_file(RECORDING_PATH + RECORDING_FILENAME)

    try:
        # Start streaming from camera to pipeline
        pipeline = start_pipeline(pipeline, config, FROM_RECORDING)

        # Start TCP server
        try:
            HOST = TCP_HOST_IP
            PORT = TCP_HOST_PORT

            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind((HOST, PORT))
                while True:
                    s.listen()
                    print('INFO: Listening on ' + str(TCP_HOST_IP) + ':' + str(TCP_HOST_PORT))
                    conn, addr = s.accept()
                    with conn:
                        print(f"INFO: Connected by {addr}")
                        myCam = vision()
                        while True:
                            dataBits = conn.recv(1024)
                            # print(dataBits)
                            # Decode bytes into string
                            data = dataBits.decode('utf-8')

                            # --- Split data into TCP_command / TCP_arg --------------------------------------------------------------------------
                            # Command matcher
                            commandMatch = re.compile("(\w+) ?p\[")

                            # Argument matcher
                            argMatch = re.compile("\[([^\]]+)\]") # Find any characted between [], except ]
                            TCP_arg = argMatch.findall(data)

                            # If found arguments...
                            if TCP_arg:
                                TCP_command = commandMatch.findall(data)
                            else:
                                TCP_command = [data[0:-1]]
                            TCP_command = TCP_command[0]
                            # -------------------------------------------------------------------------------------------------------------------

                            # --- Execute command -----------------------------------------------------------------------------------------------
                            if TCP_command == 'trigBurst':
                                print("INFO: Received command \'trigBurst\'")
                                # Get xyz of peak which is closest to center of image, in camera frame
                                x,y,z = myCam.RS_burst(pipeline, N_OF_BURST_FRAMES)
                                # Reply with the peak position
                                reply_string = '(' + str(x) + ',' + str(y) + ',' + str(z) + ')'
                                # reply_string = '(' + str(0.0) + ',' + str(0.21) + ',' + str(0.25) + ')'
                                print("INFO: Sending reply \'" + reply_string + "\'")
                                conn.sendall(bytes(reply_string,'utf-8'))

                            elif TCP_command == 'trigBurstMulti':
                                print("INFO: Received command \'trigBurstMulti\'")
                                # Get xyz of peak array, in camera frame
                                myCam.RS_burstMulti(pipeline, N_OF_BURST_FRAMES)
                                # Confirm received
                                reply_string = '(' + str(1) + ')'
                                print("INFO: Sending reply \'" + reply_string + "\'")
                                conn.sendall(bytes(reply_string,'utf-8'))

                            elif TCP_command == 'getNextPeak':
                                print("INFO: Received command \'getNextPeak\'")
                                # Get xyz of peak array, in camera frame
                                x,y,z, lastPeakFlag = myCam.getNextPeak()
                                # Reply with the peak position and lastPeakFlag
                                reply_string = '(' + str(x) + ',' + str(y) + ',' + str(z) + ',' + str(lastPeakFlag) + ')'
                                print("INFO: Sending reply \'" + reply_string + "\'")
                                conn.sendall(bytes(reply_string,'utf-8'))

                            elif TCP_command == 'trigBurstClosest':
                                print("INFO: Received command \'trigBurstClosest\'")
                                # Get xyz of point which is closest to camera by z, in camera frame
                                x,y,z = myCam.RS_burst_find_closest(pipeline, N_OF_BURST_FRAMES)
                                # Reply with the peak position
                                reply_string = '(' + str(x) + ',' + str(y) + ',' + str(z) + ')'
                                print("INFO: Sending reply \'" + reply_string + "\'")
                                conn.sendall(bytes(reply_string,'utf-8'))

                            elif TCP_command == 'set_BKframe':
                                # Check argument structure
                                if len(TCP_arg) > 1:
                                    print("WARNING: Incorrect command argument")
                                    break
                                # Get array from csv string
                                floatMatch = re.compile('[+-]?\d+\.?\d*[e]?[-]?\d*')
                                TElements = floatMatch.findall(TCP_arg[0])
                                # Check arg again
                                if len(TElements) != 6:
                                    print("WARNING: Incorrect command argument")
                                    break
                                # Convert to numpy of floats
                                myCam.T_BK_eul = np.asanyarray([float(i) for i in TElements])
                                # Set tilted cam flag
                                myCam.tilted_camera = True
                                # Confirm received
                                reply_string = '(' + str(1) + ')'
                                print("INFO: Sending reply \'" + reply_string + "\'")
                                conn.sendall(bytes(reply_string,'utf-8'))

                            elif dataBits == b'':
                                # This happens after disconnecting
                                break

                            else:
                                print('WARNING: Unknown command')
                                time.sleep(1)
                            # -----------------------------------------------------------------------------------------------------------------

        except KeyboardInterrupt:
            print("\nINFO: Exiting...\n")
        except Exception as e:
            print("ERROR: Closing socket")
            print(e)
            s.close()

    finally:
        pipeline.stop()
        print('INFO: End of stream')



# PROGRAM ENTRY POINT
if __name__ == "__main__":
    print("\nINFO: Starting Realsense.py...")
    # Check if N_CLOSEST_POINTS is odd
    if (N_CLOSEST_POINTS % 2) == 0:
        print("ERROR: N_CLOSEST_POINTS must be odd")
        sys.exit()
    # Check if N_OF_BURST_FRAMES is odd
    if (N_OF_BURST_FRAMES % 2) == 0:
        print("ERROR: N_CLOSEST_POINTS must be odd")
        sys.exit()
    
    # Start
    main()

else:
    print("WARNING: Executing as imported file")
    pass
