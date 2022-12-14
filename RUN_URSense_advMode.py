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
import cv2
import open3d as o3d
import sys
import json

# CONSTANTS
TCP_HOST_IP = "192.168.65.81" # IP adress of PC
TCP_HOST_PORT = 53002 # Port to listen on (non-privileged ports are > 1023)n

N_OF_BURST_FRAMES = 1 # Integer, MUST BE ODD
MAX_DEPTH = 0.5 # Max depth of ptCloud in meters
MAX_WIDTH = 0.1 # Max width of ptCloud in meters (only during RS_burst_find_closest)
N_CLOSEST_POINTS = 51 # How many closest points to pick from, MUST BE ODD (RS_burst_find_closest implementation 2)
N_NEIGHBOR_POINTS = 50 # How many points required in neighborhood (RS_burst_find_closest implementation 3)
NEIGHBORHOOD_BOX_SIZE = 0.010 # Length of cube edge (RS_burst_find_closest implementation 3)

RECORDING_PATH = "./URSense_data/"
RECORDING_FILENAME = "rec_0002.bag"
JSON_CONFIG_PATH = './d435_shortRange_preset.json'

DS5_product_ids = ["0AD1", "0AD2", "0AD3", "0AD4", "0AD5", "0AF6", "0AFE", "0AFF", "0B00", "0B01", "0B03", "0B07", "0B3A", "0B5C"]

### FUNCTIONS ############################################################################################################
def find_device_that_supports_advanced_mode():
    ctx = rs.context()
    devices = ctx.query_devices()
    for dev in devices:
        if dev.supports(rs.camera_info.product_id) and str(dev.get_info(rs.camera_info.product_id)) in DS5_product_ids:
            if dev.supports(rs.camera_info.name):
                print("Found device that supports advanced mode:", dev.get_info(rs.camera_info.name))
            return dev
    raise Exception("No D400 product line device that supports advanced mode was found")

def start_pipeline(pipeline, config):
    # Start streaming from camera to pipeline
    try:
        pipe_profile = pipeline.start(config)

        # -- ADVANCED MODE ----------------------------------------------------------------------------------------
        dev = find_device_that_supports_advanced_mode()
        advnc_mode = rs.rs400_advanced_mode(dev)
        print("Advanced mode is", "enabled" if advnc_mode.is_enabled() else "disabled")

        # Loop until we successfully enable advanced mode
        while not advnc_mode.is_enabled():
            print("Trying to enable advanced mode...")
            advnc_mode.toggle_advanced_mode(True)
            # At this point the device will disconnect and re-connect.
            print("Sleeping for 5 seconds...")
            time.sleep(5)
            # The 'dev' object will become invalid and we need to initialize it again
            dev = find_device_that_supports_advanced_mode()
            advnc_mode = rs.rs400_advanced_mode(dev)
            print("Advanced mode is", "enabled" if advnc_mode.is_enabled() else "disabled")

        # Load json configuration
        f = open(JSON_CONFIG_PATH)
        as_json_object = json.load(f)
        json_string = str(as_json_object).replace("'", '\"')
        advnc_mode.load_json(json_string)
        # --------------------------------------------------------------------------------------------------------

        # Throw away first 10 frames
        tmp = pipeline.wait_for_frames()
        while tmp.frame_number <= 10:
            tmp = pipeline.wait_for_frames()
        print('INFO: Streaming...')
    except:
        print('ERROR: Could not start pipeline')
        return 0.0, 0.0, 0.0

def grab_ptCloud_from_frame(pipeline):
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
    y_bins = np.linspace(y_min, y_max, 4)

    # Get histogram array
    hist, xEdges, yEdges = np.histogram2d(x, y, bins=[x_bins, y_bins])
    return hist, xEdges, yEdges

def RS_burst_find_closest(pipeline, config, n_of_frames):
    # Get xyz of peak which is closest to center of image, in camera frame

    # Start streaming from camera to pipeline
    start_pipeline(pipeline,config)
    
    try:
        #Init array of frame peaks
        tops = np.zeros((n_of_frames,3))
        # Grab n_of_frames frames
        for frame_idx in range(n_of_frames):
            ptCloud = grab_ptCloud_from_frame(pipeline)

            # Remove points more than 50 mm from origin laterally
            ptCloud = ptCloud[abs(ptCloud[:,0]) < MAX_WIDTH/2]
            
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
            # Get list of closest points
            minIdxs = ptCloud[:,2].argsort()

            box_d = NEIGHBORHOOD_BOX_SIZE/2
            # test_i = 0
            for i in minIdxs:
                point_candidate = ptCloud[i,:]
                # Count points inside neighborhood
                n_points = 0
                for j in range(np.size(ptCloud[:,0])):
                    con1 = (ptCloud[j,0] < (point_candidate[0] + box_d)) and (ptCloud[j,0] > (point_candidate[0] - box_d)) # x within range
                    con2 = (ptCloud[j,1] < (point_candidate[1] + box_d)) and (ptCloud[j,1] > (point_candidate[1] - box_d)) # y within range
                    con3 = (ptCloud[j,2] < (point_candidate[2] + box_d)) and (ptCloud[j,2] > (point_candidate[2] - box_d)) # z within range
                    if con1 and con2 and con3:
                        n_points = n_points + 1
                # print(n_points,test_i)
                # test_i = test_i + 1
                # If there is N_NEIGHBOR_POINTS around point
                if n_points >= N_NEIGHBOR_POINTS:
                    # Remember the point
                    point = point_candidate
                    tops[frame_idx,:] = point
                    break
            # ----------------------------------------------------------------------------------------------------------------------

            # -- VISUALISATION -----------------------------------------------------------------------------------------------------
            # # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize.
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(ptCloud) # Full ptCloud | ptCloud_vis
            # # pcd_area = o3d.geometry.PointCloud()
            # # pcd_area.points = o3d.utility.Vector3dVector(points) # Valid points (implementation 2)
            # pcd_point = o3d.geometry.TriangleMesh()
            # pcd_point = pcd_point.create_sphere(0.002)
            # pcd_point = pcd_point.translate(point, relative=False) # Selected point

            # # Add color for better visualization.
            # pcd.paint_uniform_color([0.5, 0.5, 0.5])
            # # pcd_area.paint_uniform_color([1, 0, 0])
            # pcd_point.paint_uniform_color([1, 1, 0])

            # #Show
            # print('Showing frame')
            # o3d.visualization.draw([pcd, pcd_point])
            # ----------------------------------------------------------------------------------------------------------------------
            
    finally:
        pipeline.stop()
        print('INFO: End of stream')

        # Get median closest point
        med_x = np.median(tops[:,0])
        med_y = np.median(tops[:,1])
        med_z = np.median(tops[:,2])

    # Return the point in camera's frame
    return med_x, med_y, med_z

def RS_burst(pipeline, config, n_of_frames):
    # Get xyz of peak which is closest to center of image, in camera frame

    # Start streaming from camera to pipeline
    start_pipeline(pipeline, config)
    
    try:
        #Init array of frame peaks
        peaks_of_frames = np.zeros((n_of_frames,3))
        # Grab n_of_frames frames
        for frame_idx in range(n_of_frames):
            ptCloud = grab_ptCloud_from_frame(pipeline)

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
            _, img = cv.threshold(hist,127,255,cv.THRESH_TOZERO)

            # Find local peaks
            detected_peaks = detect_peaks(img)

            # Get peak-pixel's idxs
            peak_idxs = np.where(detected_peaks)

            # Return if no pixels were found
            if not peak_idxs:
                return 0.0, 0.0, 0.0

            # Display
            # plt.subplot(1,3,1)
            # plt.title('Histogram')
            # plt.imshow(hist)
            # plt.subplot(1,3,2)
            # plt.title('Treshold')
            # plt.imshow(img)
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
                con2 = (ptCloud[:,2] > peak_yRange[0]) & (ptCloud[:,2] < peak_yRange[1])
                peak_points = ptCloud[con1 & con2]

                # Get median height
                peakY = np.median(peak_points[:,1])

                # Get peak's xz (lateral coordinates)
                peakX = sum(peak_xRange)/2
                peakZ = sum(peak_yRange)/2

                # Store peak in array
                peaks[i,:] = [peakX, peakY, peakZ]
                i = i + 1
            
            # Select peak with smallest abs(x) = in the center
            idxMin = np.argmin(abs(peaks[:,0]))
            frame_peak = peaks[idxMin,:]

            #Put this peak into array and then average its location across all frames
            peaks_of_frames[frame_idx,:] = frame_peak
            
    finally:
        pipeline.stop()
        print('INFO: End of stream')

        # Get median peak
        med_x = np.median(peaks_of_frames[:,0])
        med_y = np.median(peaks_of_frames[:,1])
        med_z = np.median(peaks_of_frames[:,2])

        # Return the peak in camera's frame
        return med_x, med_y, med_z



### MAIN ############################################################################################################
def main():
    # Init the camera
    pipeline = rs.pipeline()
    config = rs.config()

    # Enable depth and RGB -> resolution, data_format, FPS
    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30) # 848x480 for d435, 640x480 for l515
    config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30) # 848x480 for d435, 640x480 for l515
    # Enable recording to file
    config.enable_record_to_file(RECORDING_PATH + RECORDING_FILENAME)
    # Make sure destination exists
    if not os.path.exists(RECORDING_PATH):
        os.mkdir(RECORDING_PATH)

    # Choose alternate video source (TESTING)
    # config.enable_device_from_file('./URSense_data/rec_0001.bag')

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
                    while True:
                        data = conn.recv(1024)
                        # print(data)
                        # Decypher command
                        if data == b'trigBurst\n':
                            print("INFO: Received command \'trigBurst\'")
                            # Get xyz of peak which is closest to center of image, in camera frame
                            x,y,z = RS_burst(pipeline, config, N_OF_BURST_FRAMES)
                            # Reply with the peak position
                            reply_string = '(' + str(x) + ',' + str(y) + ',' + str(z) + ')'
                            print("INFO: Sending reply \'" + reply_string + "\'")
                            conn.sendall(bytes(reply_string,'utf-8'))
                        elif data == b'trigBurstClosest\n':
                            print("INFO: Received command \'trigBurstClosest\'")
                            # Get xyz of point which is closest to camera by z, in camera frame
                            x,y,z = RS_burst_find_closest(pipeline, config, N_OF_BURST_FRAMES)
                            # Reply with the peak position
                            reply_string = '(' + str(x) + ',' + str(y) + ',' + str(z) + ')'
                            print("INFO: Sending reply \'" + reply_string + "\'")
                            conn.sendall(bytes(reply_string,'utf-8'))
                        elif data == b'':
                            # This happens after disconnecting
                            break
                        else:
                            print('WARNING: Unknown command')
                            time.sleep(1)
    except KeyboardInterrupt:
        print("\nINFO: Exiting...\n")
    except:
        print("ERROR: Closing socket")
        s.close()

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