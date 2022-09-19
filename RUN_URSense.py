# Libraries
import pyrealsense2 as rs
import numpy as np
import cv2 as cv
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from scipy.ndimage.filters import maximum_filter
import time
import socket

# CONSTANTS
TCP_HOST_IP = "192.168.65.81" # IP adress of PC
TCP_HOST_PORT = 53002 # Port to listen on (non-privileged ports are > 1023)n
N_OF_BURST_FRAMES = 10 # Integer
MAX_DEPTH = 1.0 # Max depth of ptCloud in meters

RECORDING_PATHANDNAME = "./URSense_data/rec_0001.bag"

### FUNCTIONS ############################################################################################################
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


def RS_burst(pipeline, config, n_of_frames):
    # Get xyz of peak which is closest to center of image, in camera frame

    # Start streaming from camera to pipeline
    try:
        pipeline.start(config)
        print('INFO: Streaming...')
    except:
        print('ERROR: Could not start pipeline')
        return 0.0, 0.0, 0.0
    
    try:
        pc = rs.pointcloud()
        #Init array of frame peaks
        peaks_of_frames = np.zeros((n_of_frames,3))
        # Grab n_of_frames frames
        for frame_idx in range(n_of_frames):
            frame = pipeline.wait_for_frames()

            # Isolate depth frame
            depth_frame = frame.get_depth_frame()
            # Get ptCloud from depth
            points_object = pc.calculate(depth_frame)
            # Convert to numpy array
            v = points_object.get_vertices()
            ptCloud = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz

            # Remove zero elements
            con1 = ptCloud[:,0] != 0
            con2 = ptCloud[:,1] != 0
            con3 = ptCloud[:,2] != 0
            ptCloud = ptCloud[con1 & con2 & con3]

            # Remove points in distance
            ptCloud = ptCloud[ptCloud[:,2] < MAX_DEPTH]

            # Make histogram of XZ plane (Z = depth, Y = height)
            x = ptCloud[:,0] # ptCloud's x
            y = ptCloud[:,2] # ptCloud's z
            hist, xEdges, yEdges = get_2D_hist(x,y)

            # Convert float64 -> uint8
            maxVal = np.max(np.max(hist))
            hist = np.round_(hist * 255 / maxVal)
            hist = hist.astype('uint8')

            # Treshold image with TRIANGLE method for determining cutoff
            _, img = cv.threshold(hist,cv.THRESH_TRIANGLE,255,cv.THRESH_TOZERO)

            # Find local peaks
            detected_peaks = detect_peaks(img)

            # Get peak-pixel's idxs
            peak_idxs = np.where(detected_peaks)

            # Return if no pixels were found
            if peak_idxs.size == 0:
                return 0.0, 0.0, 0.0

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

                # Get average height
                peakY = np.mean(peak_points[:,1])

                # Get peak's xz (lateral coordinates)
                peakX = sum(peak_xRange)/2
                peakZ = sum(peak_yRange)/2

                # Store peak in array
                peaks[i,:] = [peakX, peakY, peakZ]
                i = i + 1
            
            # Select peak with smallest x
            idxMin = np.argmin(peaks[:,0])
            frame_peak = peaks[idxMin[0],:]

            #TODO: Put this peak into array and then average its location across all frames
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
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # Enable recording to file
    config.enable_record_to_file(RECORDING_PATHANDNAME)

    try:
        HOST = TCP_HOST_IP
        PORT = TCP_HOST_PORT

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((HOST, PORT))
            s.listen()
            print('INFO: Listening on ' + str(TCP_HOST_IP) + ':' + str(TCP_HOST_PORT))
            conn, addr = s.accept()
            with conn:
                print(f"INFO: Connected by {addr}")
                while True:
                    data = conn.recv(1024)
                    # Decypher command
                    if data == b'trigBurst\n':
                        # Get xyz of peak which is closest to center of image, in camera frame
                        x,y,z = RS_burst(pipeline, config, N_OF_BURST_FRAMES)
                        # Reply with the peak position
                        reply_string = '(' + x + ',' + y + ',' + z + ')'
                        conn.sendall(reply_string)
                    else:
                        print('WARNING: Unknown command')
    except KeyboardInterrupt:
        print("\nINFO: Exiting...\n")
    except:
        print("ERROR: Closing socket")
        s.close()

# PROGRAM ENTRY POINT
if __name__ == "__main__":
    print("\nINFO: Starting Realsense.py...")
    main()

else:
    print("WARNING: Executing as imported file")
    pass