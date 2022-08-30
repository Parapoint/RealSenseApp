# Libraries
import pyrealsense2 as rs
import numpy as np
import cv2
#import os.path
import time
import socket

# GLOBALS ######################################################################################################################################
TCP_HOST_IP = "192.168.65.81"
TCP_HOST_PORT = 53002

#TODO:  GUI and callback(event) architecture eventually
#       Add sourceing from other folders

# AUX FUNCTIONS ################################################################################################################################

def checkInput(input, default):
    # Checks Y/n input and returns True/False
    if input == "Y" or input == "y":
        return True
    elif input == "N" or input == "n":
        return False
    else:
        print("WARNING: Input false")
        return default

class depthCam:
    # Init func
    def __init__(self):
        try:
            # Init variables
            self.recording_enable = False
            self.recording_destination = '.'
            self.recording_fileName = 'RealSense_stream_data'
            self.recording_fileNumber = 0
            self.stream_duration = 10.0 #seconds
            self.tcp_trigger = False
            self.local_source = False
            self.source_destination = '.'
            self.source_fileName = 'RealSense_stream_data'
            self.display_enable = True

            # Set default configuration
            self.pipeline = rs.pipeline()
            self.config = rs.config()

            # Enable depth and RGB -> resolution, data_format, FPS
            self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        except:
            print("ERROR: Failed to init depthCam")


    # Get input from user to set configuration
    def queryConfig(self):
        print("SETTINGS:")
        # Get pipeline source
        local_source = input("Source from recording? (Y/n)")
        # Check input
        self.local_source = checkInput(local_source,self.local_source)
        # If streaming from local source...
        if self.local_source:
            # self.source_destination = input()
            self.source_fileName = input("Enter file name: ")

        # Get trigger mode
        manual_trigger = input("Trigger manually? (Y/n): ")
        # Check input
        tmp = checkInput(manual_trigger,self.tcp_trigger)
        self.tcp_trigger = not tmp

        # Get stream duration
        self.stream_duration = float(input("Enter stream duration in seconds: "))

        # Get display on/off
        display_enable = input("Enable display? (Y/n): ")
        # Check input
        self.display_enable = checkInput(display_enable,self.display_enable)

        # If not sourcing from file
        if not self.local_source:
            # Get user input
            record = input("Record to file? (Y/n): ")
            # Check input
            self.recording_enable = checkInput(record,self.recording_enable)
        
        if self.recording_enable:
            # Get file location UNIMPLEMENTED
            # location = input("Enter file destination (default=root): ")
            # if location == '':
            #     self.recording_destination = '.'
            # elif os.path.exists(location):
            #     self.recording_destination = location
            # else:
            #     print("WARNING: Input false")
            #     return
            
            # Get file name
            self.recording_fileName = input("Enter file name: ")


    # Set configuration
    def setConfig(self):
        # Setup recording
        if self.recording_enable:
            self.config.enable_record_to_file(self.recording_fileName + '.bag')
        
        # Setup source
        if self.local_source:
            self.config.enable_device_from_file(self.source_fileName + '.bag')


    # Start TCP server / software trigger mode
    def TCP_mode(self):
        HOST = TCP_HOST_IP # Standard loopback interface address (localhost)
        PORT = TCP_HOST_PORT  # Port to listen on (non-privileged ports are > 1023)n

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((HOST, PORT))
            s.listen()
            print('Listening on ' + str(TCP_HOST_IP) + ':' + str(TCP_HOST_PORT))
            conn, addr = s.accept()
            with conn:
                print(f"Connected by {addr}")
                while True:
                    data = conn.recv(1024)
                    if data == b'trig\n':
                        self.startStreaming()
                    else:
                        print('WARNING: Unknown command, disconnecting...')
                        break


    # Start streaming video from pipeline
    def startStreaming(self):
        # Start stream
        try:
            self.pipeline.start(self.config)
            print('Streaming...')
        except:
            print('ERROR: Could not start pipeline')
            return
        
        try:
            streamStart = time.time()
            while time.time() - streamStart < self.stream_duration:

                # Wait for a coherent pair of frames: depth and color
                frames = self.pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                if not depth_frame or not color_frame:
                    continue

                # Convert images to numpy arrays
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

                depth_colormap_dim = depth_colormap.shape
                color_colormap_dim = color_image.shape

                # If depth and color resolutions are different, resize color image to match depth image for display
                if depth_colormap_dim != color_colormap_dim:
                    resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
                    images = np.hstack((resized_color_image, depth_colormap))
                else:
                    images = np.hstack((color_image, depth_colormap))

                if self.display_enable:
                    # Show images
                    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
                    cv2.imshow('RealSense', images)
                    cv2.waitKey(1)

        finally:
            # Stop streaming
            self.pipeline.stop()
            # If recording enabled update fileName
            if self.recording_enable:
                self.recording_fileNumber = self.recording_fileNumber + 1
                self.config.enable_record_to_file(self.recording_fileName + '_' + str(self.recording_fileNumber) + '.bag')

# CALLBACKS #########################################################################################################################################
def testCallback(s):
    print('Length of the text files is: ', s)

# Main function
def main():
    # Init the camera object
    myCam = depthCam()
    # Setup configuration
    myCam.queryConfig()
    myCam.setConfig()
    try:
        while True:
            # If software trigger
            if myCam.tcp_trigger:
                myCam.TCP_mode() #start tcp server which waits for trigger
            else:
                # Manual trigger
                input("Press \"Enter\" to start streaming\n")
                myCam.startStreaming()
    except KeyboardInterrupt:
        print("\nExiting...\n")

# Entry point
if __name__ == "__main__":
    print("\nStarting Realsense.py...")
    main()

else:
    print("Warning: Executing as imported file")
    pass