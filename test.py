import numpy as np

def merge_peaksOfFrames(peaks_of_frames, peaks):
    # Adds 2d array to 3d array so that the smaller 1st dimention (=a[THIS,:,:]) is kept in case of inequality. Decides
    # which to shave off based on "x" (=a[:,THIS,:]) proximity.

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

    # --- ADD NEW FRAME ---
    peaksShape = np.shape(peaks) + (1,)
    tmp = np.zeros(peaksShape)
    tmp[:,:,0] = peaks
    peaks_of_frames = np.dstack((peaks_of_frames,tmp))
    
    return peaks_of_frames

###

peaks_of_frames = np.zeros((0,3,0))
peaks1 = np.asanyarray([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
peaks2 = peaks1 * 2
peaks3 = peaks2* 3
peaks4 = peaks3 * 4
peaks5 = peaks4 * 5

peaks_of_frames = merge_peaksOfFrames(peaks_of_frames, peaks1)
peaks_of_frames = merge_peaksOfFrames(peaks_of_frames, peaks2)
peaks_of_frames = merge_peaksOfFrames(peaks_of_frames, peaks3)
peaks_of_frames = merge_peaksOfFrames(peaks_of_frames, peaks4)
peaks_of_frames = merge_peaksOfFrames(peaks_of_frames, peaks5)

print(peaks_of_frames[:,:,4])
