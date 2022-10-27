import numpy as np

peaks_of_frames = list()
peaks1 = np.asanyarray([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
peaks2 = peaks1 * 2
peaks3 = peaks2* 3
peaks4 = peaks3 * 4
peaks5 = peaks4 * 5

#Put this peak into array and then average its location across all frames
peaks_of_frames.append(peaks1)
peaks_of_frames.append(peaks2)
peaks_of_frames.append(peaks3)
peaks_of_frames.append(peaks4)
peaks_of_frames.append(peaks5)

# --- FILTER EDGE CASES -----------------------------------------------------------------------------
# Adds 2d arrays to 3d arrays so that the smaller 1st dimention (=a[THIS,:,:]) is kept in case of inequality. Decides
# which to shave off based on "x" (=a[:,THIS,:]) proximity.

# If peaks_of_frames empty or n_of_peaks equal...
if peaks_of_frames or (np.shape(peaks3)[0] == np.shape(peaks_of_frames)[0]):
    # Do nothing
    pass
# If found more peaks than established
elif np.shape(peaks3)[0] > np.shape(peaks_of_frames)[0]:
    # Compare x values to find if extra on left or right
    dLeft = abs(peaks3[0,0] - peaks_of_frames[0,0,-1])
    dRight = abs(peaks3[-1,0] - peaks_of_frames[-1,0,-1])
    # If left is further
    if dLeft > dRight:
        # Remove 1st (=left) element from new peaks
        peaks3 = peaks3[1:len(peaks3[:,0]),:]
    else:
        # Remove last (=right) element from new peaks
        peaks3 = peaks3[0:-1,:]
# If found fewer peaks than established
else:
    # Compare x values to find if extra on left or right
    dLeft = abs(peaks3[0,0] - peaks_of_frames[0,0,-1])
    dRight = abs(peaks3[-1,0] - peaks_of_frames[-1,0,-1])
    # If left is further
    if dLeft > dRight:
        # Remove 1st (=left) element from all old peaks
        peaks_of_frames = peaks_of_frames[1:len(peaks_of_frames[:,0,0]),:,:]
    else:
        # Remove last (=right) element from all old peaks
        peaks_of_frames = peaks_of_frames[0:-1,:,:]

# TODO: Actually add the new frame