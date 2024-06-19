import os
import binvox_rw
import numpy as np


# Convert the binary voxel files in a specified directory into a .npy file
def convertBinvoxToNumpyArray(directory):
    for file in os.listdir(directory):
        if file.endswith('.binvox'):
            filename = os.path.join(directory, file)
            with open(filename, 'rb') as f:
                model = binvox_rw.read_as_3d_array(f)
            np_filename = os.path.splitext(file)[0] + '.npy'
            np_filepath = os.path.join(directory, np_filename)
            np.save(np_filepath, model.data)

