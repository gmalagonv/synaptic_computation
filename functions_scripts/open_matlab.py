import scipy.io
import h5py
import numpy as np
from timeit import default_timer as timer
###############################################################
start = timer()
# Replace 'file_path' with the path to your MATLAB file.
file_path = '/media/data/nextcloud/analysis/summary/ct20_test.mat' #/home/gerard/analysis/from_matlab/summary/ct20_test.mat'
fields2call = ['synID','x','y', 'timeFromLast']
file_name = '/ct20_test/allEv/'

# Open the MATLAB file using h5py
with h5py.File(file_path, 'r') as file:
    # Access the 'allEv' dataset
    allEv_data = np.array([])#[]

    for idx, name in enumerate(fields2call):

        allEv_dataset = file[file_name + name]

        # Convert the references to strings
        allEv_references = [ref.item() for ref in allEv_dataset]

        carrier =  []
        for ref in allEv_references:
            item = file[ref]
            carrier.append(item[:])
        carrier = np.squeeze(np.array(carrier))

        if allEv_data.size == 0:
           allEv_data = np.append(allEv_data, carrier)

        else:
            allEv_data = np.column_stack((allEv_data, carrier))
        
end = timer()
print(end - start)