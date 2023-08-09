import scipy.io
import h5py
import numpy as np
#from timeit import default_timer as timer
###############################################################
#start = timer()
# Replace 'file_path' with the path to your MATLAB file.
#path = '/media/data/nextcloud/analysis/summary' #/home/gerard/analysis/from_matlab/summary/ct20_test.mat'

def matlab2python(actionFlag, file_name, section ,fields2call, file_path='/home/gerard/nextcloud/analysis/localization/summary'):

    """"
    actionFlag : if == 0: only list the fields, else, load the fields  
    section : possible values: allEv, perSyn
    file_path = '/media/data/nextcloud/analysis/summary/ct20_test.mat' #/home/gerard/analysis/from_matlab/summary/ct20_test.mat'
    fields2call = ['synID','x','y', 'timeFromLast']
    
     """
    file_path = file_path + '/' + file_name + '.mat'
    #file_nameO = '/' + file_name
    file_name = '/' + file_name + '/'+ section+'/'
    # Open the MATLAB file using h5py
    with h5py.File(file_path, 'r') as file:

        if actionFlag == 0:
            #print(list(file[file_name].keys()))
            return list(file[file_name].keys())
        else:
            print('OPENING :',  file_path)


            # Access the 'allEv' dataset
            allEv_data = np.array([])#[]
            idx = 0 
            for name in fields2call:
                allEv_dataset = file[file_name + name]
                # Convert the references to strings
                allEv_references = [ref.item() for ref in allEv_dataset]
                carrier =  []
                for ref in allEv_references:
                    item = file[ref]
                   
                    if (item.shape) != (1,1):
                        print('long field, not included ', name)
                        break
                    carrier.append(item[:])
                carrier = np.squeeze(np.array(carrier))

                

                if allEv_data.size == 0 and carrier.size > 0:
                    allEv_data = np.append(allEv_data, carrier)
                    print(idx, name)
                    idx+=1

                elif allEv_data.size != 0 and carrier.size > 0:
                    allEv_data = np.column_stack((allEv_data, carrier))
                    print(idx, name)
                    idx+=1
                   
            return(allEv_data)


# file_path = '/home/gerard/analysis/from_matlab/summary/ct20_test.mat'
# file = h5py.File(file_path, 'r')
# #list(f.items())
# #list(f['/cjdata'].keys())
# list(file['/ct20_test'].keys())

# allEv = (file['/ct20_test/allEv'])
# print(type(allEv))
# list(file['/ct20_test/allEv'].keys())
# #allEv2 = (file['/ct20_test/allEv/A'][:])
# #print(allEv2)
