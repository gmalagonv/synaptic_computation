import scipy.io
import h5py
import numpy as np
import logging

import paramiko
import getpass
import os
import argparse
import sys
import pandas as pd
import time
#from timeit import default_timer as timer
###############################################################
#start = timer()
# Replace 'file_path' with the path to your MATLAB file.
#path = '/media/data/nextcloud/analysis/summary' #/home/gerard/analysis/from_matlab/summary/ct20_test.mat'





def call_remote_file_ssh(ssh_credentials, paths):
    
    times_try_connections = 2
    # Unpack SSH credentials
    hostname = ssh_credentials['hostname']
    #port = ssh_credentials.get('port', 22)  # Default to port 22 if not specified
    username = ssh_credentials['username']
    private_key_path = ssh_credentials['private_key_path']# '/home/gerard/.ssh/id_rsa'
    #passphrase = getpass.getpass(prompt="Enter passphrase for private key: ")
    passphrase = ssh_credentials['passphrase']

    # Construct remote file path

    # Initialize SSH client
    ssh = paramiko.SSHClient()
    # ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    # Load SSH host keys.
    ssh.load_system_host_keys()
    # Add SSH host key automatically if needed.
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    # private_key = paramiko.RSAKey.from_private_key_file('/home/gerard/.ssh/id_rsa', password='naranja8712')
    private_key = paramiko.RSAKey.from_private_key_file(private_key_path, password=passphrase)
    # 2 debug uncomment the following line
    #logging.basicConfig(level=logging.DEBUG)
    
    for i in range(0, times_try_connections):
        
        try:
            ssh.connect(hostname, username=username, password=None, pkey=private_key, allow_agent=False, look_for_keys= False)
            print("Connected to", hostname, "using SSH key authentication")
            break
        except:
            if i == 0:
                os.system("ssh gerard@186.154.6.93 -p 24 'python3 python_stuff/servo/servo_on_pc.py'")
                print("Waiting for connection...")
                time.sleep(15)
        
    
    
    # Use SCPClient to transfer the local script to the remote server
    local_script_path = paths['local_script_path']
    remote_script_path = paths['remote_script_path']

    # Check if the local file exists
    if not os.path.isfile(local_script_path):
        raise FileNotFoundError(f"Local script '{local_script_path}' not found.")
    else:
        print('script will be copied:')
    

    #function_name = paths['function_name']
    args = paths['args']
    print ('local script path :', local_script_path)
    print ('remote script path :', remote_script_path)
    sftp = ssh.open_sftp()    
    sftp.put(local_script_path, remote_script_path)
    
    # Path to Anaconda activation script
    conda_activate = '/home/gerard/anaconda3/bin/activate'

    # Anaconda environment name
    conda_env = paths['conda_env']

    # Construct the command to execute the specific function 
    # command = f"source {conda_activate} {conda_env} && python3 {remote_script_path} {function_name} " + ' '.join(args)
    command = f"source {conda_activate} {conda_env} && python3 {remote_script_path}  " + ' '.join(args)
    print('command 2 execute:', command)
    # Execute the remote command
    stdin, stdout, stderr = ssh.exec_command(command)
    


    # Check the exit status
    exit_status = stdout.channel.recv_exit_status()

    # Read output from stdout
    output = stdout.read().decode('utf-8')

    # Read error messages from stderr
    errors = stderr.read().decode('utf-8')

    # Check exit status and handle accordingly
    if exit_status == 0:
        print("Command executed successfully")
        print("Output:")
        print(output)
    else:
        print(f"Command execution failed with exit status: {exit_status}")
        print("Errors:")
        print(errors)

    args = paths['args'] 
   

    if int(args[0]) != 0:
        if len(args) <= 5:
            results_path = '/home/gerard/nextcloud/analysis/synaptic_computation/results/' + args[1] +'.csv'
        elif len(args) > 5:
            results_path = args[5]

            results_path = results_path.split()

            # Extract the last element from the split string, which is the file path
            results_path = results_path[-1]

        print("file was saved locally as: " + results_path)
        sftp.get(results_path, results_path)

    sftp.close()
    ssh.close()

    




def matlab2python(
                  actionFlag, 
                  file_name, 
                  section,
                  fields2call, 
                  #file_path='/media/data/analysis/2020/02.21.2020',
                  file_path='/media/data/analysis/summary',
                  results_path = None# 

                  ):
    


    """"
    actionFlag : if == 0: only list the fields, else, load the fields  
    section : possible values: allEv / perSyn
    file_path = '/media/data/nextcloud/analysis/summary/ct20_test.mat' #/home/gerard/analysis/from_matlab/summary/ct20_test.mat'
    fields2call = ['synID','x','y', 'timeFromLast']
    
     """
    if results_path == None:
        results_path = '/home/gerard/nextcloud/analysis/synaptic_computation/results/'+ file_name +'.csv'
    
    # Inside matlab2python function
    print(f"Executing matlab2python with arguments: actionFlag={actionFlag}, file_name='{file_name}', section='{section}', fields2call={fields2call}, results_path='{results_path}'")
    #print(f"Executing matlab2python with arguments: actionFlag={actionFlag}, file_name='{file_name}', section='{section}', fields2call={fields2call}, results_path='{results_path}'", file=sys.stderr)
    print(file_path, file_name)
    file_path = file_path + '/' + file_name + '.mat'
    
    if "NOr_" in file_name:
        file_name = '/Data/'+ section + '/'
    else:
        file_name = '/' + file_name + '/'+ section + '/'


    # Open the MATLAB file using h5py
    print('MATLAB file:', file_path )
    with h5py.File(file_path, 'r') as file:

        if actionFlag == 0:
            print(list(file[file_name].keys()))
            return list(file[file_name].keys())
        else:
            print('OPENING :',  file_path)


            # Access the dataset
            allEv_data = np.array([])#[]
            column_names = []
            idx = 0 

            for field in fields2call:
                allEv_dataset = file[file_name + field]
                # Convert the references to strings
                allEv_references = [ref.item() for ref in allEv_dataset]
                carrier =  []
                #carrier.append(field)

                for ref in allEv_references:
                    item = file[ref]
                    
                    if isinstance(item, h5py.Dataset):
                        if (item.shape) != (1,1):
                            print('long field, not included ', field)
                            break
                        carrier.append(item[:])
                 # this would be useful for cluster information, work in progress...
                    elif isinstance(item, h5py.Group):
                        for name in item:
                            print(item[name])

                 ####       
                carrier = np.squeeze(np.array(carrier))

                

                if allEv_data.size == 0 and carrier.size > 0:
                    allEv_data = np.append(allEv_data,carrier)
                    print(idx, field)
                    idx+=1

                elif allEv_data.size != 0 and carrier.size > 0:
                    allEv_data = np.column_stack((allEv_data, carrier))
                    print(idx, field)
                    idx+=1
                column_names.append(field)

            df = pd.DataFrame(allEv_data, columns=column_names)
            
            #change some columns to int
            columns_int = ['synID', 
                           'frame', 
                           'frameRelat',
                           'timeFromLast',
                           ]
            
            columns2change = list(set(column_names) & set(columns_int))

            if columns2change:
                for k in columns2change:
                    #df[k] = df[k].astype(int)
                    df[k] = pd.to_numeric(df[k], errors='coerce').fillna(np.nan).astype('Int64')

            try:
                #np.savetxt(results_path, allEv_data, delimiter=",")
                #np.savetxt(results_path, df, delimiter=",")
                df.to_csv(results_path, index=False)  # Set index=False to exclude row indices from the CSV file

                if not os.path.isfile(results_path):
                    raise FileNotFoundError(f"Local script '{results_path}' not found.")
                else:
                    print('file was saved remotelly as:', results_path)
                
            except Exception as e:
                print('Error occurred while saving the file:', e)
            
            return(allEv_data)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run matlab2python function from command line.")
    parser.add_argument("actionFlag", type=int, help="Action flag (0 for listing fields, 1 for loading fields).")
    parser.add_argument("file_name", type=str, help="File name.")
    parser.add_argument("section", type=str, help="Section (allEv or perSyn).")
    parser.add_argument("fields2call", type=str, help="Comma-separated list of fields to call (e.g., 'synID,x,y,timeFromLast').")
    parser.add_argument("--file_path", type=str, default="/media/data/analysis/summary", help="File path.")
    parser.add_argument("--results_path", type=str, default=None, help="Output filename.")

    # parser.add_argument("--file_path", type=str, default="/media/data/analysis/2020/02.21.2020", help="File path.")

    args = parser.parse_args()

    # Parse fields2call into a list of strings
    fields2call = args.fields2call.split(',')

    # Call matlab2python function with parsed arguments
    matlab2python(args.actionFlag, args.file_name, args.section, fields2call, args.file_path, args.results_path)
