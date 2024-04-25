import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import itertools
from collections import Counter

################################################################
import networkx as nx
import pickle
################################################################
from sklearn.cluster import DBSCAN, OPTICS, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster

################################################################


def loader(csv_file_path):
    df = pd.read_csv(csv_file_path)
    return df


def select_evtype(df,evtype):
    if evtype == 0:
        filt_df = df
    else:
        if evtype < 3:
            mask_evtype = (df['frameRelat'] == evtype)
        else: 
            mask_evtype = (df['frameRelat'] >= evtype)
        
        filt_df = df[mask_evtype]
        filt_df = filt_df.reset_index(drop=True)
    
    return filt_df



def recalc_lastFrom(df):

    frameExposure = 50
    pixel_size = 86.6667
    # timeFromLast
    #df.loc[1::, 'timeFromLast'] = (df.loc[1::, 'frame'].values - df.loc[:len(df)-2, 'frame'].values) * 50
    df.loc[:, 'timeFromLast'] = df['frame'].diff() * frameExposure

    # distFromLast (euclidean distance)

    dx = df['x'].diff()
    dy = df['y'].diff()

        # Compute squared Euclidean distance and take square root to obtain Euclidean distance
    euclidean_distance = (np.sqrt(dx**2 + dy**2)) * pixel_size

  
        # fill column with 'euclidean_distance' in the DataFrame
    df.loc[:, 'distFromLast'] = euclidean_distance
    
    # add NaNs
    vals = df['synID'].diff()
    vals = (vals != 0) & (~np.isnan(vals))

    df.loc[vals,'timeFromLast'] = np.nan
    df.loc[vals,'distFromLast'] = np.nan

    
    return(df)




def repeatedSynID_remover(df):
    
    unique_IDsa = df['synID'].unique()
    unique_IDsb_msk = (df['synID'].diff()) != 0
    unique_IDsb = df.loc[unique_IDsb_msk, 'synID']
    
    if len(unique_IDsa) != len(unique_IDsb):
        counts = Counter(unique_IDsb)
        non_unique_synID = [elem for elem, count in counts.items() if count > 1]
        #print(non_unique_synID)

        for i in range(len(non_unique_synID)):
            filtered_indx = df.index[df['synID'] == non_unique_synID[i]].tolist()
            conc_sum_indx = ([filtered_indx[j] - filtered_indx[j-1] for j in range(1, len(filtered_indx))])
            conc_sum_indx = [1] + conc_sum_indx

            msk_indx = [idx != 1 for idx in conc_sum_indx]
            
            filtered_indx1 = list(itertools.compress(filtered_indx, msk_indx))
            val2add = 666 
            for k in filtered_indx1:
                filtered_indx_cont = [k + i for i in range(20)]

                common_elements = list(set(filtered_indx) & set(filtered_indx_cont))
                df.loc[common_elements, 'synID'] += val2add
                val2add += 123
                
        unique_IDsa = df['synID'].unique()
        unique_IDsb_msk = (df['synID'].diff()) != 0
        unique_IDsb = df.loc[unique_IDsb_msk, 'synID']
    
        if len(unique_IDsa) == len(unique_IDsb):
            print('no more repeated synIDs')
            
    else:
        print('no repeated synIDs')

    return(df)



   ######################################################################## 
   #THE FOLLOWING FUNCTIONS ARE USE ONLY INSIDE PERSYN
def grapher(df_syn):

    attributes_nodes = ['x','y','frame','frameRelat']
    attributes_edges = ['timeFromLast','distFromLast']
    df_syn['G'] = np.nan
    G = nx.DiGraph()

    # G.add_nodes_from(df_syn.index)
    for node_index in df_syn.index:
        node_attributes = df_syn.loc[node_index, attributes_nodes].to_dict()  # Get attributes for the node from DataFrame row
        G.add_node(node_index, **node_attributes)  # Add node with attributes

    for i in range(len(df_syn) - 1):
        source_node = df_syn.index[i]
        target_node = df_syn.index[i + 1]
        edge_aributes = df_syn.loc[(i + 1), attributes_edges].to_dict()
        G.add_edge(source_node, target_node, **edge_aributes)
    # Serialize the graph object to a binary string using pickle
    #serialized_graph using pickle
    df_syn.at[0, 'G'] = pickle.dumps(G)
    #print(G.nodes[0]) 
    #print(G[0][1])# G.edges[0, 1]

    return(df_syn)




def clustering(df_syn, algorithmFlag, standarizeFlag=False):

    pixel_size = 86.6667
    eps_nm = 50
    eps = eps_nm / pixel_size   # Maximum distance between samples to be considered part of the same cluster
    min_samples = 2  # Minimum number of samples in a neighborhood to form a core point

    #df_syn = df_syn[['x','y']]

    if standarizeFlag:
        # Standardize the data (optional but recommended for DBSCAN)
        scaler = StandardScaler()
        df_syn = scaler.fit_transform(df_syn)

    if algorithmFlag == 1:
        # Perform DBSCAN clustering
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        #df_syn['cluster'] = dbscan.fit_predict(df_syn)
        df_syn['cluster'] = dbscan.fit_predict(df_syn[['x','y']])
    
    elif algorithmFlag == 2:


        # Perform OPTICS clustering
        optics = OPTICS(min_samples=min_samples, max_eps=eps)  # Adjust min_samples based on density requirements
        #cluster_labels = optics.fit_predict(df_scaled)

        # Add cluster labels to DataFrame
        df_syn['cluster'] = optics.fit_predict(df_syn[['x','y']])
    
    elif algorithmFlag == 3:
        #similar 2 MATLAB clustering
        # methods at: https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html

        # Perform hierarchical clustering
        Z = linkage( df_syn[['x', 'y']], method='complete')  # Adjust method as needed
        df_syn['cluster'] = fcluster(Z, eps, criterion='distance')


    
    return(df_syn)


def re_grapher():
#only used after clusters have been calculated
    print("Bla!")


def plottter(df_syn):
    
    plt.figure(figsize=(6, 6))  # Adjust figure size (optional)
    
    if 'G'in df_syn.columns:

        serialized_graph = df_syn.at[0, 'G']
        G = pickle.loads(serialized_graph)

        pos = {node: (G.nodes[node]['x'], G.nodes[node]['y']) for node in G.nodes()}
        # Draw nodes with labels
        if 'cluster'in df_syn.columns:
            node_color = df_syn['cluster']
            # Create a ScalarMappable to define the color mapping
            sm = plt.cm.ScalarMappable(cmap='viridis')
            sm.set_array(node_color)
            plt.colorbar(sm, label='Node Color')

        else:
            node_color = 'skyblue'

        nx.draw(G, pos, with_labels=True, node_color = node_color, node_size=200, font_size=12, font_color='cyan', edge_color='gray')

        plt.title('NetworkX Graph for synapse ' + str(df_syn.loc[0, 'synID']))
        # Set equal scaling for x and y axes
        plt.axis('equal')

    
    elif 'cluster'in df_syn.columns:

        #Visualize ONLY clusters
        plt.scatter(df_syn['x'], df_syn['y'], c=df_syn['cluster'], cmap='viridis', s=100, alpha=0.8)
        plt.title('Clustering for synapse '+ str(df_syn.loc[0, 'synID']))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.colorbar(label='Cluster')
        plt.show()





   ########################################################################
    
    
def persyn(df, 
           singleSynFlag = False, #if False, it runs it over the whole dataset. if True,it will ask for a syn number (1 based).If a list, it will use the syn numbers of that list (1 based) 
           grapherFlag = False, 
           clusteringFlag = False,
           clusteringAlgorithmFlag = 2,
           plotFlag = False,
           ):

    
    collumns2add = []

    if clusteringFlag:
        collumns2add.append('cluster')

    if grapherFlag:
        collumns2add.append('G')

  

    df = repeatedSynID_remover(df)
    unique_IDs = df['synID'].unique()

    if isinstance(singleSynFlag, bool):
        if singleSynFlag:
            message = "Please enter a syn number between 1 and " + str(len(unique_IDs))
            synNum = int(input(message)) - 1
            unique_IDs = ([unique_IDs[synNum]])

    elif isinstance(singleSynFlag, list):
        if singleSynFlag [0] == 0:
            #meaning, the second number in the list is an synID
            synIDss =  singleSynFlag [1]
            index = next((i for i, x in enumerate(unique_IDs) if x == synIDss), None)
            unique_IDs  = unique_IDs [index:index+1] if index is not None else []
   
        else:
            singleSynFlag = [x - 1 for x in singleSynFlag] # 1-based to 0-based  [x - 1 for x in original_list]
            unique_IDs = unique_IDs[singleSynFlag]  
        
    for i in unique_IDs:
        df_syn = df[(df['synID'] == i)]
        original_indexes = df_syn.index
        df_syn = df_syn.reset_index(drop=True)
        
        if grapherFlag:
            df_syn = grapher(df_syn)

        if clusteringFlag:
            df_syn = clustering(df_syn, clusteringAlgorithmFlag)
        
        if plotFlag:
            plottter(df_syn)

####### filling up the original df with df_syn

        #return to original indexes:
        df_syn.index = original_indexes
        for col in collumns2add:
            if col not in df.columns:
                df[col] = np.nan
            
            df.loc[original_indexes, col] = df_syn[col]
        
    return(df)




            




        




    
    
   
def total(
        csv_file_path,
        evtype,
        singleSynFlag,
        grapherFlag,
        clusteringFlag,
        clusteringAlgorithmFlag,
        plotFlag

          ):
    
    df = loader(csv_file_path)
    df = select_evtype(df,evtype)
    df = recalc_lastFrom(df)
    
    df = persyn(
        df,
        singleSynFlag,
        grapherFlag,
        clusteringFlag,
        clusteringAlgorithmFlag,
        plotFlag

    )
    # persyn(df, 
    #        singleSynFlag = False, #if False, it runs it over the whole dataset. if True,it will ask for a syn number (1 based).If a list, it will use the syn numbers of that list (1 based) 
    #        grapherFlag = False, 
    #        clusteringFlag = False,
    #        clusteringAlgorithmFlag = 2,
    #        plotFlag = False,
    #        ):



    return(df)





