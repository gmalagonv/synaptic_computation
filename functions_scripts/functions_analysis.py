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
def grapher(df_syn, fusedClustersFlag=False):

    attributes_nodes = ['x','y'] #'frame' & 'frameRelat' deleted: irrelevant..... irrelevant?
    attributes_edges = ['timeFromLast','distFromLast']
    df_syn['G'] = np.nan
    G = nx.DiGraph()
    node_source = df_syn.index

    if fusedClustersFlag:


        nu_clusters = (df_syn['cluster'] - 1).tolist()
        print(nu_clusters)
        checked_indexes = [None] * len(node_source)
        startval = 0

        for i in node_source:
            if i not in checked_indexes:
                msk = [cluster == nu_clusters[i] for cluster in nu_clusters]
                #print(msk)
                #nu_clusters = [startval if mask else cluster for mask, cluster in zip(msk, nu_clusters)]
                nu_clusters = [startval if msk[i] else cluster for i, cluster in enumerate(nu_clusters)]

                idxs = [value for value, mask_value in zip(node_source, msk) if mask_value]
                checked_indexes.extend(idxs)
                startval += 1

        node_source = nu_clusters



  

    # G.add_nodes_from(df_syn.index)
    for node_index in set(node_source):
        node_indexName = node_index
        
        if fusedClustersFlag:

            msk = [cluster == node_index for cluster in nu_clusters]

            indexes = df_syn.index[msk]
            node_index = indexes[0]
        
        node_attributes = df_syn.loc[node_index, attributes_nodes].to_dict()  # Get attributes for the node from DataFrame row
        G.add_node(node_indexName, **node_attributes)  # Add node with attributes
    

    for i in range(len(node_source) - 1):
        source_node = node_source[i]
        target_node = node_source[i + 1]
        edge_aributes = df_syn.loc[(i + 1), attributes_edges].to_dict()
        G.add_edge(source_node, target_node, **edge_aributes)
    
    print('edges in grapher', G.edges())

    df_syn.at[0, 'G'] = pickle.dumps(G)
 

    return(df_syn)





def rename_clusters(clusters):
    # to rename clusters and change from 0-base to 1-based
    #mask = 
    clusters[clusters >= 0] += 1
    start_val = max(clusters) + 1
    
    if start_val == 0:
        start_val = 1
    for i in range(len(clusters)):
        if clusters[i] == -1:
            clusters[i] = start_val
            start_val += 1
    return clusters



def fuse_clusters(df_syn):
    if 'cluster' in df_syn.columns:
        df_syn['x_o'] = df_syn['x']
        df_syn['y_o'] = df_syn['y']

        unique_clusters = df_syn['cluster'].unique()
        if len(unique_clusters) < len(df_syn['cluster']):
            # print('SYNID: ', df_syn.loc[0, 'synID'])
            for i in range(len(unique_clusters)):
               
                msk = df_syn['cluster'] == unique_clusters[i]
                if sum(msk) > 1:
                  
                    df_syn.loc[msk, 'x'] = (df_syn.loc[msk, 'x']).mean()
                    df_syn.loc[msk, 'y'] = (df_syn.loc[msk, 'y']).mean()
        
        
             
        
    else:
        print('none cluster yet defined')

    return df_syn






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
        #https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters  = dbscan.fit_predict(df_syn[['x','y']])
        df_syn['cluster_o'] = clusters
        clusters = rename_clusters(clusters)
        df_syn['cluster'] = clusters


    elif algorithmFlag == 2:


        # Perform OPTICS clustering
        optics = OPTICS(min_samples=min_samples, max_eps=eps)  # Adjust min_samples based on density requirements
        #cluster_labels = optics.fit_predict(df_scaled)

        # Add cluster labels to DataFrame
        clusters = optics.fit_predict(df_syn[['x','y']])
        df_syn['cluster_o'] = clusters
        clusters = rename_clusters(clusters)
        df_syn['cluster'] = clusters

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

        print('nodes in plotter', G.nodes())
        
        pos = {node: (G.nodes[node]['x'], G.nodes[node]['y']) for node in G.nodes()}
        #print (pos)
        # Draw nodes with labels
        if 'cluster'in df_syn.columns:
            print('here ', len(df_syn['cluster']), len(list(pos.keys())))
            if len(df_syn['cluster']) == len(list(pos.keys())):
                
                node_color = df_syn['cluster']
            else:
                node_color = range(len(list(pos.keys())))
            #node_color = list(pos.keys())
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
           fusedClustersFlag = False, 
           clusteringFlag = False,
           clusteringAlgorithmFlag = 2,
           fuse_clusters_flag = False,
           plotFlag = False,
           ):

    
    collumns2add = []

    if clusteringFlag:
        collumns2add.append('cluster')
        if clusteringAlgorithmFlag == 1 or clusteringAlgorithmFlag == 2:
            collumns2add.append('cluster_o')
    if fuse_clusters_flag:
        collumns2add.append('x_o')
        collumns2add.append('y_o')

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
        
        
        if clusteringFlag:
            df_syn = clustering(df_syn, clusteringAlgorithmFlag)

        if fuse_clusters_flag:
           
            df_syn = fuse_clusters(df_syn)

        if grapherFlag:
            df_syn = grapher(df_syn, fusedClustersFlag)


        
        if plotFlag:
            plottter(df_syn)

        
####### filling up the original df with df_syn

        #return to original indexes:
        df_syn.index = original_indexes
        for col in collumns2add:
            if col not in df.columns:
                df[col] = np.nan
            

            #df.loc[original_indexes, col] = df_syn[col]
        df.loc[original_indexes] = df_syn
    return(df)
   
    
    
   
def total(
        csv_file_path,
        evtype,
        singleSynFlag,
        clusteringFlag,
        fuse_clusters_flag,
        grapherFlag,
        fusedClustersFlag,
        clusteringAlgorithmFlag,
        plotFlag

          ):
    
    df = loader(csv_file_path)
    df = select_evtype(df,evtype)
    
    df = persyn(
        df,
        singleSynFlag,
        grapherFlag,
        fusedClustersFlag,
        clusteringFlag,
        clusteringAlgorithmFlag,
        fuse_clusters_flag,
        plotFlag,

    )
    df = recalc_lastFrom(df)

    # persyn(df, 
    #        singleSynFlag = False, #if False, it runs it over the whole dataset. if True,it will ask for a syn number (1 based).If a list, it will use the syn numbers of that list (1 based) 
    #        grapherFlag = False, 
    #        clusteringFlag = False,
    #        clusteringAlgorithmFlag = 2,
    #        plotFlag = False,
    #        ):



    return(df)






