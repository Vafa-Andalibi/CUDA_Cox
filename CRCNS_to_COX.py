__author__ = 'V_AD'
# This file convert the data from CRCNS datasets to a format usable for Cox method. 
# The main file used from the datasets are ".clu" and ".res" files. 
from numpy import *
import pickle
electrodes = 4
neurons_in_cluster = 0
neurons_added = 0
spikes_dic = {}
# SR = 1/ 20000 #sampling rate
counter = 0
spikes_dic['electrodes'] = {}
for elec in range (1,electrodes+1): # replace the PATH with the path of the dataset downloaded from CRCNS
    str1 = '''with open ("PATH/crcns/ec012ec.11/ec012ec.187/ec012ec.187.clu.%d") as clust : 
                with open ("PATH/crcns/ec012ec.11/ec012ec.187/ec012ec.187.res.%d") as spikes:
                    idx = array([int(v1) for v1 in clust])
                    neuron_added = int(idx[0])
                    idx= idx[1:]
                    time =  array([int(float(v1)*0.05) for v1 in spikes])
                    spikes_dic['electrodes'][elec] = zeros([neuron_added])
                    for neuron in range (neuron_added):
                        temp_spikes = time [where (idx == neuron)]
                        spikes_dic['electrodes'][elec][neuron] = int(len(temp_spikes))
                        spikes_dic['electrodes'][elec] =spikes_dic['electrodes'][elec].astype(int)
                        spikes_dic[counter] = temp_spikes
                        counter+=1
                        # if int(len(temp_spikes)) > 256:
                            # if int(len(temp_spikes)) > 1000 :
                            #     spikes_dic[counter] = temp_spikes [:1000]
                            #     counter+=1
                            # else:
                            #     spikes_dic[counter] = temp_spikes
                            #     counter+=1



           '''%(elec,elec)

    exec str1
path = '___' # path for saving the result (it will be imported in main file of cox method)
with open (path,'wb') as handle:
    pickle.dump(spikes_dic, handle)
