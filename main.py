__author__ = 'V_AD'
from matplotlib.pyplot import *
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from numpy import *
from datetime import *
from nt import P_DETACH
from scipy.stats import norm
from scipy.optimize.zeros import results_c
from scipy.io import *
from cox import cox
from Tkconstants import FIRST
import pickle
path = '___' # Put the path of spiking files in this variable 
with open (path,'rb') as handle:
    spikes = pickle.load(handle)
nn = len(spikes)
p = nn-1
results = zeros((nn,nn))
confidence = zeros ((nn*3-1 , nn))
ztime = datetime.now()
ztime = ztime - ztime
total = datetime.now()
total = total - total
total_length = 1096400
win_len = 15000 # length of window
overlap = 100
radius =2
win_num = int(float(total_length)/(win_len-overlap))
final_adjacency = zeros ([nn,nn])
final_results_dict = {}
print "number of windows:%d"%win_num
for win_n in range (win_num):
    l_band = win_n * win_len if win_n == 0 else (win_n * win_len)-overlap
    u_band = l_band + win_len
    temp_spikes = {}
    for qp in spikes:
        temp_spikes[qp] = array([tt for tt in spikes[qp] if (tt <u_band and tt>l_band)])
    temp_lengths = array([len(temp_spikes[q]) for q in temp_spikes])
    non_small_indices = array(where(temp_lengths>256)[0])
    selected_spikes = {}
    for non_small in non_small_indices :
        selected_spikes[non_small] = temp_spikes[non_small]
    nn_for_cox = len(non_small_indices)
    p_for_cox = nn_for_cox-1
    results_temp = zeros((nn_for_cox,nn_for_cox))
    confidence_temp = zeros ((nn_for_cox*3-1 , nn_for_cox))
    for neuron in range ( 0, nn_for_cox) :
        print neuron
        target_b = selected_spikes[non_small_indices[neuron]]
        if (neuron == 0):
            print ("Windows No. %d Length of each train in average: " %win_n)
            print(temp_lengths[non_small_indices].astype(int))
            print(average(non_small_indices))
        maxi_b = 0
        for i,q in list (enumerate (non_small_indices)):
            if i!= neuron and temp_lengths[q] > maxi_b :
                maxi_b = temp_lengths[q]
        ref_b = zeros((maxi_b,nn_for_cox-1))-1
        idx = 0
        for q in range ( 0, nn_for_cox):
            if non_small_indices[q] != non_small_indices[neuron] :
                ref_b[0:temp_lengths[non_small_indices[q]],idx] = selected_spikes[non_small_indices[q]]
                idx+=1
        tsp_b = ref_b
        delta = zeros ([nn_for_cox-1])
        tot_st = datetime.now()
        betahat,betaci,zt = cox(nn_for_cox,maxi_b , target_b, tsp_b.astype(int), delta)
        ztime = zt + ztime
        if (neuron == 0):
                results_temp[0,1:] = betahat
                confidence_temp[0:2,1:] = betaci.T
        elif (neuron == p):
            results_temp [neuron,0:neuron] = betahat
            confidence_temp [nn*3-3:,0:neuron] = betaci.T
        else:
            results_temp [neuron,0:neuron] = betahat[0:neuron]
            results_temp [neuron,neuron+1:] = betahat [neuron:]
            ind_temp = 3*(neuron+1) - 3
            confidence_temp [ind_temp:ind_temp+2 , 0:neuron] = betaci.T [:,0:neuron]
            confidence_temp [ind_temp:ind_temp+2 , neuron+1:] = betaci.T [:, neuron:]
        tot_en = datetime.now()
        total_temp = tot_en-tot_st
        total = total + total_temp
    fro = array ([])
    to = array ([])
    p_temp = 0
    for i in range (0,nn_for_cox):
        for j in range (0,nn_for_cox):
            if (results_temp[i,j]>0):
                if ((confidence_temp[p_temp+1,j]> 0 and confidence_temp[p_temp,j]>0) or (confidence_temp[p_temp+1,j]< 0 and confidence_temp[p_temp,j]< 0)):
                    final_adjacency[non_small_indices[j], non_small_indices[i]] += radius
                    fro = append(fro,non_small_indices[j])
                    to = append(to,non_small_indices[i])
        p_temp = p_temp + 3
    Xs = arange(1,nn+1)
    Ys = arange(1,nn+1)
    Ss = ones([nn])*10
    for y in range(nn) :
        for x in range(nn) :
            if final_adjacency[x,y] != 0 :
                Xs = append(Xs,x)
                Ys= append(Ys,y)
                Ss = append(Ss,final_adjacency[x,y] )
    print "hi"
with open ('PATH/Xs','wb') as h1:
    pickle.dump(Xs, h1)
with open ('PATH/Ys','wb') as h2:
    pickle.dump(Ys, h2)
with open ('PATH/Ss','wb') as h3:
    pickle.dump(Ss, h3)

fig = figure ( )
ax = fig.gca()
ax.set_xticks(arange(-0.5,35.5,1))
ax.set_yticks(arange(-0.5,35.5,1))
scatter (Xs,Ys,s=Ss)
axis('equal')
ax.set_xlim([0,35])
ax.set_ylim([0,35])
grid()
show()

########### following lines are for applying the cox method on ".res" file 
###########

# for neuron in range (0,nn):
    # with open ("PATH/restest1.res") as f:
        # #print (dir(f))
        # a = f.read().split()
        # a = map(int,a)
        # each = (len(a)+1)/nn
        # target = a[neuron*each:neuron*each+each]
        # target = nonzero(target)[0]
        # target = target + 1
        # target_b = spikes[neuron]
        # lengths = zeros (nn)
        # lengths_b = zeros (nn)
        # for i in range(0,nn):
            # lengths[i] = len(nonzero(a[(i*each):(i+1)*each])[0])
        # for q in spikes :
            # lengths_b[q] = len (spikes[q])
        # if (neuron == 0):
            # # print ("Each neuron is: " , each)
            # print ("Length of each train in average: ")
            # print(lengths_b.astype(int))
            # print(average(lengths_b))
        # maxi = 0
        # maxi_b = 0
        # for i,q in list (enumerate (lengths_b)):
            # if i!= neuron and q > maxi_b :
                # maxi_b = q
        # if (neuron == 0):
            # maxi = max(lengths[1:])
        # elif (neuron == p):
            # maxi = max(lengths[0:nn])
        # else:
            # maxi = max([max(lengths[0:neuron]),max(lengths[neuron+1:])])
        # ref = zeros((maxi,nn-1))-1
        # ref_b = zeros((maxi_b,nn-1))-1
        # idx = 0
        # for q in spikes:
            # if q != neuron :
                # ref_b[0:lengths_b[q],idx] = spikes[q]
                # idx+=1
        # if (neuron==0):
            # for i in range (0,p):
                # ref[0:lengths[i+1],i] = nonzero(a[((i+1)*each):(i+2)*each])[0]+1
        # elif (neuron == p):
            # for i in range (0,p):
                # ref[0:lengths[i],i] = nonzero(a[(i*each):(i+1)*each])[0]+1

        # else:
            # for i in range (0,neuron):
                # ref[0:lengths[i],i] = nonzero(a[(i*each):(i+1)*each])[0]+1
            # for i in range (neuron+1,nn):
                # ref[0:lengths[i],i-1] = nonzero(a[(i*each):(i+1)*each])[0]+1
        # #print (lengths)
        # #print (target)
    # # first = array([qp for qp in ref_b[:,2] if qp != -1])
    # # for delay in range (50) :
    # #     print "delay: %d" %delay
    # #     b1,b2,z1 = cox (2, len(first), target_b, first.astype(int),array([delay]))
    # #     print b1,b2,z1
    # tsp = ref
    # tsp_b = ref_b
    # delta = zeros ([p])
    # tot_st = datetime.now()
    # # betahat,betaci,zt = cox(nn,maxi,target,int_(tsp),delta)
    # tsp_b [where(tsp_b) == -1] = 0
    # betahat,betaci,zt = cox(nn,maxi_b , target_b, tsp_b.astype(int), delta)
    # ztime = zt + ztime
    # if (neuron == 0):
            # results[0,1:] = betahat
            # confidence[0:2,1:] = betaci.T
    # elif (neuron == p):
        # results [neuron,0:neuron] = betahat
        # confidence[nn*3-3:,0:neuron] = betaci.T
    # else:
        # results [neuron,0:neuron] = betahat[0:neuron]
        # results [neuron,neuron+1:] = betahat [neuron:]
        # ind_temp = 3*(neuron+1) - 3
        # confidence [ind_temp:ind_temp+2 , 0:neuron] = betaci.T [:,0:neuron]
        # confidence [ind_temp:ind_temp+2 , neuron+1:] = betaci.T [:, neuron:]
    # tot_en = datetime.now()
    # total_temp = tot_en-tot_st
    # total = total + total_temp


# print (results)
# print("\n\n\n\n\n")
# print (confidence)

# p = 0
# fro = []
# to = []
# thickness = []
# for i in range (0,nn):
    # for j in range (0,nn):
        # if (results[i,j]>0):
            # if ((confidence[p+1,j]> 0.005 and confidence[p,j]>0.005) or (confidence[p+1,j]< -0.005 and confidence[p,j]< -0.005)):
                # fro =  append(fro,j+1)
                # to = append(to,i+1)
                # thickness = append(thickness, results[i,j])
    # p = p + 3
# to_r = append (append([fro], [to],axis=0), [thickness], axis = 0)
# to_file = to_r.T
# savemat('d:/new2.mat', mdict = {'arr' : to_file})
# print("\n\n\n\n\n")
# print (to_file)
# print (ztime/nn)
# print (total)

