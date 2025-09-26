# %%
import matplotlib.pyplot as plt
import torch
from torch import nn
import time
import numpy as np
from scipy.io import savemat, loadmat
import os, sys
import glob
device = torch.device('cpu') # 'cuda:0' 'cpu'

exec(open('NS_exec.py').read())
from AUNN_tools import AUNN_create, AUNN_cal, AUNN_training
os.chdir(current_path)
sys.path.append('.')

u = np.transpose(u)
y_ref = np.transpose(y_ref)
bias_status=False

all_size=np.empty(shape=(0, 3))
for i in range(0,21):
    j=i-1
    k=i+j
    all_size=np.insert(np.array([[i, j, k]], dtype=np.int32), 0, all_size, axis=0)
N_size=np.size(all_size, 0)
non_layer_s=1
# %%
N = 20000 # training num 20000
All_start = time.time()
for i in range(10,21):
    x_s=all_size[i,0]
    y_s=all_size[i,1]
    in_s=x_s+y_s
    non_neuron = np.zeros(non_layer_s, dtype=np.int32)
    non_neuron[:]=all_size[i,2]
    AUNN_model = AUNN_create(in_s, non_layer_s, non_neuron, device, bias_status)
    # total_num = sum(p.numel() for p in AUNN_model.parameters())
    trainable_num = sum(p.numel() for p in AUNN_model.parameters() if p.requires_grad)

    best_params, loss_all, loss_m, cost_time, im, cost_time_str = AUNN_training(N1, Nt, u[:N1,:], AUNN_model, x_s, in_s, N, y_ref[:N1,:], device)
    torch.save(best_params, './saved_models/'+name+'_AUNN_'+str(x_s)+'_'+'{:.4e}'.format(loss_m).replace('e+0', 'e').replace('e-0', 'e-')+'_'+str(im)+'_'+cost_time_str+'.pt')
    savemat('./saved_data/'+name+'_AUNN_'+str(x_s)+'_'+'{:.4e}'.format(loss_m).replace('e+0', 'e').replace('e-0', 'e-')+'_'+str(im)+'_'+cost_time_str+'.mat', {'loss_all': loss_all,
     'cost_time': cost_time, 'trainable_num': trainable_num})

All_end = time.time()
All_cost_time = All_end - All_start
All_cost_time_str = time_str(All_cost_time)
print('Total training time for all cases: ' + All_cost_time_str)
# %%
x_s=17
in_s=x_s+x_s-1
non_layer_s=1
non_neuron = np.zeros(non_layer_s, dtype=np.int32)
non_neuron[:]=in_s
AUNN_model = AUNN_create(in_s, non_layer_s, non_neuron, 'cpu', bias_status)
trainable_num = sum(p.numel() for p in AUNN_model.parameters() if p.requires_grad)
pt_match = glob.glob(os.path.join('./saved_models', '*_AUNN_'+str(x_s)+'_*.pt'))
AUNN_model.load_state_dict(torch.load(pt_match[0], map_location='cpu'))
start = time.time()
y_pre_torch = AUNN_cal(N2, Nt, u, AUNN_model, x_s, in_s, 'cpu')
end = time.time()
inference_time = end - start 
y_pre = y_pre_torch.detach().numpy()
from sklearn.metrics import r2_score
R2_train=0
R2_test=0
for i in range(0,N1):
    R2_train = R2_train + r2_score(y_ref[i, 0:Nt], y_pre[i, 0:Nt])/N1
for i in range(N1,N2):
    R2_test= R2_test+r2_score(y_ref[i, 0:Nt], y_pre[i, 0:Nt])/(N2-N1)
# R2_train, R2_test, trainning time, inference_time, parameter number
# 0.9996227481167287, 0.9984199164244437,35m42s, 0.1321103572845459s, 1155

i=0
plt.subplot(2, 1, 1)
plt.plot(t, y_ref[i, :])
plt.plot(t, y_pre[i, :])
i=1
plt.subplot(2, 1, 2)
plt.plot(t, y_ref[i, :])
plt.plot(t, y_pre[i, :])
plt.show()
# %%
for i in range(N1,N2):
    plt.plot(t, y_ref[i, :])
    plt.plot(t, y_pre[i, :])
    plt.show()