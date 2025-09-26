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
from RNN_tools import RNN_create, RNN_training
os.chdir(current_path)
sys.path.append('.')

bias_status=False
layer_s=1
in_s=1
# %%
N = 20000 # training num 20000
All_start = time.time()
for i in range(2,41,2):
    hidden_s=i
    RNN_model = RNN_create(in_s, layer_s, hidden_s, device, bias_status)
    # total_num = sum(p.numel() for p in RNN_model.parameters())
    trainable_num = sum(p.numel() for p in RNN_model.parameters() if p.requires_grad)

    best_params, loss_all, loss_m, cost_time, im, cost_time_str = RNN_training(N1, Nt, u[:,:N1], RNN_model, in_s, N, y_ref[:,:N1], device)
    torch.save(best_params, './saved_models/'+name+'_RNN_'+str(hidden_s)+'_'+'{:.4e}'.format(loss_m).replace('e+0', 'e').replace('e-0', 'e-')+'_'+str(im)+'_'+cost_time_str+'.pt')
    savemat('./saved_data/'+name+'_RNN_'+str(hidden_s)+'_'+'{:.4e}'.format(loss_m).replace('e+0', 'e').replace('e-0', 'e-')+'_'+str(im)+'_'+cost_time_str+'.mat', {'loss_all': loss_all,
     'cost_time': cost_time, 'trainable_num': trainable_num})

All_end = time.time()
All_cost_time = All_end - All_start
All_cost_time_str = time_str(All_cost_time)
print('Total training time for all cases: ' + All_cost_time_str)
# %%
hidden_s=40
RNN_model = RNN_create(in_s, layer_s, hidden_s, 'cpu', bias_status)
trainable_num = sum(p.numel() for p in RNN_model.parameters() if p.requires_grad)
pt_match = glob.glob(os.path.join('./saved_models', '*_RNN_'+str(hidden_s)+'_*.pt'))
RNN_model.load_state_dict(torch.load(pt_match[0], map_location='cpu'))
u_torch = torch.tensor(u).reshape([Nt, N2, in_s])
start = time.time()
y_pre_torch = RNN_model(u_torch)
end = time.time()
inference_time = end - start 
y_pre = y_pre_torch.reshape([Nt, N2]).detach().numpy()
from sklearn.metrics import r2_score
R2_train=0
R2_test=0
for i in range(0,N1):
    R2_train = R2_train + r2_score(y_ref[0:Nt, i], y_pre[0:Nt, i])/N1
for i in range(N1,N2):
    R2_test= R2_test+r2_score(y_ref[0:Nt, i], y_pre[0:Nt, i])/(N2-N1)
# R2_train, R2_test, trainning time, inference_time, parameter number
# 0.9997709579672038, 0.9956913561859403, 10m52s, 0.06981849670410156s, 1680

i=0
plt.subplot(2, 1, 1)
plt.plot(t,y_ref[:,i])
plt.plot(t,y_pre[:,i])
i=1
plt.subplot(2, 1, 2)
plt.plot(t,y_ref[:,i])
plt.plot(t,y_pre[:,i])
plt.show()
# %%
for i in range(N1,N2):
    plt.plot(t,y_ref[:,i])
    plt.plot(t,y_pre[:,i])
    plt.show()

#%%
f = loadmat('NS_y_PhyCNN.mat')
y_pre = f['y_PhyCNN']

from sklearn.metrics import r2_score
R2_train=0
R2_test=0
for i in range(0,N1):
    R2_train = R2_train + r2_score(y_ref[0:Nt, i], y_pre[0:Nt, i])/N1
for i in range(N1,N2):
    R2_test= R2_test+r2_score(y_ref[0:Nt, i], y_pre[0:Nt, i])/(N2-N1)

# 0.8758613649416562 0.7473817323420459