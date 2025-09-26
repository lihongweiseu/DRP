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
from DSNN_tools import DSNN_create, DSNN_cal, DSNN_training
os.chdir(current_path)
sys.path.append('.')

u = u.reshape([Nt, N2, 1])
y_ref = y_ref.reshape([Nt, N2, 1])
bias_status=False

# Define the configuration of DSNN
in_s = 1
out_s = 1
state_non_layer_s=1
out_non_layer_s=1
state_non_neuron = np.zeros(state_non_layer_s, dtype=np.int32)  # size of each nonlinear layer
out_non_neuron = np.zeros(out_non_layer_s, dtype=np.int32)  # size of each nonlinear layer
out_non_neuron[0] = out_s  # user defined
# %%
N = 20000 # training num 20000
All_start = time.time()
for i in range(15,21,1):
    state_s=i
    state_non_neuron[0] = state_s  # user defined
    DSNN_model = DSNN_create(in_s, state_s, out_s, state_non_layer_s, out_non_layer_s, state_non_neuron, out_non_neuron, device, bias_status)
    # total_num = sum(p.numel() for p in DSNN_model.parameters())
    trainable_num = sum(p.numel() for p in DSNN_model.parameters() if p.requires_grad)

    best_params, loss_all, loss_m, cost_time, im, cost_time_str = DSNN_training(N1, Nt, u[:,0:N1,:], DSNN_model, N, y_ref[:,0:N1,:], state_s, out_s, device)
    torch.save(best_params, './saved_models/'+name+'_DSNN_'+str(state_s)+'_'+'{:.4e}'.format(loss_m).replace('e+0', 'e').replace('e-0', 'e-')+'_'+str(im)+'_'+cost_time_str+'.pt')
    savemat('./saved_data/'+name+'_DSNN_'+str(state_s)+'_'+'{:.4e}'.format(loss_m).replace('e+0', 'e').replace('e-0', 'e-')+'_'+str(im)+'_'+cost_time_str+'.mat', {'loss_all': loss_all,
     'cost_time': cost_time, 'trainable_num': trainable_num})

All_end = time.time()
All_cost_time = All_end - All_start
All_cost_time_str = time_str(All_cost_time)
print('Total training time for all cases: ' + All_cost_time_str)
# %%
state_s=19
state_non_neuron[0] = state_s  # user defined
DSNN_model = DSNN_create(in_s, state_s, out_s, state_non_layer_s, out_non_layer_s, state_non_neuron, out_non_neuron, 'cpu', bias_status)
trainable_num = sum(p.numel() for p in DSNN_model.parameters() if p.requires_grad)
pt_match = glob.glob(os.path.join('./saved_models', '*_DSNN_'+str(state_s)+'_*.pt'))
DSNN_model.load_state_dict(torch.load(pt_match[0], map_location='cpu'))
u_torch = torch.tensor(u)
start = time.time()
y_pre_torch = DSNN_cal(N2, Nt, u, DSNN_model, state_s, out_s, 'cpu')
end = time.time()
inference_time = end - start 
y_pre = y_pre_torch.reshape([Nt, N2]).detach().numpy()
from sklearn.metrics import r2_score
R2_train=0
R2_test=0
for i in range(0,N1):
    R2_train = R2_train + r2_score(y_ref.reshape([Nt, N2])[0:Nt, i], y_pre[0:Nt, i])/N1
for i in range(N1,N2):
    R2_test= R2_test+r2_score(y_ref.reshape([Nt, N2])[0:Nt, i], y_pre[0:Nt, i])/(N2-N1)
# R2_train, R2_test, trainning time, inference_time, parameter number
# 0.999947524450895, 0.99941593281556, 1h6m6s, 0.14338013553619385s, 1162

i=0
plt.subplot(2, 1, 1)
plt.plot(t, y_ref.reshape([Nt, N2])[:,i])
plt.plot(t, y_pre[:,i])
i=1
plt.subplot(2, 1, 2)
plt.plot(t, y_ref.reshape([Nt, N2])[:,i])
plt.plot(t, y_pre[:,i])
plt.show()