# %%
import matplotlib.pyplot as plt
import torch
from torch import nn
import time
import numpy as np
from scipy.io import savemat, loadmat
import os, sys
import glob
current_path = os.path.dirname(__file__)
parent_path = os.path.dirname(current_path)
os.chdir(parent_path)
sys.path.append('.')
from TCN_tools import TCN, TCN_training
from general_tools import time_str
os.chdir(current_path)
sys.path.append('.')
device = torch.device('cpu') # 'cuda:0' 'cpu'

name='Tank'
# Prepare data
dt = 4
Nt=1024
f = loadmat(name+'_data.mat')
u = f['u'].transpose().reshape([2, 1, Nt])
y_ref = f['y_ref'].transpose().reshape([2, 1, Nt])

tend = (Nt - 1) * dt
t = np.linspace(0, tend, Nt).reshape(-1, 1)
del f, dt, tend
bias_status=False

# Define the configuration of DSNN
in_s = 1
out_s = 1
channel_sizes = [8]*4
dropout = .0
func = 1

u_torch = torch.tensor(u).to(device)
y_ref_torch = torch.tensor(y_ref).to(device)
# %%
N = 20000 # training num 20000
All_start = time.time()
j=20
for i in range(j,10,-1):
    kernel_size = i
    model_params = {
        'input_size':   1,
        'output_size':  1,
        'num_channels': channel_sizes,
        'kernel_size':  kernel_size,
        'dropout':      dropout,
        'func':       func
    }
    TCN_model = TCN(**model_params).to(device)
    trainable_num = sum(p.numel() for p in TCN_model.parameters() if p.requires_grad)

    best_params, loss_all, loss_m, cost_time, im, cost_time_str = TCN_training(TCN_model, N, u_torch[0:1,:,:], y_ref_torch[0:1,:,:])
    torch.save(best_params, './saved_models/'+name+'_TCN_'+str(kernel_size)+'_'+'{:.4e}'.format(loss_m).replace('e+0', 'e').replace('e-0', 'e-')+'_'+str(im)+'_'+cost_time_str+'.pt')
    savemat('./saved_data/'+name+'_TCN_'+str(kernel_size)+'_'+'{:.4e}'.format(loss_m).replace('e+0', 'e').replace('e-0', 'e-')+'_'+str(im)+'_'+cost_time_str+'.mat', {'loss_all': loss_all,
     'cost_time': cost_time, 'trainable_num': trainable_num})

All_end = time.time()
All_cost_time = All_end - All_start
All_cost_time_str = time_str(All_cost_time)
print('Total training time for all cases: ' + All_cost_time_str)
# %%
kernel_size = 14
model_params = {
    'input_size':   1,
    'output_size':  1,
    'num_channels': channel_sizes,
    'kernel_size':  kernel_size,
    'dropout':      dropout,
    'func':       func
}
TCN_model = TCN(**model_params).to('cpu')
trainable_num = sum(p.numel() for p in TCN_model.parameters() if p.requires_grad)
u_torch = u_torch.to('cpu')
y_ref_torch = y_ref_torch.to('cpu')

pt_match = glob.glob(os.path.join('./saved_models', '*_TCN_'+str(kernel_size)+'_*.pt'))
# pt_match = glob.glob(os.path.join('./saved_models', 'Tank_TCN_5_2.7400e-3_19996_1m 25.902s.pt'))
best_params=torch.load(pt_match[0], map_location='cpu')

TCN_model.load_state_dict(best_params)
start = time.time()
y_pre_torch = TCN_model(u_torch)
end = time.time()
inference_time = end - start
y_pre = y_pre_torch.detach().numpy()
from sklearn.metrics import r2_score
R2_train= r2_score(y_ref[0,:,:].transpose()[0:Nt, 0], y_pre[0,:,:].transpose()[0:Nt, 0])
R2_test= r2_score(y_ref[1,:,:].transpose()[0:Nt, 0], y_pre[1,:,:].transpose()[0:Nt, 0])
# R2_train, R2_test, trainning time, inference_time, parameter number
# 0.9999561785402504, 0.8697710387701961, 2m8s, 0.005629062652587891s, 6537

i=0
plt.subplot(2, 1, 1)
plt.plot(t,y_ref[i,:,:].transpose())
plt.plot(t,y_pre[i,:,:].transpose())
i=1
plt.subplot(2, 1, 2)
plt.plot(t,y_ref[i,:,:].transpose())
plt.plot(t,y_pre[i,:,:].transpose())
plt.show()
# %%
for j in range(1, 21, 1):
    state_s=j
    state_non_neuron[0] = state_s  # user defined
    TCN_model = TCN_create(in_s, state_s, out_s, state_non_layer_s, out_non_layer_s, state_non_neuron, out_non_neuron, 'cpu', bias_status)
    pt_match = glob.glob(os.path.join('./saved_models', '*_TCN_'+str(state_s)+'_*.pt'))
    TCN_model.load_state_dict(torch.load(pt_match[0], map_location='cpu'))
    u_torch = torch.tensor(u)
    y_pre_torch = TCN_cal(2, Nt, u, TCN_model, state_s, out_s, 'cpu')
    y_pre = y_pre_torch.reshape([Nt, 2]).detach().numpy()

    i=1
    plt.plot(t, y_ref.reshape([Nt, 2])[:,i-1:i])
    plt.plot(t, y_pre[:,i-1:i])
    plt.show()
# %%
y_ref_torch = torch.tensor(y_ref[:,1:2,:])
criterion = nn.MSELoss()
loss_DSNN=np.zeros((20, 2))
for i in range(1,21):
    mat_match = glob.glob(os.path.join('./saved_data', '*_TCN_'+str(i)+'_*.mat'))
    f = loadmat(mat_match[0])
    loss_all=f['loss_all']
    loss_DSNN[i-1:i, :1]=np.min(loss_all)
    pt_match=mat_match[0].replace('data', 'models').replace('mat', 'pt')
    state_s=i
    state_non_neuron[0] = state_s  # user defined
    TCN_model = TCN_create(in_s, state_s, out_s, state_non_layer_s, out_non_layer_s, state_non_neuron, out_non_neuron, device, bias_status)
    TCN_model.load_state_dict(torch.load(pt_match, map_location='cpu'))
    y_pre_torch = TCN_cal(1, Nt, u[:,1:2,:], TCN_model, state_s, out_s, 'cpu')
    loss_DSNN[i-1:i, 1:2] = criterion(y_pre_torch, y_ref_torch).item()

del mat_match, pt_match, f, loss_all
savemat(name+'_loss_DSNN.mat', {name+'_loss_DSNN': loss_DSNN})