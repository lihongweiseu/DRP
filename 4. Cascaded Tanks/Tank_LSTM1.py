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
from LSTM_tools import LSTM_create, LSTM_training
from general_tools import time_str
os.chdir(current_path)
sys.path.append('.')
device = torch.device('cpu') # 'cuda:0' 'cpu'

name='Tank'
# Prepare data
dt = 4
Nt=1024
f = loadmat(name+'_data.mat')
u = f['u']
y_ref = f['y_ref']
tend = (Nt - 1) * dt
t = np.linspace(0, tend, Nt).reshape(-1, 1)
del f, dt, tend
bias_status=False
layer_s=1
in_s=1
# %%
N = 20000 # training num 20000
All_start = time.time()
for i in range(1,21):
    hidden_s=i
    LSTM_model = LSTM_create(in_s, layer_s, hidden_s, device, bias_status)
    # total_num = sum(p.numel() for p in LSTM_model.parameters())
    trainable_num = sum(p.numel() for p in LSTM_model.parameters() if p.requires_grad)

    best_params, loss_all, loss_m, cost_time, im, cost_time_str = LSTM_training(1, Nt, u[:,:1], LSTM_model, in_s, N, y_ref[:,:1], device)
    torch.save(best_params, './saved_models/'+name+'_LSTM_'+str(hidden_s)+'_'+'{:.4e}'.format(loss_m).replace('e+0', 'e').replace('e-0', 'e-')+'_'+str(im)+'_'+cost_time_str+'.pt')
    savemat('./saved_data/'+name+'_LSTM_'+str(hidden_s)+'_'+'{:.4e}'.format(loss_m).replace('e+0', 'e').replace('e-0', 'e-')+'_'+str(im)+'_'+cost_time_str+'.mat', {'loss_all': loss_all,
     'cost_time': cost_time, 'trainable_num': trainable_num})

All_end = time.time()
All_cost_time = All_end - All_start
All_cost_time_str = time_str(All_cost_time)
print('Total training time for all cases: ' + All_cost_time_str)
# %%
hidden_s=20
LSTM_model = LSTM_create(in_s, layer_s, hidden_s, 'cpu', bias_status)
trainable_num = sum(p.numel() for p in LSTM_model.parameters() if p.requires_grad)
pt_match = glob.glob(os.path.join('./saved_models', '*_LSTM_'+str(hidden_s)+'_*.pt'))
LSTM_model.load_state_dict(torch.load(pt_match[0], map_location='cpu'))
u_torch = torch.tensor(u).reshape([Nt, 2, in_s])
start = time.time()
y_pre_torch = LSTM_model(u_torch)
end = time.time()
inference_time = end - start  
y_pre = y_pre_torch.reshape([Nt, 2]).detach().numpy()
from sklearn.metrics import r2_score
R2_train= r2_score(y_ref[0:Nt, 0], y_pre[0:Nt, 0])
R2_test= r2_score(y_ref[0:Nt, 1], y_pre[0:Nt, 1])
# R2_train, R2_test, trainning time, inference_time, parameter number
# 0.9996545262588384, 0.9603526560927145, 28m7s, 0.03133511543273926s, 1700

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
y_ref_torch = torch.tensor(y_ref[:,1:2]).reshape([Nt, 1, in_s])
criterion = nn.MSELoss()
loss_LSTM=np.zeros((20, 2))
for i in range(1,21):
    mat_match = glob.glob(os.path.join('./saved_data', '*_LSTM_'+str(i)+'_*.mat'))
    f = loadmat(mat_match[0])
    loss_all=f['loss_all']
    loss_LSTM[i-1:i, :1]=np.min(loss_all)
    pt_match=mat_match[0].replace('data', 'models').replace('mat', 'pt')
    hidden_s=i
    LSTM_model = LSTM_create(in_s, layer_s, hidden_s, 'cpu', bias_status)
    LSTM_model.load_state_dict(torch.load(pt_match, map_location='cpu'))
    u_torch = torch.tensor(u[:,1:2]).reshape([Nt, 1, in_s])
    y_pre_torch = LSTM_model(u_torch)
    loss_LSTM[i-1:i, 1:2] = criterion(y_pre_torch, y_ref_torch).item()

del mat_match, pt_match, f, loss_all
savemat(name+'_loss_LSTM.mat', {name+'_loss_LSTM': loss_LSTM})