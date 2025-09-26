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
from LSTM_tools import LSTM_create, LSTM_training
os.chdir(current_path)
sys.path.append('.')

# Define the configuration of LSTM
bias_status=False
layer_s=1
in_s = 1
# %%
N = 20000 # training num 20000
All_start = time.time()
for i in range(1,21):
    hidden_s=i
    LSTM_model = LSTM_create(in_s, layer_s, hidden_s, device, bias_status)
    # total_num = sum(p.numel() for p in LSTM_model.parameters())
    trainable_num = sum(p.numel() for p in LSTM_model.parameters() if p.requires_grad)

    best_params, loss_all, loss_m, cost_time, im, cost_time_str = LSTM_training(N1, Nt, u[:,:N1], LSTM_model, in_s, N, y_ref[:,:N1], device)
    torch.save(best_params, './saved_models/'+name+'_LSTM_'+str(hidden_s)+'_'+'{:.4e}'.format(loss_m).replace('e+0', 'e').replace('e-0', 'e-')+'_'+str(im)+'_'+cost_time_str+'.pt')
    # savemat('./saved_data/'+name+'_LSTM_'+str(hidden_s)+'_'+'{:.4e}'.format(loss_m).replace('e+0', 'e').replace('e-0', 'e-')+'_'+str(im)+'_'+cost_time_str+'.mat', {'loss_all': loss_all,
    #  'cost_time': cost_time, 'trainable_num': trainable_num})

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
u_torch = torch.tensor(u).reshape([Nt, N2, in_s])
start = time.time()
y_pre_torch = LSTM_model(u_torch)
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
# 0.9999796432189628, 0.9994425155679736, 30m58s, 0.11157941818237305s, 1700

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