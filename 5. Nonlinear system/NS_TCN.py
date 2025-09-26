# %%
import matplotlib.pyplot as plt
import torch
from torch import nn
import time
import numpy as np
from scipy.io import savemat, loadmat
import os, sys
import glob
device = torch.device('cuda:0') # 'cuda:0' 'cpu'

exec(open('NS_exec.py').read())
from TCN_tools import TCN, TCN_training
os.chdir(current_path)
sys.path.append('.')

u = np.transpose(u).reshape([N2, 1, Nt])
y_ref = np.transpose(y_ref).reshape([N2, 1, Nt])
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
for i in range(j,1,-1):
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

    best_params, loss_all, loss_m, cost_time, im, cost_time_str = TCN_training(TCN_model, N, u_torch[0:N1,:,:], y_ref_torch[0:N1,:,:])
    torch.save(best_params, './saved_models/'+name+'_TCN_'+str(kernel_size)+'_'+'{:.4e}'.format(loss_m).replace('e+0', 'e').replace('e-0', 'e-')+'_'+str(im)+'_'+cost_time_str+'.pt')
    savemat('./saved_data/'+name+'_TCN_'+str(kernel_size)+'_'+'{:.4e}'.format(loss_m).replace('e+0', 'e').replace('e-0', 'e-')+'_'+str(im)+'_'+cost_time_str+'.mat', {'loss_all': loss_all,
     'cost_time': cost_time, 'trainable_num': trainable_num})

All_end = time.time()
All_cost_time = All_end - All_start
All_cost_time_str = time_str(All_cost_time)
print('Total training time for all cases: ' + All_cost_time_str)
# %%
kernel_size = 19
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
R2_train=0
R2_test=0
for i in range(0,N1):
    R2_train= R2_train + r2_score(y_ref[i,:,:].transpose()[0:Nt, 0], y_pre[i,:,:].transpose()[0:Nt, 0])/N1
for i in range(N1,N2):
    R2_test= R2_test + r2_score(y_ref[i,:,:].transpose()[0:Nt, 0], y_pre[i,:,:].transpose()[0:Nt, 0])/(N2-N1)
# R2_train, R2_test, trainning time, inference_time, parameter number
# 0.9995211342084681, 0.5436568703751266, 2m8s, 0.005629062652587891s, 6537

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
for i in range(N1,N2):
    plt.plot(t,y_ref[i,:,:].transpose())
    plt.plot(t,y_pre[i,:,:].transpose())
    plt.show()