import random
import datetime
from time import strftime
import numpy as np
import torch
from torch import nn, optim
from general_tools import time_str#, early_stop_check
from torch.nn.utils.parametrizations import weight_norm
import copy

# To guarantee same results for every running, which might slow down the training speed
torch.set_default_dtype(torch.float64)
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)

class Crop(nn.Module):
 
    def __init__(self, crop_size):
        super(Crop, self).__init__()
        self.crop_size = crop_size
 
    def forward(self, x):
        return x[:, :, :-self.crop_size].contiguous()

class TemporalCasualLayer(nn.Module):
 
    def __init__(self, in_ch, out_ch, kernel_size, stride, dilation, dropout = 0.2, func = 1):
        super(TemporalCasualLayer, self).__init__()
        padding = (kernel_size - 1) * dilation
        conv_params = {
            'kernel_size': kernel_size,
            'stride':      stride,
            'padding':     padding,
            'dilation':    dilation
        }
 
        self.conv1 = weight_norm(nn.Conv1d(in_ch, out_ch, **conv_params))
        self.crop1 = Crop(padding)
        self.dropout1 = nn.Dropout(dropout)
 
        self.conv2 = weight_norm(nn.Conv1d(out_ch, out_ch, **conv_params))
        self.crop2 = Crop(padding)
        self.dropout2 = nn.Dropout(dropout)

        if func == 1:
            self.func1 = nn.ReLU()
            self.func2 = nn.ReLU()
            self.func = nn.ReLU()
        else:
            self.func1 = nn.Tanh()
            self.func2 = nn.Tanh()
            self.func = nn.Tanh()
 
        self.net = nn.Sequential(self.conv1, self.crop1, self.func1, self.dropout1,
                                 self.conv2, self.crop2, self.func2, self.dropout2)
        #shortcut connect
        self.bias = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
 
    def forward(self, x):
        y = self.net(x)
        b = x if self.bias is None else self.bias(x)
        return self.func(y + b)

class TemporalConvolutionNetwork(nn.Module):
 
    def __init__(self, num_inputs, num_channels, kernel_size = 2, dropout = 0.2, func = 1):
        super(TemporalConvolutionNetwork, self).__init__()
        layers = []
        num_levels = len(num_channels)
        tcl_param = {
            'kernel_size': kernel_size,
            'stride':      1,
            'dropout':     dropout,
            'func':       func
        }
        for i in range(num_levels):
            dilation = 2**i
            in_ch = num_inputs if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            tcl_param['dilation'] = dilation
            tcl = TemporalCasualLayer(in_ch, out_ch, **tcl_param)
            layers.append(tcl)
 
        self.network = nn.Sequential(*layers)
 
    def forward(self, x):
        return self.network(x)

class TCN(nn.Module):
 
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout, func):
        super(TCN, self).__init__()
        self.tcn = TemporalConvolutionNetwork(input_size, num_channels, kernel_size = kernel_size, dropout = dropout, func=func)
        self.linear = nn.Linear(num_channels[-1], output_size)  if num_channels[-1] != output_size else None
 
    def forward(self, x):
        y = self.tcn(x)#[N,C_out,L_out=L_in]
        if self.linear is None:
            None
        else:
            t2 = y.transpose(1, 2)
            t3 = self.linear(t2)
            y = t3.transpose(1, 2)
        return y
    
def TCN_training(TCN_model, N, u, y_ref):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(TCN_model.parameters(), 1e-3)
    im=1
    loss_all = np.zeros((N + 1, 1))

    y_pre = TCN_model(u)
    loss = criterion(y_pre, y_ref)
    loss_all[0:1, :] = loss.item()
    loss_m = loss.item()
    best_params = copy.deepcopy(TCN_model.state_dict())

    start = datetime.datetime.now()
    for i in range(N):
        TCN_model.zero_grad()
        loss.backward()
        optimizer.step()

        y_pre = TCN_model(u)
        loss = criterion(y_pre, y_ref)
        i1 = i + 1
        loss_all[i1:i1 + 1, :] = loss.item()

        if loss.item() < loss_m:
            loss_m = loss.item()
            best_params = copy.deepcopy(TCN_model.state_dict())
            im=i1

        if i1 % 10 == 0 or i == 0:
            print(f'Iteration: {i1}/{N}({i1/N*100:.2f}%), loss: ' + '{:.6e}'.format(loss.item()))
            end = datetime.datetime.now()
            cost_time = (end - start).total_seconds()
            cost_time_str = time_str(cost_time)
            per_time = cost_time / i1
            
            print('Average time per training: '+'{:.4f}'.format(per_time)+'s, Cumulative training time: '+cost_time_str)
            left_time = (N - i1) * per_time
            left_time_str = time_str(left_time)
            print('Executed at ' + strftime('%Y-%m-%d %H:%M:%S', end.timetuple()) +
                  ', left time: ' + left_time_str + '\n')
        
    end = datetime.datetime.now()
    cost_time = (end - start).total_seconds()
    cost_time_str = time_str(cost_time)
    print('Total training time: ' + cost_time_str + ', final loss: ' + '{:.6e}'.format(loss.item()))

    val, idx = min((val, idx) for (idx, val) in enumerate(loss_all))
    print('Minimal loss: ' + '{:.6e}'.format(val.item())+ ' (iteration: ' + str(idx) +')'+ '\n')
    return best_params, loss_all, loss_m, cost_time, im, cost_time_str