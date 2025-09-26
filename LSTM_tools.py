# %% 
import random
import datetime
from time import strftime
import numpy as np
import torch
from torch import nn, optim
from general_tools import time_str#, early_stop_check
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

def LSTM_create(in_s, layer_s, hidden_s, device, bias_status):
    # Define LSTM
    class Lstm(nn.Module):
        def __init__(self):
            super(Lstm, self).__init__()
            self.lstm = nn.LSTM(in_s, hidden_s, layer_s, bias = bias_status)
            self.linear = nn.Linear(hidden_s, 1, bias = bias_status)

        def forward(self, u):
            h0 = torch.zeros(layer_s, np.size(u, 1), hidden_s).to(device)
            c0 = torch.zeros(layer_s, np.size(u, 1), hidden_s).to(device)
            y, (hn, cn) = self.lstm(u, (h0, c0))
            y = self.linear(y)
            return y

    LSTM_model = Lstm().to(device)
    return LSTM_model

def LSTM_training(Amp_N, Nt, u, LSTM_model, in_s, N, y_ref, device):
    y_ref_torch = torch.tensor(y_ref, dtype=torch.float64).reshape([Nt, Amp_N, in_s]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(LSTM_model.parameters(), 1e-3, weight_decay=1e-8)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=10, threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=1e-4, eps=1e-8)
    # loss_last100=0.0
    im=1
    loss_all = np.zeros((N + 1, 1))

    u_torch = torch.tensor(u, dtype=torch.float64).reshape([Nt, Amp_N, in_s]).to(device)
    y_pre_torch = LSTM_model(u_torch)
    loss = criterion(y_pre_torch, y_ref_torch)
    loss_all[0:1, :] = loss.item()
    loss_m = loss.item()
    best_params = copy.deepcopy(LSTM_model.state_dict())

    start = datetime.datetime.now()
    for i in range(N):
        LSTM_model.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step(loss)

        y_pre_torch = LSTM_model(u_torch)
        loss = criterion(y_pre_torch, y_ref_torch)
        i1 = i + 1
        loss_all[i1:i1 + 1, :] = loss.item()

        if loss.item() < loss_m:
            loss_m = loss.item()
            best_params = copy.deepcopy(LSTM_model.state_dict())
            im=i1

        if i1 % 10 == 0 or i == 0:
            print(f'Iteration: {i1}/{N}({i1/N*100:.2f}%), loss: ' + '{:.4e}'.format(loss.item()))
            end = datetime.datetime.now()
            cost_time = (end - start).total_seconds()
            cost_time_str = time_str(cost_time)
            per_time = cost_time / i1
            print('Average time per training: '+'{:.4f}'.format(per_time)+'s, Cumulative training time: '+cost_time_str)
            left_time = (N - i1) * per_time
            left_time_str = time_str(left_time)
            print('Executed at ' + strftime('%Y-%m-%d %H:%M:%S', end.timetuple()) +
                  ', left time: ' + left_time_str + '\n')
        
        # if i1==threshold_N-100:
        #     loss_last100=sum(loss_all[i1-99:i1+1])
        
        # if i1>threshold_N-100:
        #     scheduler.step(loss)

        # if i1>=threshold_N:
        #     early_stop, loss_last100 = early_stop_check(loss_all, loss_last100, i1)
        #     if early_stop==1:
        #         if i1<N:
        #             loss_all=loss_all[:i1-N, :]
        #             print("Early stopping")
        #         break

    end = datetime.datetime.now()
    cost_time = (end - start).total_seconds()
    cost_time_str = time_str(cost_time)
    print('Total training time: ' + cost_time_str + ', final loss: ' + '{:.4e}'.format(loss.item()))

    # val, idx = min((val, idx) for (idx, val) in enumerate(loss_all))
    print('Minimal loss: ' + '{:.4e}'.format(loss_m) + ', iteration: ' + str(im))
    return best_params, loss_all, loss_m, cost_time, im, cost_time_str