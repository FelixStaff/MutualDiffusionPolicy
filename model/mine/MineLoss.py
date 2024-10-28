# import the torch
import torch

# torch.mean(T(x_t1, x_t+1)) - torch.log(torch.mean(torch.exp(T(x_t2, x_t+1))))

# Define the loss function
def mine_loss(mine, T_x1, T_x_1, timestep, global_cond=None):
    # Calculate the loss
    # [Input Shape] T_x1, T_x2: (256, 16 ,2)
    # [Output Shape] loss: (256,)
    T = mine(T_x1, T_x_1, timestep, global_cond)
    idx = torch.randperm(T_x1.size(0))
    T_x2 = T_x1[idx].clone()
    timesteps = timestep[idx].clone()
    global_conds = global_cond[idx].clone()
    
    T_q = torch.mean(torch.exp(mine(T_x2, T_x_1, timesteps, global_conds)))
    loss = torch.mean(T) - torch.log(T_q)
    return -loss

# Define the loss function for the MINE
def train_loss_mine(mine, T_x1, T_x2, timestep, global_cond=None, ma_et=0.0001):
    # Calculate the loss
    # [Input Shape] T_x1, T_x2: (256, 16 ,2)
    # [Output Shape] loss: (256,)
    
    idx = torch.randperm(T_x1.size(0))
    T_xbar = T_x1[idx].clone()

    timestep_bar = timestep[idx].clone()
    global_cond_bar = global_cond[idx].clone()
    
    T = torch.mean(mine(T_x1, T_x2, timestep, global_cond))
    ET = torch.mean(torch.exp(mine(T_xbar, T_x2, timestep_bar, global_cond_bar)))
    if mine.ma_et is None:
        mine.ma_et = ET.detach().item()
    mine.ma_et += mine.ma_et * (ET.detach().cpu().numpy() - mine.ma_et)
    loss = T - torch.log(ET) * ET.detach() / mine.ma_et
    return -loss