# coding=utf-8
import os
import torch
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from loader.ngsimloader import NgsimDataset
from model.interaction import Interaction
from model.denoise import TransformerDenoisingModel

#-----------------------------------超参数设置----------------------------------------------------
args = {}
args['use_cuda'] = True

#denoise
args['NUM_Tau'] = 5
args['beta_schedule'] = 'linear'
args['n_steps'] = 10
args['beta_start'] = 1.e-5
args['beta_end'] = 5.e-2

# initial
args['decoder_size'] = 128
args['grid_size'] = (13,5)
args['soc_conv_depth'] = 64
args['conv_3x1_depth'] = 16
args['dyn_embedding_size'] = 32
args['input_embedding_size'] = 32
args['use_maneuvers'] = True
args['train_flag'] = False
args['traj_linear_hidden'] = 32
args['lstm_encoder_size'] = 64
args['encoder_size'] = 64
args['relu'] = 0.1
args['n_head'] = 4
args['att_out'] = 48
args['dropout'] = 0
args['in_length'] = 16
args['out_length'] = 25
args['use_elu'] = True
args['lat_length'] = 3
args['lon_length'] = 2


# ------------------------------------实例化数据集------------------------------------------------
tsSet = NgsimDataset(mat_file='./datasets/NGSIM/TestSet.mat')
ts_loader = DataLoader(tsSet, batch_size=args['batch_size'], shuffle=True, collate_fn=tsSet.collate_fn, num_workers=8, drop_last = True,pin_memory=True)


#------------------------------------定义模型---------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
cuda_idx=0
net = TransformerDenoisingModel().cuda(cuda_idx)
net.load_state_dict(torch.load('./result/denoise_highd/denoise.tar'))
torch.cuda.set_device(cuda_idx)
model = LEDInitializer(args).cuda(cuda_idx)
model.load_state_dict(torch.load('./ngsim/check/initial.tar'))

for param in net.parameters():
    param.requires_grad = False
opt = torch.optim.AdamW(model.parameters(), lr=args['learning_rate'])
# scheduler_model = torch.optim.lr_scheduler.StepLR(opt, step_size=decay_step, gamma=decay_gamma)
crossEnt = torch.nn.BCELoss()


#-------------------------------------辅助函数-------------------------------------------------------
def make_beta_schedule(schedule: str = 'linear', n_timesteps: int = 1000, 
    start: float = 1e-5, end: float = 1e-2) -> torch.Tensor:
    if schedule == 'linear':
        betas = torch.linspace(start, end, n_timesteps)
    elif schedule == "quad":
        betas = torch.linspace(start ** 0.5, end ** 0.5, n_timesteps) ** 2
    elif schedule == "sigmoid":
        betas = torch.linspace(-6, 6, n_timesteps)
        betas = torch.sigmoid(betas) * (end - start) + start
    return betas


# ------------------------------计算去噪过程中使用的参数--------------------------------------------------
'''
beta, alphas, alphas_prod, alphas_bar_sqrt, one_minus_alphas_bar_sqrt维度均为(n_steps,)[100,]
'''
betas = make_beta_schedule(schedule=args['beta_schedule'], n_timesteps=args['n_steps'], start=args['beta_start'], end=args['beta_end']).cuda(cuda_idx)
alphas = 1 - betas
alphas_prod = torch.cumprod(alphas, 0)
alphas_bar_sqrt = torch.sqrt(alphas_prod)
one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)


def extract(input, t, x):
    shape = x.shape
    out = torch.gather(input, 0, t.to(input.device))
    reshape = [t.shape[0]] + [1] * (len(shape) - 1)
    return out.reshape(*reshape)


def p_sample_accelerate(hist, mask, cur_y, i):
    #hist:B,16,2; mask:B,B; fut_pred:B,10,25,2, t
    if i==0:
        z = torch.zeros_like(cur_y).to(hist.device)
    else:
        z = torch.randn_like(cur_y).to(hist.device)
    i = torch.tensor([i]).cuda(cuda_idx)
    # Factor to the model output
    eps_factor = ((1 - extract(alphas, i, cur_y)) / extract(one_minus_alphas_bar_sqrt, i, cur_y))
    beta = extract(betas, i.repeat(hist.shape[0]), cur_y)
    eps_theta = net.generate_accelerate(cur_y, beta, hist, mask) 
    mean = (1 / extract(alphas, i, cur_y).sqrt()) * (cur_y - (eps_factor * eps_theta))
    # Generate z
    z = torch.randn_like(cur_y).to(hist.device)
    sigma_t = extract(betas, i, cur_y).sqrt()		
    sample = mean + sigma_t * z * 0.00001
    return (sample)


def p_sample_loop_accelerate(hist, mask, fut_pred):
    #hist:B,16,2; mask:B,B; fut_pred:B,K,25,2
    prediction_total = torch.Tensor().cuda(cuda_idx)
    cur_y = fut_pred[:, :10].to(hist.device)
    # 逐步去噪
    for i in reversed(range(args['NUM_Tau'])):
        cur_y = p_sample_accelerate(hist, mask, cur_y, i)
    cur_y_ = fut_pred[:, 10:].to(hist.device) 
    for i in reversed(range(args['NUM_Tau'])):
        cur_y_ = p_sample_accelerate(hist, mask, cur_y_, i)
    prediction_total = torch.cat((cur_y_, cur_y), dim=1)
    return prediction_total

def data_preprocess(hist,fut_pred):
    '''
    hist:(16,B,2)
    fut_pred.shape:(25,B,K,2)
    '''
    hist = hist.permute(1,0,2)
    fut_pred = fut_pred.permute(1,2,0,3) 
    batch_size = hist.shape[0]
    mask = torch.zeros(batch_size, batch_size).cuda(cuda_idx)
    for i in range(batch_size):
        mask[i:i+1, i:i+1] = 1.
    return hist, mask, fut_pred



performance = { 'FDE': [0, 0, 0, 0, 0],
				'ADE': [0, 0, 0, 0, 0]}
samples = 0
args['train_flag'] = False
with torch.no_grad():
    for i, data in enumerate(ts_loader):
        hist, nbrs, nbmask, lat_enc, lon_enc, fut, op_mask, _ = data
        #use gpu
        nbrs = nbrs.cuda(cuda_idx)
        nbmask = nbmask.cuda(cuda_idx)
        hist = hist.cuda(cuda_idx)
        fut = fut.cuda(cuda_idx)
        lat_enc = lat_enc.cuda(cuda_idx)
        lon_enc = lon_enc.cuda(cuda_idx)
        op_mask = op_mask.cuda(cuda_idx)

        if args['use_maneuvers']:
            # Forward pass
            fut_pred, lat_pred, lon_pred = model(hist, nbrs, nbmask, lat_enc, lon_enc)
            fut_pred_max = torch.zeros_like(fut_pred[0])
            for k in range(lat_pred.shape[0]):
                lat_man = torch.argmax(lat_pred[k, :]).detach()
                lon_man = torch.argmax(lon_pred[k, :]).detach()
                indx = lon_man*3 + lat_man
                fut_pred_max[:,k,:] = fut_pred[indx][:,k,:] 
            hist, mask, fut_pred_max = data_preprocess(hist,fut_pred_max)
            # denoise
            y_pred = p_sample_loop_accelerate(hist, mask, fut_pred_max) 
        else:
            fut_pred, _, _ = model(hist, nbrs, nbmask, lat_enc, lon_enc)
            hist, mask, fut_pred = data_preprocess(hist,fut_pred)
            # Denoise
            y_pred = p_sample_loop_accelerate(hist, mask, fut_pred) 

        fut = fut.unsqueeze(1).repeat(1, 20, 1, 1)
        fut = fut.permute(2,1,0,3)  #B,20,25,2
        # op_mask = op_mask.unsqueeze(1).repeat(1, 20, 1, 1)
        # op_mask = op_mask.permute(2,1,0,3)  #B,20,25,2

        # y_pred = y_pred*op_mask

        distances = torch.norm(fut - y_pred, dim=-1)
        for time_i in range(1, 6):
            ade = (distances[:, :, :5*time_i]).mean(dim=-1).min(dim=-1)[0].sum()
            fde = (distances[:, :, 5*time_i-1]).min(dim=-1)[0].sum()
            performance['ADE'][time_i-1] += ade.item()
            performance['FDE'][time_i-1] += fde.item()
        samples += distances.shape[0]
            # if count==2:
            # 	break
for time_i in range(5):
    print(time_i + 1)
    print(performance['ADE'][time_i]*0.3048/samples)
    print(performance['FDE'][time_i]*0.3048/samples)
