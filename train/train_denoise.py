import os
import time
import random
import numpy as np
import math
import torch
from torch.utils.data import DataLoader
from loader.ngsimloader import NgsimDataset
from Denoise import TransformerDenoisingModel
from utils import print_log

os.environ["CUDA_VISIBLE_DEVICES"] = '3'
#超参设置——————————————————————————————————————————————
batch_size = 256
trainEpochs = 100
NUM_Tau = 5
beta_schedule = 'linear'
n_steps = 100
beta_start = 1.e-3
beta_end = 5.e-2
learning_rate = 1.e-4
decay_step = 16
decay_gamma = 0.6
log_dir = './result/denoise_highd/'
log = open(os.path.join(log_dir, 'tst.txt'), 'a+')

#--------------------------------数据集加载--------------------------------------------------------------
trSet = NgsimDataset(mat_file='./datasets/NGSIM/TrainSet.mat')
tsSet = NgsimDataset(mat_file='./datasets/NGSIM/TestSet.mat')

train_loader =  DataLoader(trSet, batch_size=batch_size, shuffle=True, collate_fn=trSet.collate_fn, num_workers=8, pin_memory=True)
test_loader = DataLoader(tsSet, batch_size=batch_size, shuffle=False, collate_fn=tsSet.collate_fn, num_workers=8, pin_memory=True)


#------------------------------------------模型加载-------------------------------------------------------
cuda_idx=0
torch.cuda.set_device(cuda_idx)
net = TransformerDenoisingModel().cuda(cuda_idx)
opt = torch.optim.AdamW(net.parameters(), lr=learning_rate)
scheduler_model = torch.optim.lr_scheduler.StepLR(opt, step_size=decay_step, gamma=decay_gamma)

#--------------------------------------------参数输出--------------------------------------------------------
total_num1 = sum(p.numel() for p in net.parameters())
trainable_num1 = sum(p.numel() for p in net.parameters() if p.requires_grad)
print_log("[{}] \tTrainable/Total: {}/{}".format('Core Denoising Model',trainable_num1, total_num1), log)
print_log("denoise step:[{}]".format(n_steps),log)

#-----------------------------------------训练相关函数--------------------------------------------------------
# 设置随机数
def prepare_seed(rand_seed):
	np.random.seed(rand_seed)
	random.seed(rand_seed)
	torch.manual_seed(rand_seed)
	torch.cuda.manual_seed_all(rand_seed) 

def make_beta_schedule(schedule: str = 'linear', 
    n_timesteps: int = 1000, 
    start: float = 1e-5, end: float = 1e-2) -> torch.Tensor:
    if schedule == 'linear':
        betas = torch.linspace(start, end, n_timesteps)
    elif schedule == "quad":
        betas = torch.linspace(start ** 0.5, end ** 0.5, n_timesteps) ** 2
    elif schedule == "sigmoid":
        betas = torch.linspace(-6, 6, n_timesteps)
        betas = torch.sigmoid(betas) * (end - start) + start
    return betas


# ------------------------------------------计算去噪过程中使用的参数----------------------------------------------------
'''
beta, alphas, alphas_prod, alphas_bar_sqrt, one_minus_alphas_bar_sqrt维度均为(n_steps,)[100,]
'''
betas = make_beta_schedule(schedule='linear', n_timesteps=n_steps, start=1e-4, end=5e-2).cuda(cuda_idx)
alphas = 1 - betas
alphas_prod = torch.cumprod(alphas, 0)
alphas_bar_sqrt = torch.sqrt(alphas_prod)
one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

def extract(input, t, x):
    shape = x.shape
    out = torch.gather(input, 0, t.to(input.device))
    reshape = [t.shape[0]] + [1] * (len(shape) - 1)
    return out.reshape(*reshape)

def noise_estimation_loss(x, y_0, mask):
    #past_traj:B,16,6;  fut_traj:B,25,2;  mask:B,B
    batch_size = x.shape[0]
    # Select a random step for each example
    t = torch.randint(0, n_steps, size=(batch_size // 2 + 1,)).to(x.device)
    t = torch.cat([t, n_steps - t - 1], dim=0)[:batch_size]
    # x0 multiplier
    a = extract(alphas_bar_sqrt, t, y_0)
    beta = extract(betas, t, y_0)
    # eps multiplier
    am1 = extract(one_minus_alphas_bar_sqrt, t, y_0)
    e = torch.randn_like(y_0)
    # model input
    y = y_0 * a + e * am1
    # y:B,25,2; beta:B,1,1; x:B,16,6; mask:B,B
    output = net(y, beta, x, mask)
    return (e - output).square().mean()

def p_sample(x, mask, cur_y, t):
    if t==0:
        z = torch.zeros_like(cur_y).to(x.device)
    else:
        z = torch.randn_like(cur_y).to(x.device)
    t = torch.tensor([t]).cuda(cuda_idx)
    # Factor to the model output
    eps_factor = ((1 - extract(alphas, t, cur_y)) / extract(one_minus_alphas_bar_sqrt, t, cur_y))
    beta = extract(betas, t.repeat(x.shape[0]), cur_y)
    eps_theta = net(cur_y, beta, x, mask)
    mean = (1 / extract(alphas, t, cur_y).sqrt()) * (cur_y - (eps_factor * eps_theta))
    # Generate z
    # z = torch.randn_like(cur_y).to(x.device)
    # Fixed sigma
    sigma_t = extract(betas, t, cur_y).sqrt()
    sample = mean + sigma_t * z
    return (sample)


def p_sample_loop( x, mask, shape):
    net.eval()
    prediction_total = torch.Tensor().cuda(cuda_idx)
    for _ in range(20):
        cur_y = torch.randn(shape).to(x.device)
        for i in reversed(range(n_steps)):
            cur_y = p_sample(x, mask, cur_y, i)
        prediction_total = torch.cat((prediction_total, cur_y.unsqueeze(1)), dim=1)
    return prediction_total

def rotate_traj(past, future, past_abs):
    # past:B,16,2; fut_traj:B,25,2; past_traj_abs:B,16,2
    past_diff = past[:, 0]  #B,2
    past_theta = torch.atan(torch.div(past_diff[:, 1], past_diff[:, 0]+1e-5))   #B
    past_theta = torch.where((past_diff[:, 0]<0), past_theta+math.pi, past_theta)

    rotate_matrix = torch.zeros((past_theta.size(0), 2, 2)).to(past_theta.device)   #B,2,2
    rotate_matrix[:, 0, 0] = torch.cos(past_theta)
    rotate_matrix[:, 0, 1] = torch.sin(past_theta)
    rotate_matrix[:, 1, 0] = - torch.sin(past_theta)
    rotate_matrix[:, 1, 1] = torch.cos(past_theta)

    past_after = torch.matmul(rotate_matrix, past.transpose(1, 2)).transpose(1, 2)
    future_after = torch.matmul(rotate_matrix, future.transpose(1, 2)).transpose(1, 2)
    past_abs = torch.matmul(rotate_matrix, past_abs.transpose(1, 2)).transpose(1, 2)
    return past_after, future_after, past_abs

#---------------------------------------------Train-------------------------------------------------------------------
train_loss = []
for epoch_num in range(trainEpochs):
    print_log("learning_rate:[{}]".format(opt.param_groups[0]['lr']), log)
    net.train()
    avg_tr_loss = 0
    count = 1
    for i, data in enumerate(train_loader):
        st_time = time.time()
        hist, _, _, _, _, fut, _, _ = data
        #hist:16,B,2; fut:25,B,2
        # use gpu
        hist = hist.cuda(cuda_idx)
        fut = fut.cuda(cuda_idx)

        targets_batch_size = hist.shape[1]
        mask = torch.zeros(targets_batch_size, targets_batch_size).cuda(cuda_idx)   #(B,B)
        for id in range(targets_batch_size):
            mask[id:id+1, id:id+1] = 1.
        past_traj_abs = hist.permute(1,0,2)     #B,16,2
        past_traj_rel = past_traj_abs - past_traj_abs[:, -1:]
        fut_traj = (fut.permute(1,0,2) - past_traj_abs[:, -1:])
        # if rotate:
        #past_traj_rel:B,16,2; fut_traj:B,25,2; past_traj_abs:B,16,2
        past_traj_rel, fut_traj, past_traj_abs = rotate_traj(past_traj_rel, fut_traj, past_traj_abs)
        past_traj_vel = torch.cat((past_traj_rel[:, 1:] - past_traj_rel[:, :-1], torch.zeros_like(past_traj_rel[:, -1:])), dim=1)
        past_traj = torch.cat((past_traj_abs, past_traj_rel, past_traj_vel), dim=-1)

        #past_traj:B,16,6;  fut_traj:B,25,2;  mask:B,B
        loss = noise_estimation_loss(past_traj, fut_traj, mask)
        avg_tr_loss += loss.item()
        count += 1

        # Backprop and update weights
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
        opt.step()

        if i%100 == 99:
            print_log('[{}] Epoch: {}\t\tLoss: {:.4f}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), epoch_num+1, avg_tr_loss/100), log)
            # wandb.log({"loss":avg_tr_loss/100, "epoch": trainEpochs})
            train_loss.append(avg_tr_loss/100)
            avg_tr_loss = 0
    if epoch_num%2 == 1:
        torch.save(net.state_dict(), './result/denoise_ngsim/denoise'+str(epoch_num+1)+'.tar')
    scheduler_model.step()
#____________________________________________________________________________________
torch.save(net.state_dict(), './result/denoise_ngsim/denoise.tar')


#Test______________________________________________________________________
#加载训练模型
torch.cuda.set_device(cuda_idx)
net = TransformerDenoisingModel().cuda(cuda_idx)
net.load_state_dict(torch.load('./result/denoise_ngsim/denoise.tar'))
ade = 0
fde = 0
samples = 0

with torch.no_grad():
    for i, data in enumerate(test_loader):
        hist, _, _, _, _, fut, _, _ = data
        # use gpu
        hist = hist.cuda(cuda_idx)
        fut = fut.cuda(cuda_idx)
        targets_batch_size = hist.shape[1]
        mask = torch.zeros(targets_batch_size, targets_batch_size).cuda(cuda_idx)
        for i in range(targets_batch_size):
            mask[i:i+1, i:i+1] = 1.
        past_traj_abs = hist.permute(1,0,2)
        past_traj_rel = past_traj_abs - past_traj_abs[:, -1:]
        fut_traj = (fut.permute(1,0,2) - past_traj_abs[:, -1:])

        past_traj_rel, fut_traj, past_traj_abs = rotate_traj(past_traj_rel, fut_traj, past_traj_abs)
        past_traj_vel = torch.cat((past_traj_rel[:, 1:] - past_traj_rel[:, :-1], torch.zeros_like(past_traj_rel[:, -1:])), dim=1)
        past_traj = torch.cat((past_traj_abs, past_traj_rel, past_traj_vel), dim=-1)
        pred_traj = p_sample_loop(past_traj, mask, fut_traj.shape)
        fut_traj = fut_traj.unsqueeze(1).repeat(1, 20, 1, 1)
        distances = torch.norm(fut_traj - pred_traj, dim=-1)
        ade += distances.mean(dim=-1).min(dim=-1)[0].sum()
        fde += (distances[:, :, -1]).min(dim=-1)[0].sum()
        samples += distances.shape[0]
print_log('--ADE: {:.4f}\t--FDE: {:.4f}'.format(ade/samples, fde/samples), log)
