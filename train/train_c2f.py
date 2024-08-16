# coding=utf-8
import os
import torch
import time
import math
from trainer.utils import print_log
from torch.utils.data import DataLoader
from loader.ngsimloader import NgsimDataset
from model.interaction import Interaction
from model.denoise import TransformerDenoisingModel

#-----------------------------------超参数设置----------------------------------------------------
args = {}
#train
args['batch_size'] = 128
args['pretrainEpochs'] = 5
args['trainEpochs'] = 15
args['learning_rate'] = 1e-4
args['use_cuda'] = True
args['decay_step'] = 6
args['decay_gamma'] = 0.5

#denoise
args['NUM_Tau'] = 5
args['beta_schedule'] = 'linear'
args['n_steps'] = 100
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
args['train_flag'] = True
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
trSet = NgsimDataset(mat_file='./datasets/NGSIM/TrainSet.mat')
valSet = NgsimDataset(mat_file='./datasets/NGSIM/ValSet.mat')
train_loader =  DataLoader(trSet, batch_size=args['batch_size'], shuffle=True, collate_fn=trSet.collate_fn, num_workers=8, drop_last = True,pin_memory=True)
val_loader = DataLoader(valSet, batch_size=args['batch_size'], shuffle=True, collate_fn=valSet.collate_fn, num_workers=8, drop_last = True,pin_memory=True)


#------------------------------------定义模型---------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
cuda_idx=0
net = TransformerDenoisingModel().cuda(cuda_idx)
net.load_state_dict(torch.load('./result/denoise_highd/denoise.tar'))
torch.cuda.set_device(cuda_idx)
model = Interaction(args).cuda(cuda_idx)

for param in net.parameters():
    param.requires_grad = False
opt = torch.optim.AdamW(model.parameters(), lr=args['learning_rate'])
scheduler_model = torch.optim.lr_scheduler.StepLR(opt, step_size=args['decay_step'], gamma=args['decay_gamma'])
crossEnt = torch.nn.BCELoss()


#-----------------------------------训练log路径-----------------------------------------------------
log_dir = './ngsim/check/'
log = open(os.path.join(log_dir, 'tst.txt'), 'a+')

#----------------------------------模型参数输出------------------------------------------------------
total_num1 = sum(p.numel() for p in net.parameters())
trainable_num1 = sum(p.numel() for p in net.parameters() if p.requires_grad)
print_log("[{}] \tTrainable/Total: {}/{}".format('Core Denoising Model',trainable_num1, total_num1), log)
total_num2 = sum(p.numel() for p in model.parameters())
trainable_num2 = sum(p.numel() for p in model.parameters() if p.requires_grad)
print_log("[{}] \tTrainable/Total: {}/{}".format('Initialization Model',trainable_num2, total_num2), log)


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
    hist = hist.permute(1,0,2)	#B,16,2
    fut_pred = fut_pred.permute(1,2,0,3).cuda(cuda_idx)   #B,K,25,2
    batch_size = hist.shape[0]
    mask = torch.zeros(batch_size, batch_size).cuda(cuda_idx)
    for i in range(batch_size):
        mask[i:i+1, i:i+1] = 1.
    return hist, mask, fut_pred


train_loss = []
val_loss = []
prev_val_loss = math.inf

## Train:______________________________________________________________________________
for epoch_num in range(args['pretrainEpochs'] + args['trainEpochs']):
    print_log("learning_rate:[{}]".format(opt.param_groups[0]['lr']), log)
    if epoch_num == 0:
        print_log('Pre-training', log)
    elif epoch_num == args['pretrainEpochs']:
        print_log('Training', log)

    model.train()
    model.train_flag = True
    # Variables to track training performance:
    avg_tr_loss = 0
    # avg_tr_time = 0
    avg_lat_acc = 0
    avg_lon_acc = 0

    for i, data in enumerate(train_loader):
        hist, nbrs, nbmask, lat_enc, lon_enc, fut, op_mask,_ = data
        #use gpu
        nbrs = nbrs.cuda(cuda_idx)
        nbmask = nbmask.cuda(cuda_idx)
        hist = hist.cuda(cuda_idx)
        fut = fut.cuda(cuda_idx)
        lat_enc = lat_enc.cuda(cuda_idx)
        lon_enc = lon_enc.cuda(cuda_idx)
        op_mask = op_mask.cuda(cuda_idx)

        # Forward pass
        fut_pred, lat_pred, lon_pred = model(hist, nbrs, nbmask, lat_enc, lon_enc)

        hist, mask, fut_pred = data_preprocess(hist, fut_pred)
        # denoise
        y_pred = p_sample_loop_accelerate(hist, mask, fut_pred)   # B,20,25,2

        fut = fut.unsqueeze(1).repeat(1, 20, 1, 1)
        fut = fut.permute(2,1,0,3)  #B,20,25,2


        if args['use_maneuvers']:
            # Pre-train with MSE loss to speed up training
            if epoch_num < args['pretrainEpochs']:
                loss = (y_pred - fut).norm(p=2, dim=-1).mean(dim=-1).min(dim=1)[0].mean()
            else:
                # y_pred:B,25,2; fut:B,25,2
                loss = (y_pred - fut).norm(p=2, dim=-1).mean(dim=-1).min(dim=1)[0].mean() + crossEnt(lat_pred, lat_enc) + crossEnt(lon_pred, lon_enc)
                avg_lat_acc += (torch.sum(torch.max(lat_pred.data, 1)[1] == torch.max(lat_enc.data, 1)[1])).item() / lat_enc.size()[0]
                avg_lon_acc += (torch.sum(torch.max(lon_pred.data, 1)[1] == torch.max(lon_enc.data, 1)[1])).item() / lon_enc.size()[0]
        else:
            loss = (y_pred - fut).norm(p=2, dim=-1).mean(dim=-1).min(dim=1)[0].mean()


        # Backprop and update weights
        opt.zero_grad()
        loss.backward()
        a = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        opt.step()

        # Track average train loss and average train time:
        # batch_time = time.time()-st_time
        avg_tr_loss += loss.item()
        # avg_tr_time += batch_time
        
        if i%100 == 99:
            epoch_progress = i*args['batch_size']/len(trSet)
            print_log('[{}] Epoch No: {}\t Epoch propress(%): {:.2f}\t Loss: {:.4f}\t Acc: {:.4f}, {:.4f}\t Val Loss: {:.4f}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), 
                epoch_num+1, epoch_progress*100, avg_tr_loss/100, avg_lat_acc, avg_lon_acc, prev_val_loss), log)
            avg_tr_loss = 0
            avg_lat_acc = 0
            avg_lon_acc = 0
            # avg_tr_time = 0
        torch.cuda.empty_cache()

    torch.save(model.state_dict(), './two/check/interaction_'+str(epoch_num+1)+'.tar')
    scheduler_model.step()
    # _______________________________________________________________________________________________________

    #validate________________________________________________________________________________________________
    model.train_flag = False
    print_log('[{}] Epoch No: {}\t Complete, Calculating validation loss...'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), epoch_num+1),log)
    avg_val_loss = 0
    avg_val_lat_acc = 0
    avg_val_lon_acc = 0
    val_batch_count = 0
    total_points = 0

    for i, data  in enumerate(val_loader):
        net.eval()
        model.eval()
        hist, nbrs, nbmask, lat_enc, lon_enc, fut, op_mask,_ = data
        # use gpu
        nbrs = nbrs.cuda(cuda_idx)
        nbmask = nbmask.cuda(cuda_idx)
        hist = hist.cuda(cuda_idx)
        fut = fut.cuda(cuda_idx)
        lat_enc = lat_enc.cuda(cuda_idx)
        lon_enc = lon_enc.cuda(cuda_idx)
        op_mask = op_mask.cuda(cuda_idx)

        # Forward pass
        if args['use_maneuvers']:
            fut_pred, lat_pred, lon_pred = model(hist, nbrs, nbmask, lat_enc, lon_enc)
            fut_pred_max = torch.zeros_like(fut_pred[0])
            for k in range(lat_pred.shape[0]):
                lat_man = torch.argmax(lat_pred[k, :]).detach()
                lon_man = torch.argmax(lon_pred[k, :]).detach()
                indx = lon_man*3 + lat_man
                fut_pred_max[:,k,:] = fut_pred[indx][:,k,:] 
            hist, mask, fut_pred_max = data_preprocess(hist,fut_pred_max)
            # Denoise
            y_pred = p_sample_loop_accelerate(hist, mask, fut_pred_max)
        else:
            fut_pred, _, _ = model(hist, nbrs, nbmask, lat_enc, lon_enc)
            hist, mask, fut_pred = data_preprocess(hist,fut_pred)
            # Denoise
            y_pred = p_sample_loop_accelerate(hist, mask, fut_pred)

        fut = fut.unsqueeze(1).repeat(1, 20, 1, 1)
        fut = fut.permute(2,1,0,3)  #B,20,25,2
        op_mask = op_mask.unsqueeze(1).repeat(1, 20, 1, 1)
        op_mask = op_mask.permute(2,1,0,3)  #B,20,25,2

        y_pred = y_pred*op_mask

        # Forward pass
        if args['use_maneuvers']:
            # Pre-train with MSE loss to speed up training
            if epoch_num < args['pretrainEpochs']:
                loss = (y_pred - fut).norm(p=2, dim=-1).mean(dim=-1).min(dim=1)[0].mean()
            else:
                # y_pred:B,25,2; fut:B,25,2
                loss = (y_pred - fut).norm(p=2, dim=-1).mean(dim=-1).min(dim=1)[0].mean() + crossEnt(lat_pred, lat_enc) + crossEnt(lon_pred, lon_enc)
                avg_lat_acc += (torch.sum(torch.max(lat_pred.data, 1)[1] == torch.max(lat_enc.data, 1)[1])).item() / lat_enc.size()[0]
                avg_lon_acc += (torch.sum(torch.max(lon_pred.data, 1)[1] == torch.max(lon_enc.data, 1)[1])).item() / lon_enc.size()[0]
        else:
            loss = (y_pred - fut).norm(p=2, dim=-1).mean(dim=-1).min(dim=1)[0].mean()

        avg_val_loss += loss.item()
        val_batch_count += 1
    print_log('Validation loss: {:.4f}\t Val Acc: {:.4f},{:.4f}'.format(avg_val_loss/val_batch_count, avg_val_lat_acc/val_batch_count*100, avg_val_lon_acc/val_batch_count*100),log)
    val_loss.append(avg_val_loss/val_batch_count)
    prev_val_loss = avg_val_loss/val_batch_count
    # _________________________________________________________________________________________________________________

torch.save(model.state_dict(), './two/check/initial.tar')
