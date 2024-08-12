from torch.utils.data import DataLoader
import loader.ngsimloader as lo
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
import os
from evaluate import Evaluate
from config import *


def main():
    args['train_flag'] = True
    evaluate = Evaluate()
    model = Sampling(args).to(device)
    model.train()

    # load dataset
    if dataset == "ngsim":
        t1 = lo.ngsimDataset('./datasets/NGSIM/TrainSet.mat')
        trainDataloader = DataLoader(t1, batch_size=args['batch_size'], shuffle=True, num_workers=args['num_worker'], collate_fn=t1.collate_fn) 
    else:
        t1 = lo.highdDataset('./datasets/highD/TrainSet.mat')
        trainDataloader = DataLoader(t1, batch_size=args['batch_size'], shuffle=True, num_workers=args['num_worker'], collate_fn=t1.collate_fn)  

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ExponentialLR(model, gamma=0.6)

    # train
    for epoch in range(args['epoch']):
        with open('./result/ngsim/tst.txt', 'a') as f:
            print("Epoch",epoch+1,'lr',optimizer.param_groups[0]['lr'],file=f)
        loss_gi1 = 0
        for idx, data in enumerate(tqdm(trainDataloader)):

            hist, nbrs, mask, lat_enc, lon_enc,fut,op_mask,_,_,_,_=data
            hist = hist.to(device)
            nbrs = nbrs.to(device)
            mask = mask.to(device)
            lat_enc = lat_enc.to(device)
            lon_enc = lon_enc.to(device)
            fut = fut[:args['out_length'], :, :]
            fut = fut.to(device)
            op_mask = op_mask[:args['out_length'], :, :]
            op_mask = op_mask.to(device)    #values:B,16,64; lat_enc:B,3;  lon_enc:B,2

            if args['use_maneuvers']:
                fut_pred, lat_pred, lon_pred= model(hist,nbrs,mask, lat_enc,lon_enc)
                if epoch < args['pre_epoch']:
                    loss_g1 = lo.MSELoss2(fut_pred, fut, op_mask)
                else:
                    loss_g1 = lo.maskedNLL(fut_pred, fut, op_mask)
                loss_gx_3 = lo.CELoss(lat_pred, lat_enc)
                loss_gx_2 = lo.CELoss(lon_pred, lon_enc)
                loss_gx = loss_gx_3 + loss_gx_2
                loss_g = loss_g1 + 1 * loss_gx
            else:
                fut_pred, lat_pred, lon_pred= model(hist,nbrs,mask, lat_enc,lon_enc)
                if epoch < args['pre_epoch']:
                    loss_g1 = lo.MSELoss2(fut_pred, fut, op_mask)
                else:
                    loss_g1 = lo.maskedNLL(fut_pred, fut, op_mask)
                loss_g = loss_g1
            
            optimizer.zero_grad()
            loss_g.backward()
            a = t.nn.utils.clip_grad_norm_(model.parameters(), 10)

            optimizer.step()
            loss_gi1 += loss_g1.item()

            if idx % 10000 == 9999:
                with open('./result/ngsim/tst.txt', 'a') as f:
                    print('mse:', loss_gi1 / 10000,file=f)
                loss_gi1 = 0

        save_model(name=str(epoch + 1), model = model, path = args['path'])
        evaluate.main(name=str(epoch + 1), val=True)
        scheduler.step()

# save model
def save_model(name, model, path):
    l_path = args['path']
    if not os.path.exists(l_path):
        os.makedirs(l_path)
    t.save(model.state_dict(), l_path + '/epoch' + name + '.tar')


if __name__ == '__main__':
    main()
