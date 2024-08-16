# coding=utf-8
import torch as t
from torch import nn
import math
import numpy as np
import torch.nn.functional as F
from layers import WaveBlock, outputActivation
from einops import repeat


class Interaction(nn.Module):
    def __init__(self, args):
        super(Interaction, self).__init__()

        # unpack arguments
        self.args = args

        # use gpu
        self.use_cuda = args['use_cuda']

        # use multimodal
        self.use_maneuvers = args['use_maneuvers']

        # train or not
        self.train_flag = args['train_flag']

        # Sizes of network layers
        self.traj_linear_hidden = args['traj_linear_hidden']
        self.lstm_encoder_size = args['lstm_encoder_size']
        self.encoder_size = args['encoder_size']
        self.relu_param = args['relu']
        self.n_head = args['n_head']
        self.att_out = args['att_out']
        self.dropout = args['dropout']
        self.in_length = args['in_length']
        self.out_length = args['out_length']
        self.dropout = args['dropout']
        self.lat_length = args['lat_length']
        self.lon_length = args['lon_length']
        self.use_mse = args['use_mse']
        self.use_true_man = args['use_true_man']
        self.Decoder = Decoder(args=args)
      
        # define network
        self.linear = nn.Linear(2, self.traj_linear_hidden)
        self.lstm = nn.LSTM(self.traj_linear_hidden, self.lstm_encoder_size)
        # Activations:
        self.activation = nn.ELU()

        # social pooling
        self.soc_emb=WaveBlock(self.encoder_size)

        # glu
        self.glu = GLU(self.n_head*self.att_out, self.lstm_encoder_size, self.dropout)

        # time attention
        self.qt = nn.Linear(self.lstm_encoder_size, self.n_head * self.att_out)
        self.kt = nn.Linear(self.lstm_encoder_size, self.n_head * self.att_out)
        self.vt = nn.Linear(self.lstm_encoder_size, self.n_head * self.att_out)
          
        # Addnorm
        self.addAndNorm = AddAndNorm(self.lstm_encoder_size)

        # mapping
        self.mu_fc1 = t.nn.Linear(self.lstm_encoder_size, self.n_head * self.att_out)
        self.mu_fc = t.nn.Linear(self.n_head * self.att_out, self.lstm_encoder_size)
        self.mapping = t.nn.Parameter(t.Tensor(self.in_length, self.out_length, self.lat_length + self.lon_length))
        self.op_lat = t.nn.Linear(self.lstm_encoder_size, self.lat_length)
        self.op_lon = t.nn.Linear(self.lstm_encoder_size, self.lon_length)
        self.normalize = nn.LayerNorm(self.lstm_encoder_size)
    
    
    def forward(self, hist,nbrs,mask, lat_enc,lon_enc):
        # motion encoder---------------------------
        # ego
        hist_enc = self.activation(self.linear(hist))
        hist_hidden_enc, (_, _) = self.lstm(hist_enc)
        hist_hidden_enc = hist_hidden_enc.permute(1, 0, 2)
        # nbrs
        nbrs_enc = self.activation(self.linear(nbrs))
        nbrs_hidden_enc, (_, _) = self.lstm(nbrs_enc)
        #--------------------------------------------------------

        # social Pooling---------------------------
        mask = mask.permute(0,3,2,1)
        mask = repeat(mask, 'b g h w -> b t g h w', t=self.in_length)   #128,64,13,3 -> 128,16,64,13,3
        mask = mask.reshape(mask.size(0)*mask.size(1),  mask.size(2), mask.size(3), mask.size(4))# 128*16,64,13,3
        soc_enc = t.zeros_like(mask).float()
        soc_enc = soc_enc.masked_scatter_(mask, nbrs_hidden_enc)
        soc_enc = self.soc_emb(soc_enc) 
        soc_enc = soc_enc[:,:,6,1]    #128*16,64
        # hist:128,16,64
        B = hist.size(1)
        T = hist.size(0)
        soc_enc = soc_enc.reshape(B,T, soc_enc.size(1))
        values = soc_enc
        #--------------------------------------------------------

        # temporal capture---------------------------
        qt = t.cat(t.split(self.qt(values), int((self.n_head * self.att_out) / self.n_head), -1), 0)
        kt = t.cat(t.split(self.kt(values), int((self.n_head * self.att_out) / self.n_head), -1), 0).permute(0, 2, 1)
        vt = t.cat(t.split(self.vt(values), int((self.n_head * self.att_out) / self.n_head), -1), 0)
        a = t.matmul(qt, kt)
        a /= math.sqrt(self.lstm_encoder_size)
        a = t.softmax(a, -1)
        values = t.matmul(a, vt)
        values = t.cat(t.split(values, int(hist.shape[1]), 0), -1)
        #--------------------------------------------------------
        #gate
        time_values, _ = self.second_glu(values)
        values = self.addAndNorm(hist_hidden_enc, time_values)

        # mapping
        maneuver_state = values[:, -1, :]
        maneuver_state = self.activation(self.mu_fc1(maneuver_state))
        maneuver_state = self.activation(self.normalize(self.mu_fc(maneuver_state)))
        lat_pred = F.softmax(self.op_lat(maneuver_state), dim=-1)
        lon_pred = F.softmax(self.op_lon(maneuver_state), dim=-1)
        
        # 多模态
        if self.train_flag:
            if self.use_true_man:
                lat_man = t.argmax(lat_enc, dim=-1).detach()
                lon_man = t.argmax(lon_enc, dim=-1).detach()
            else:
                lat_man = t.argmax(lat_pred, dim=-1).detach().unsqueeze(1)
                lon_man = t.argmax(lon_pred, dim=-1).detach().unsqueeze(1)
                lat_enc_tmp = t.zeros_like(lat_pred)
                lon_enc_tmp = t.zeros_like(lon_pred)
                lat_man = lat_enc_tmp.scatter_(1, lat_man, 1)
                lon_man = lon_enc_tmp.scatter_(1, lon_man, 1)
            index = t.cat((lat_man, lon_man), dim=-1).permute(-1, 0)
            mapping = F.softmax(t.matmul(self.mapping, index).permute(2, 1, 0), dim=-1)
            dec = t.matmul(mapping, values).permute(1, 0, 2)
            if self.use_maneuvers:
                fut_pred = self.Decoder(dec, lat_enc, lon_enc)
                return fut_pred, lat_pred, lon_pred
            else:
                fut_pred = self.Decoder(dec, lat_pred, lon_pred)
                return fut_pred, lat_pred, lon_pred
        else:
            out = []
            for k in range(self.lon_length):
                for l in range(self.lat_length):
                    lat_enc_tmp = t.zeros_like(lat_enc)
                    lon_enc_tmp = t.zeros_like(lon_enc)
                    lat_enc_tmp[:, l] = 1
                    lon_enc_tmp[:, k] = 1
                    index = t.cat((lat_enc_tmp, lon_enc_tmp), dim=-1).permute(-1, 0)
                    mapping = F.softmax(t.matmul(self.mapping, index).permute(2, 1, 0), dim=-1)
                    dec = t.matmul(mapping, values).permute(1, 0, 2)
                    fut_pred = self.Decoder(dec, lat_enc_tmp, lon_enc_tmp)
                    out.append(fut_pred)
            return out, lat_pred, lon_pred  



class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.relu_param = args['relu']
        self.use_elu = args['use_elu']
        self.use_maneuvers = args['use_maneuvers']
        self.in_length = args['in_length']
        self.out_length = args['out_length']
        self.encoder_size = args['lstm_encoder_size']
        self.n_head = args['n_head']
        self.att_out = args['att_out']
        self.lon_length = args['lon_length']
        self.lat_length = args['lat_length']
        self.activation = nn.ELU()

        self.lstm = t.nn.LSTM(self.encoder_size, self.encoder_size)
        self.op = nn.Linear(self.encoder_size, 5)
        self.dec_linear = nn.Linear(self.encoder_size + self.lat_length + self.lon_length, self.encoder_size)

    def forward(self, dec, lat_enc, lon_enc):
        if self.use_maneuvers:
            lat_enc = lat_enc.unsqueeze(1).repeat(1, self.out_length, 1).permute(1, 0, 2)
            lon_enc = lon_enc.unsqueeze(1).repeat(1, self.out_length, 1).permute(1, 0, 2)
            dec = t.cat((dec, lat_enc, lon_enc), -1)
            dec = self.dec_linear(dec)
        h_dec, _ = self.lstm(dec)
        fut_pred = self.op(h_dec)
        return outputActivation(fut_pred)
    

class AddAndNorm(nn.Module):
    def __init__(self, hidden_layer_size):
        super(AddAndNorm, self).__init__()
        self.normalize = nn.LayerNorm(hidden_layer_size)

    def forward(self, x1, x2):
        x = t.add(x1, x2)
        return self.normalize(x)
    

# gate
class GLU(nn.Module):
    # Gated Linear Unit+
    def __init__(self, input_size, hidden_layer_size, dropout_rate=None):
        super(GLU, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.dropout_rate = dropout_rate
        if dropout_rate is not None:
            self.dropout = nn.Dropout(self.dropout_rate)
        self.activation_layer = t.nn.Linear(input_size, hidden_layer_size)
        self.gated_layer = t.nn.Linear(input_size, hidden_layer_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if self.dropout_rate is not None:
            x = self.dropout(x)
        activation = self.activation_layer(x)
        gated = self.sigmoid(self.gated_layer(x))
        return t.mul(activation, gated), gated
