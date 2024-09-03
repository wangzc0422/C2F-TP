import math
import torch
import torch.nn as nn
from timm.models.layers import DropPath
import torch.nn.functional as F
from torch.nn import Module, Linear

class PositionalEncoding(nn.Module):
	def __init__(self, d_model, dropout=0.1, max_len=5000):
		super().__init__()

		self.dropout = nn.Dropout(p=dropout)
		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(
			torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
		)
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(0).transpose(0, 1)
		self.register_buffer("pe", pe)

	def forward(self, x):   # 10,128,512
		x = x + self.pe[: x.size(0), :] #10,128,512
		return self.dropout(x)

class ConcatSquashLinear(Module):
	def __init__(self, dim_in, dim_out, dim_ctx):   #"1":2,512,259   "3":512,256,259
		super(ConcatSquashLinear, self).__init__()
		self._layer = Linear(dim_in, dim_out)
		self._hyper_bias = Linear(dim_ctx, dim_out, bias=False)
		self._hyper_gate = Linear(dim_ctx, dim_out)

	def forward(self, ctx, x):
		# ctx: (B, 1, F+3)
		# x: (B, T, 2)
		gate = torch.sigmoid(self._hyper_gate(ctx))
		bias = self._hyper_bias(ctx)
		ret = self._layer(x) * gate + bias
		return ret
	
	def batch_generate(self, ctx, x):
		gate = torch.sigmoid(self._hyper_gate(ctx))
		bias = self._hyper_bias(ctx)
		ret = self._layer(x) * gate + bias
		return ret


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1)
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x   


## Phase-Attention Module卷积神经网络模块
class PATM(nn.Module):
    def __init__(self, dim, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,mode='fc'):
        super().__init__()
        
        self.fc_h = nn.Conv2d(dim, dim, 1, 1,bias=qkv_bias)
        self.fc_w = nn.Conv2d(dim, dim, 1, 1,bias=qkv_bias) 
        self.fc_c = nn.Conv2d(dim, dim, 1, 1,bias=qkv_bias)
        
        self.tfc_h = nn.Conv2d(2*dim, dim, (1,7), stride=1, padding=(0,7//2), groups=dim, bias=False) 
        self.tfc_w = nn.Conv2d(2*dim, dim, (7,1), stride=1, padding=(7//2,0), groups=dim, bias=False)  
        self.reweight = MLP(dim, dim // 4, dim * 3)
        self.proj = nn.Conv2d(dim, dim, 1, 1,bias=True)
        self.proj_drop = nn.Dropout(proj_drop)   
        self.mode = mode
        #对h和w都学出相位
        if mode=='fc':
            self.theta_h_conv=nn.Sequential(nn.Conv2d(dim, dim, 1, 1,bias=True),nn.BatchNorm2d(dim),nn.ReLU())
            self.theta_w_conv=nn.Sequential(nn.Conv2d(dim, dim, 1, 1,bias=True),nn.BatchNorm2d(dim),nn.ReLU())  
        else:
            self.theta_h_conv=nn.Sequential(nn.Conv2d(dim, dim, 3, stride=1, padding=1, groups=dim, bias=False),nn.BatchNorm2d(dim),nn.ReLU())
            self.theta_w_conv=nn.Sequential(nn.Conv2d(dim, dim, 3, stride=1, padding=1, groups=dim, bias=False),nn.BatchNorm2d(dim),nn.ReLU())                   


    def forward(self, x):
        B, C, H, W = x.shape
        # C, H, W = x.shape
        # 提取相位信息
        theta_h=self.theta_h_conv(x)
        theta_w=self.theta_w_conv(x)
        # Channel-FC提取振幅信息
        x_h=self.fc_h(x)
        x_w=self.fc_w(x)      
        # 用欧拉公式对特征进行展开
        x_h=torch.cat([x_h*torch.cos(theta_h),x_h*torch.sin(theta_h)],dim=1)
        x_w=torch.cat([x_w*torch.cos(theta_w),x_w*torch.sin(theta_w)],dim=1)
        # 特征转换
        h = self.tfc_h(x_h)
        w = self.tfc_w(x_w)
        c = self.fc_c(x)
        # 自适应平均池化和重新加权
        a = F.adaptive_avg_pool2d(h + w + c,output_size=1)
        a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(-1).unsqueeze(-1)
        # 合并特征
        x = h * a[0] + w * a[1] + c * a[2]
        x = self.proj(x)
        x = self.proj_drop(x)           
        return x
   

class WaveBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.BatchNorm2d, mode='fc'):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = PATM(dim, qkv_bias=qkv_bias, qk_scale=None, attn_drop=attn_drop,mode=mode)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x))) 
        x = x + self.drop_path(self.mlp(self.norm2(x))) 
        return x

## Batchwise MSE loss, uses mask for variable output lengths
def maskedMSE(y_pred, y_gt, mask):
    acc = torch.zeros_like(mask)
    muX = y_pred[:,:,:,0]
    muY = y_pred[:,:,:,1]
    x = y_gt[:,:,:,0]
    y = y_gt[:,:,:,1]
    out = torch.pow(x-muX, 2) + torch.pow(y-muY, 2)
    acc[:,:,:,0] = out
    acc[:,:,:,1] = out
    acc = acc*mask
    lossVal = torch.sum(acc)/torch.sum(mask)
    return lossVal


## Custom activation for output layer (Graves, 2015)
# def outputActivation(x):
#     muX = x[:,:,0:1]
#     muY = x[:,:,1:2]
#     sigX = x[:,:,2:3]
#     sigY = x[:,:,3:4]
#     rho = x[:,:,4:5]
#     sigX = torch.exp(sigX)
#     sigY = torch.exp(sigY)
#     rho = torch.tanh(rho)
#     out = torch.cat([muX, muY, sigX, sigY, rho],dim=2)
#     return out

def outputActivation(x, K= 20):
    muX = x[:,:,0:1]
    muY = x[:,:,1:2]
    sigX = x[:,:,2:3]
    sigY = x[:,:,3:4]
    rho = x[:,:,4:5]
    sigX = torch.exp(sigX)
    sigY = torch.exp(sigY)
    rho = torch.tanh(rho)
    # out = torch.cat([muX, muY, sigX, sigY, rho],dim=2)
    # 生成高斯噪声
    out = torch.zeros(x.size(0), x.size(1), K, 2)
    for i in range(K):
        noiseX = torch.randn_like(muX)
        noiseY = torch.randn_like(muY)
        x = muX + sigX * noiseX
        y = muY + sigY * (rho * noiseX + torch.sqrt(1 - rho**2) * noiseY)
        out[:,:,i,0] = x.squeeze()
        out[:,:,i,1] = y.squeeze()
    return out
