import math
import torch
import torch.nn as nn
from torch.nn import Module
from layers import PositionalEncoding, ConcatSquashLinear


class st_encoder(nn.Module):
	def __init__(self):
		super().__init__()
		channel_in = 2
		channel_out = 32
		dim_kernel = 3
		self.dim_embedding_key = 256
		# 空间卷积层
		self.spatial_conv = nn.Conv1d(channel_in, channel_out, dim_kernel, stride=1, padding=1)
		# 时间序列编码器
		self.temporal_encoder = nn.GRU(channel_out, self.dim_embedding_key, 1, batch_first=True)
		self.relu = nn.ReLU()
		self.reset_parameters()

	def reset_parameters(self):
		# 初始化权重
		nn.init.kaiming_normal_(self.spatial_conv.weight)
		nn.init.kaiming_normal_(self.temporal_encoder.weight_ih_l0)
		nn.init.kaiming_normal_(self.temporal_encoder.weight_hh_l0)
		# 偏置初始化为0
		nn.init.zeros_(self.spatial_conv.bias)
		nn.init.zeros_(self.temporal_encoder.bias_ih_l0)
		nn.init.zeros_(self.temporal_encoder.bias_hh_l0)

	def forward(self, X):
		X_t = torch.transpose(X, 1, 2)
		X_after_spatial = self.relu(self.spatial_conv(X_t))
		X_embed = torch.transpose(X_after_spatial, 1, 2)
		output_x, state_x = self.temporal_encoder(X_embed)
		state_x = state_x.squeeze(0)

		return state_x


class social_transformer(nn.Module):
	def __init__(self):
		super(social_transformer, self).__init__()
		self.encode_past = nn.Linear(32, 256, bias=False)
		self.layer = nn.TransformerEncoderLayer(d_model=256, nhead=2, dim_feedforward=256)
		self.transformer_encoder = nn.TransformerEncoder(self.layer, num_layers=2)

	def forward(self, h, mask):	
		h_feat = self.encode_past(h.reshape(h.size(0), -1)).unsqueeze(1)
		h_feat_ = self.transformer_encoder(h_feat, mask)
		h_feat = h_feat + h_feat_
		return h_feat


class soc_transformer(nn.Module):
	def __init__(self):
		super(soc_transformer, self).__init__()
		self.encode_past = nn.Linear(96, 256, bias=False)
		self.layer = nn.TransformerEncoderLayer(d_model=256, nhead=2, dim_feedforward=256)
		self.transformer_encoder = nn.TransformerEncoder(self.layer, num_layers=2)

	def forward(self, h, mask):
		h_feat = self.encode_past(h.reshape(h.size(0), -1)).unsqueeze(1)
		h_feat_ = self.transformer_encoder(h_feat, mask)
		h_feat = h_feat + h_feat_

		return h_feat

# transformer去噪模型
class TransformerDenoisingModel(Module):

	def __init__(self, context_dim=256, tf_layer=2):
		super().__init__()
		self.encoder_context1 = social_transformer()
		self.encoder_context2 = soc_transformer()
		self.pos_emb = PositionalEncoding(d_model=2*context_dim, dropout=0.1, max_len=50)
		self.concat1 = ConcatSquashLinear(2, 2*context_dim, context_dim+3)
		self.layer = nn.TransformerEncoderLayer(d_model=2*context_dim, nhead=2, dim_feedforward=2*context_dim)
		self.transformer_encoder = nn.TransformerEncoder(self.layer, num_layers=tf_layer)
		self.concat3 = ConcatSquashLinear(2*context_dim,context_dim,context_dim+3)
		self.concat4 = ConcatSquashLinear(context_dim,context_dim//2,context_dim+3)
		self.linear = ConcatSquashLinear(context_dim//2, 2, context_dim+3)


	def forward(self, y, beta, x, mask):
		beta = beta.view(beta.size(0), 1, 1)
		mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
		x = self.encoder_context2(x, mask)
		time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)
		ctx_emb = torch.cat([time_emb, x], dim=-1)
		
		y = self.concat1(ctx_emb, y)
		final_emb = y.permute(1,0,2)
		final_emb = self.pos_emb(final_emb)
		
		trans = self.transformer_encoder(final_emb).permute(1,0,2)
		trans = self.concat3(ctx_emb, trans)
		trans = self.concat4(ctx_emb, trans)
		return self.linear(ctx_emb, trans)
	

	def generate_accelerate(self, cur_y, beta, hist, mask):
		batch_size = cur_y.size(0)
		beta = beta.view(beta.size(0), 1, 1)
		mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
		hist = self.encoder_context1(hist, mask)
		time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)
		ctx_emb = torch.cat([time_emb, hist], dim=-1).repeat(1, 10, 1).unsqueeze(2)
		cur_y = self.concat1.batch_generate(ctx_emb, cur_y).contiguous().view(-1, 25, 512)
		final_emb = cur_y.permute(1, 0, 2)
		final_emb = self.pos_emb(final_emb)
		trans = self.transformer_encoder(final_emb).permute(1, 0, 2).contiguous().view(-1, 10, 25, 512)
		trans = self.concat3.batch_generate(ctx_emb, trans)
		trans = self.concat4.batch_generate(ctx_emb, trans)
		
		return self.linear.batch_generate(ctx_emb, trans)
