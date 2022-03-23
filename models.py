import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import LocalizerEncoderOnly, \
		PositionwiseFeedForward, PositionalEncoding, EncoderLayer, \
		MultiHeadedAttention, Encoder, Generator, Embeddings

def Transpotter(vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
	c = copy.deepcopy

	attn = MultiHeadedAttention(h, d_model, dropout=dropout)
	ff = PositionwiseFeedForward(d_model, d_ff, dropout)
	position = PositionalEncoding(d_model, dropout)

	model = LocalizerEncoderOnly(
		video_encoder=Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
		text_encoder=Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N//2),
		pos_embed=c(position), text_embed=Embeddings(d_model, vocab),
		decoder=Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
		classifier=Generator(d_model), detector=Generator(d_model), 
		cls_tag=nn.Parameter(torch.randn(1, 1, d_model)),)
	
	for p in model.parameters():
		if p.dim() > 1:
			nn.init.xavier_uniform_(p)
	return model
