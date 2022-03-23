import torch, os, pickle, cv2, random, re

import pandas as pd

import numpy as np
from config import load_args, list_of_phones

args = load_args()

class Batch:
	"Object for holding a batch of data with mask."
	def __init__(self, video, word=None, word_mask=None, pad=0, device='cuda'):
		## video : (B, 512, T)

		self.video = video.to(device) # (B, T, 512)

		self.video_mask = (video.sum(1) != pad).unsqueeze(-2).to(device) # (B, 1, T)
		self.device = device

		if word is not None:
			self.word = word.to(device)
			self.word_mask = word_mask.unsqueeze(-2).to(device)

class FeatsDataset(object):
	def __init__(self, args, mode='test'):

		self.fpaths = []
		self.texts = []
		self.lens = []

		discarded = 0
		self.phones = list_of_phones

		blank_tokens = ['<pad>']
		blank_tokens += [' ']
		self.space_idx = 1
		self.v = blank_tokens + self.phones
		self.v2i = {self.v[i]: i for i in range(len(self.v))}
		self.i2v = {i : v for v, i in self.v2i.items()}

		self.phonedict = {}
		with open('checkpoints/cmudict.dict') as cmudict:
			for line in cmudict:
				if '(' in line or '#' in line: continue
				parts = line.strip().split()
				word, phones = parts[0].strip(), parts[1:]
				self.phonedict[word] = phones

		if mode == 'real_world_inference': return

		print('Loading from: {}'.format(args.test_pkl_file))
		with open(args.test_pkl_file, 'rb') as f:
			data = pickle.load(f)
			self.fpaths = [f for f in data.keys() if data[f][1] is not None]
			self.texts = [data[f][0] for f in self.fpaths if data[f][1] is not None]
			self.boundaries = [data[f][1] for f in self.fpaths if data[f][1] is not None]
		
		print('Testng with {} files'.format(len(self.fpaths)))

		self.w2v = {} # dictionary that maps words to videos
		self.w2p = {} # maps words to phone seqs
		self.boundaries_list = []
		for i, f in enumerate(self.fpaths):
			wb = self.boundaries[i]
			b = {w : (s, e) for s, e, w in wb if w in self.phonedict}
			self.boundaries_list.append(b)

			words = list(b.keys())
			for w in words:
				phonelist = self.phonedict[w]
				if len(phonelist) < args.thresh: continue

				if w in self.w2p:
					self.w2v[w].append(i)
				else:
					pt = torch.LongTensor(np.array([self.v2i[p] for p in phonelist]))
					self.w2p[w] = pt
					self.w2v[w] = [i]

		self.words = list(self.w2p.keys())
		print('Number of query words: {}'.format(len(self.words)))

		print('Number of instances: {}'.format(sum([len(v) for k, v in self.w2v.items()])))

		self.data_root = args.data_root

	def load_feats_file(self, fpath):
		feats = np.load(fpath)
		feats = torch.FloatTensor(feats)
		return feats

	def __len__(self):
		return len(self.fpaths)

	def __getitem__(self, idx):
		feats = self.load_feats_file('{}/{}.npy'.format(self.data_root, self.fpaths[idx]))
		return feats, self.boundaries_list[idx]

def pad_texts(texts):
	padded_texts = []
	text_mask = []
	max_len = max([len(t) for t in texts])
	for f in texts:
		if f.size(0) == max_len:
			padded_texts.append(f)
			text_mask.append(torch.ones(len(f)))
		else:
			padded_texts.append(torch.cat([f, torch.zeros(max_len - f.size(0))], dim=0))
			text_mask.append(torch.cat([torch.ones(len(f)), torch.zeros(max_len - len(f))]))

	padded_texts = torch.stack(padded_texts, dim=0)
	text_mask = torch.stack(text_mask, dim=0).bool().unsqueeze(1)
	return padded_texts.long(), text_mask
