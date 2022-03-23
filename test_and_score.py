import numpy as np
import torch, os, copy, pickle
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from config import load_args
from models import Transpotter
from utils import load, precision, recall, accuracy, batch_IOU
from torch.cuda.amp import autocast

from icecream import ic

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

args = load_args()

from dataloader_test import FeatsDataset, pad_texts

def init(args):
	test_iter = FeatsDataset(args, mode='test')
	model = Transpotter(len(test_iter.v), N=args.num_blocks, 
							d_model=args.hidden_units, h=args.num_heads, 
							dropout=args.dropout_rate)
	
	return model.to(args.device).eval(), test_iter

@autocast()
def get_preds(model, feats, words, feats_mask, words_mask):
	if args.localization:
		out, loc_out = model(feats, words, feats_mask, words_mask)[:2]
		loc_out = torch.sigmoid(loc_out.squeeze(-1)).detach()
	else:
		out = model(feats, words, feats_mask, words_mask)
		if isinstance(out, tuple):
			out = out[0]

		loc_out = None

	out = torch.sigmoid(out.squeeze(-1)).detach().cpu()

	return out, loc_out

def run_epoch(data_iter, model):
	num_steps = len(data_iter.fpaths)

	prog_bar = tqdm(range(num_steps))
	words, words_mask = pad_texts([data_iter.w2p[w] for w in data_iter.words])
	words = words.to(args.device)
	words_mask = words_mask.to(args.device)
	sims = []
	threshed_ious = []
	loc_probs = []
	chunk_size = len(words) // 4 # in order to fit on a 12G GPU, can increase if you have more GPU memory
	for i in prog_bar:
		feats, bounds_list = data_iter[i]
		feats = feats.unsqueeze(0).to(args.device)
		feats_mask = torch.ones(1, 1, feats.size(1)).to(args.device).bool()

		with autocast():
			with torch.no_grad():
				# split into halves other-wise OOM
				out1, loc_out1 = get_preds(model, feats, words[:chunk_size], 
											feats_mask, words_mask[:chunk_size])
				out2, loc_out2 = get_preds(model, feats, words[chunk_size:2*chunk_size], 
										feats_mask, words_mask[chunk_size:2*chunk_size])
				out3, loc_out3 = get_preds(model, feats, words[2*chunk_size:], 
										feats_mask, words_mask[2*chunk_size:])
				out = torch.cat([out1, out2, out3], dim=0)
				if args.localization:
					loc_out = torch.cat([loc_out1, loc_out2, loc_out3], dim=0)

		sims.append(out)

		if args.localization:
			gt_loc = torch.zeros_like(feats_mask).squeeze(1).repeat(len(loc_out), 
																1).long() # (num_words x T)
			for j, w in enumerate(data_iter.words):
				if w in bounds_list:
					s, e = bounds_list[w]

					gt_loc[j][s : e] = 1
			
			loc_out_binary = copy.deepcopy(loc_out)
			loc_out_binary[loc_out_binary >= args.loc_thresh] = 1.
			loc_out_binary[loc_out_binary < args.loc_thresh] = 0.
			ious = batch_IOU(gt_loc, loc_out_binary.long())

			threshed_ious.append(ious >= args.iou_thresh)
			loc_probs.append(loc_out.max(dim=1).values)

	mAP = []
	sims = torch.stack(sims, dim=0).transpose(0, 1) # num_words x files
	retrievals = torch.argsort(sims, dim=1, descending=True).cpu().numpy() # num_words x files

	# np.save('retrievals.npy', retrievals) # uncomment if you want to save and visualize later.

	if not args.localization:	
		for k in [1, 5, 10]:
			r, acc, mAP = 0., 0., 0.
			for w, predicted in zip(data_iter.words, retrievals):
				gt = data_iter.w2v[w]

				r += recall(gt, predicted, k=k)
				a = accuracy(gt, predicted, k=k)
				acc += a

				if k == 1: # does not depend on k, so we will do it once only

					binary_gt = np.zeros(len(predicted))
					binary_gt[gt] = 1

					mAP += precision(binary_gt, predicted, denom=len(gt))

			r /= len(retrievals)
			acc /= len(retrievals)
			if k == 1:
				mAP /= len(retrievals)
				print('mAP: {}'.format(mAP))

			print('R @ {} : {}'.format(k, r))
			print('Acc @ {} : {}'.format(k, acc))

	else:
		threshed_ious = torch.stack(threshed_ious, dim=0).transpose(0, 
												1).bool() # num_words x files
		loc_probs = torch.stack(loc_probs, dim=0).transpose(0, 1) # num_words x files

		for k in [1, 5, 10]:
			r, acc, mAP = 0., 0., 0.
			for w, ranked_files, is_localized_in_file in zip(data_iter.words, retrievals, 
																threshed_ious):
				gt = set(data_iter.w2v[w])
				num_positives = len(gt)

				# index i directly corresponds to file i
				for i in range(len(is_localized_in_file)):
					if not is_localized_in_file[i]:
						gt.discard(i)
				
				gt = list(gt)

				r += recall(gt, ranked_files, k=k, denom=num_positives) if len(gt) > 0 else 0
				acc += accuracy(gt, ranked_files, k=k)

				if k == 1: # does not depend on k, so we will do it once only
					binary_gt = np.zeros(len(is_localized_in_file))
					binary_gt[gt] = 1

					mAP += precision(binary_gt, ranked_files, denom=num_positives)
			
			r /= len(retrievals)
			acc /= len(retrievals)
			if k == 1:
				mAP /= len(retrievals)
				print('mAP: {}'.format(mAP))
			print('R @ {} : {}'.format(k, r))
			print('Acc @ {} : {}'.format(k, acc))

def main(args):
	model, test_loader = init(args)

	assert args.ckpt_path is not None, 'Specify a trained checkpoint!'
	print('Resuming from: {}'.format(args.ckpt_path))
	model = load(model, args.ckpt_path, device=args.device)[0]

	run_epoch(test_loader, model)

if __name__ == '__main__':
	main(args)