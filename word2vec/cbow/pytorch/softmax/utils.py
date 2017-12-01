# !/usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------------------------------#
# model utils		                     #
# author: sean lee                           #
# locate: Shanxi university, Taiyuan, China  #
# email: lxm_0828@163.com                    #
#--------------------------------------------#

import torch
import os, glob
import numpy as np

class Utils(object):

	@staticmethod
	def save_model(model, epoch, save_dir, max_keep=5):
		if not os.path.exists(save_dir):
			os.makedirs(save_dir)
		f_list = glob.glob(os.path.join(save_dir, 'model') + '_*.ckpt')
		if len(f_list) >= max_keep + 2:
			epoch_list = [int(i.split('_')[-1].split('.')[0]) for i in f_list]
			to_delete = [f_list[i] for i in np.argsort(epoch_list)[-max_keep:]]
			for f in to_delete:
				os.remove(f)
		name = 'model_{}.ckpt'.format(epoch)
		file_path = os.path.join(save_dir, name)
		#torch.save(model.state_dict(), file_path)
		torch.save(model, file_path)

	@staticmethod
	def load_previous_model(save_dir):
		if not os.path.exists(save_dir):
			os.makedirs(save_dir)
		f_list = glob.glob(os.path.join(save_dir, 'model') + '_*.ckpt')
		start_epoch = 1
		model = None
		if len(f_list) >= 1:
			epoch_list = [int(i.split('_')[-1].split('.')[0]) for i in f_list]
			last_checkpoint = f_list[np.argmax(epoch_list)]
			if os.path.exists(last_checkpoint):
				#print('load from {}'.format(last_checkpoint))
				# CNN 不支持参数保存
				#model.load_state_dict(torch.load(last_checkpoint))
				model = torch.load(last_checkpoint)
				start_epoch = np.max(epoch_list)
		return model, start_epoch

	@staticmethod
	def remove_models(save_dir):
		f_list = glob.glob(os.path.join(save_dir, 'model') + '_*.ckpt')
		f_list.append(os.path.join(save_dir, 'param.pkl'))
		f_list.append(os.path.join(save_dir, 'log.txt'))
		for filename in f_list:
			try:
				os.remove(filename)
			except:
				pass
