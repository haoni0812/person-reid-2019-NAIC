import torchvision.transforms as transforms

import argparse
import os
import ast

from core import Loaders, Base, train_an_epoch, test, visualize_ranking_list, evaluate
from tools import make_dirs, Logger, os_walk, time_now


def main(config):

	# init loaders and base
	loaders = Loaders(config)
	base = Base(config, loaders)

	# make directions
	make_dirs(base.output_path)
	make_dirs(base.save_model_path)
	make_dirs(base.save_logs_path)
	make_dirs(base.save_visualize_comp_official_path)
	make_dirs(base.save_visualize_comp_val_path)

	# init logger
	logger = Logger(os.path.join(os.path.join(config.output_path, 'logs/'), 'log.txt'))
	logger('\n'*3)
	logger(config)


	if config.mode == 'train':  # train mode

		# resume model from the resume_train_epoch
		if config.resume_train_epoch >= 0:
			base.resume_model(config.resume_train_epoch)
			start_train_epoch = config.resume_train_epoch
		else:
			start_train_epoch = 0

		# automatically resume model from the latest one
		if config.auto_resume_training_from_lastest_steps:
			root, _, files = os_walk(base.save_model_path)
			if len(files) > 0:
				# get indexes of saved models
				indexes = []
				for file in files:
					indexes.append(int(file.replace('.pkl', '').split('_')[-1]))
				indexes = sorted(list(set(indexes)), reverse=False)
				# resume model from the latest model
				base.resume_model(indexes[-1])

				start_train_epoch = indexes[-1]
				# start_train_epoch = 0
				logger('Time: {}, automatically resume training from the latest step (model {})'.format(time_now(), indexes[-1]))

		# main loop
		for current_epoch in range(start_train_epoch, config.total_train_epochs):

			# save model
			base.save_model(current_epoch)

			# train
			base.lr_scheduler.step(current_epoch)
			_, results = train_an_epoch(config, base, loaders)
			logger('Time: {};  Epoch: {};  {}'.format(time_now(), current_epoch, results))

			# test
			# if (current_epoch+1) % 40 == 0 and current_epoch+1 >= 0:
			# 	comp_map, comp_rank = test(config, base, loaders, 'comp_val')
			# 	logger('Time: {},  Dataset: Comp  \nmAP: {} \nRank: {}'.format(time_now(), comp_map, comp_rank))
			# 	logger('')


	elif config.mode == 'test':	# test mode
		# resume from the resume_test_epoch
		if config.resume_test_epoch >= 0:
			base.resume_model(config.resume_test_epoch)
		# test
		comp_map, comp_rank = test(config, base, loaders, 'comp_val')
		logger('Time: {},  Dataset: Comp  \nmAP: {} \nRank: {}'.format(time_now(), comp_map, comp_rank))
		logger('')


	elif config.mode == 'visualize': # visualization mode
		# resume from the resume_visualize_epoch
		if config.resume_visualize_epoch >= 0:
			base.resume_model(config.resume_visualize_epoch)
		# visualization
		# visualize_ranking_list(config, base, loaders, 'comp_official')
		# visualize_ranking_list(config, base, loaders, 'comp_val')
		evaluate(config, base, loaders, 'comp_official')



if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	#
	parser.add_argument('--cuda', type=str, default='cuda')
	parser.add_argument('--mode', type=str, default='train', help='train, test or visualize')
	parser.add_argument('--output_path', type=str, default='output/4768_one_ibn101a_rankedloss/', help='path to save related informations')

	# dataset configuration
	parser.add_argument('--comp_path', type=str, default='/home/')
	parser.add_argument('--train_dataset', type=str, default='comp_official_train',
						help='comp_official_train, comp_val_train')
	parser.add_argument('--image_size', type=int, nargs='+', default=[256, 128])
	parser.add_argument('--p', type=int, default=16, help='person count in a batch')
	parser.add_argument('--k', type=int, default=4, help='images count of a person in a batch')

	# model configuration
	parser.add_argument('--pid_num', type=int, default=4768,
						help='4768 for Comp_official, 4568 for Comp_val')
	parser.add_argument('--margin', type=float, default=0.3, help='margin for the triplet loss with batch hard')

	# train configuration
	parser.add_argument('--milestones', nargs='+', type=int, default=[40, 70], help='milestones for the learning rate decay')
	parser.add_argument('--base_learning_rate', type=float, default=0.00035)
	parser.add_argument('--weight_decay', type=float, default=0.0005)
	parser.add_argument('--resume_train_epoch', type=int, default=-1, help='-1 for no resuming')
	parser.add_argument('--total_train_epochs', type=int, default=120)
	parser.add_argument('--auto_resume_training_from_lastest_steps', type=ast.literal_eval, default=True)
	parser.add_argument('--max_save_model_num', type=int, default=1, help='0 for max num is infinit')

	# test configuration
	parser.add_argument('--resume_test_epoch', type=int, default=119, help='-1 for no resuming')

	# visualization configuration
	parser.add_argument('--resume_visualize_epoch', type=int, default=119, help='-1 for no resuming')


	# main
	config = parser.parse_args()
	main(config)


