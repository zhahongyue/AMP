# -*- coding: utf-8 -*-

"""
202206
zhahongyue
"""

import argparse

parser = argparse.ArgumentParser(description='Hyper-parameter:')
# about model
parser.add_argument('--model_type',
					type=str,
					help='steganalysis network type')
parser.add_argument('--steg_type',
					type=str,
					default = 'hill',
					help='steganography type')
parser.add_argument('--payload',
					type=float,
					default = 0.5,
					help='payload ratio')
parser.add_argument('--model_path',
					type = str,
					help='path of model')

parser.add_argument('--cover_dir',
					type=str,
					default = '/public/zhahongyue/DATASET/BOSSBOWS20000/',
					help='cover directory')
parser.add_argument('--stego_dir',
					type=str,
					help='stego directory')
# about dataset partition
# the number denotes the dataset image number rather than cover+stego image number
parser.add_argument('--num_image',
					type=int,
					default = 20000,
					help='total image number')
parser.add_argument('--num_ori',
					type=int,
					default = 8000,
					help='the number of original training images')
parser.add_argument('--num_valid',
					type=int,
					default = 2000,
					help='the number of original validating images')
parser.add_argument('--num_adv',
					type=int,
					default = 8000,
					help='the number of adversarial training images')
parser.add_argument('--num_test',
					type=int,
					default = 2000,
					help='the number of test images')		

#about random seed
parser.add_argument('--seed',
					type = int,
					default = 71,
					help='the seed used for shuffle the image dataset')
#about attack
parser.add_argument('--adv_save_dir',
					type = str,
					default='/public/zhahongyue/AMP/temp/',
					help='directory to save adversarial images')
parser.add_argument('--attack_name',
					type = str,
					help='name of attack')
parser.add_argument('--embed_from_rho',
					type = int,
					default = 0,
					help='name of attack')
parser.add_argument('--attack_debug',
					type = int,
					default= 0,
					help='name of attack')


parser.add_argument('--amp_random_map_batchnum',
					type = int,
					default = 1,
					help='name of attack')
parser.add_argument('--amp_nonstop_batchnum',
					type = int,
					default = 0,
					help='name of attack')
parser.add_argument('--amp_lambda_l0',
					type = float,
					default = 0.0,
					help='name of attack')
parser.add_argument('--amp_random_map_batchsize',
					type = int,
					default = 24,
					help='name of attack')
parser.add_argument('--amp_iter_each_batch',
					type = int,
					default = 100,
					help='name of attack')
parser.add_argument('--amp_init_learning_rate',
					type = float,
					default = 0.001,
					help='name of attack')
parser.add_argument('--amp_init_stddev',
					type = float,
					default = 0.00001,
					help='name of attack')

args = parser.parse_args()