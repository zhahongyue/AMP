# -*- coding: utf-8 -*-

"""
202206
zhahongyue
"""

from __future__ import print_function
import numpy as np
import tensorflow as tf
from attack_config import args
from glob import glob
from attack_utils import attack
from test_utils import test

def main(argv = None):  
  kwargs = vars(args)
  print('===============')
  for key, values in kwargs.items():
    print(key,':',values)
  print('===============')
  
  model_type = kwargs.pop('model_type')
  steg_type = kwargs.pop('steg_type')
  payload = kwargs.pop('payload')
  model_path = kwargs.pop('model_path')
  cover_dir = kwargs.pop('cover_dir')
  stego_dir = kwargs.pop('stego_dir')
  num_image = kwargs.pop('num_image')
  num_ori = kwargs.pop('num_ori')
  num_valid = kwargs.pop('num_valid')
  num_adv = kwargs.pop('num_adv')
  num_test = kwargs.pop('num_test')
  adv_save_dir = kwargs.pop('adv_save_dir')
  seed = kwargs.pop('seed')
  attack_name = kwargs.pop('attack_name')

  
  # create random indices for dividing training/testing/validation stegos
  full_indexes = np.arange(0, num_image)
  random_indexes = np.arange(0, num_image)
  np.random.seed(seed)
  np.random.shuffle(random_indexes)
  full_indexes = random_indexes[0 : num_ori]
  valid_indexes = random_indexes[num_ori:num_ori+num_valid]
  adv_indexes = random_indexes[num_ori+num_valid : num_ori+num_valid+num_adv]
  test_indexes = random_indexes[num_ori + num_valid + num_adv: num_image ]

  full_cover_list = sorted(glob(cover_dir + '/*'))    
  full_stego_list = sorted(glob(stego_dir + '/*'))

  ori_cover_list = [full_cover_list[i-1] for i in full_indexes]
  ori_stego_list = [full_stego_list[i-1] for i in full_indexes]
  valid_cover_list = [full_cover_list[i-1] for i in valid_indexes]
  valid_stego_list = [full_stego_list[i-1] for i in valid_indexes]
  adv_cover_list = [full_cover_list[i-1] for i in adv_indexes]
  adv_stego_list = [full_stego_list[i-1] for i in adv_indexes]
  test_cover_list = [full_cover_list[i-1] for i in test_indexes]
  test_stego_list = [full_stego_list[i-1] for i in test_indexes]

  adv_save_list = []
  for _,list_i in enumerate(adv_stego_list):
    adv_name =  list_i[len(stego_dir):len(list_i)]
    adv_save_list.append(adv_save_dir + adv_name)

  debug_num = 100

  ###################### attack ######################
  attack(
    model_type = model_type,
    steg_type = steg_type,
    payload = payload,
    model_path = model_path,
    atk_image_list = adv_cover_list[0:debug_num],
    atk_cover_list = adv_cover_list[0:debug_num],
    adv_rhoplus1_save_list = [],
    adv_rhominus1_save_list = [],
    attack_name = attack_name,
    advstego_save_list = adv_save_list[0:debug_num],
    **(kwargs)
  )
  
  print('\ntesting the model on oristego')
  test(
    model_type = model_type,
    steg_type = steg_type,
    payload = payload,
    model_path = model_path,
    test_cover_list = [],
    test_stego_list = adv_stego_list[0:debug_num]
  )
  
  print('\ntesting the model on advstego')
  test(
    model_type = model_type,
    steg_type = steg_type,
    payload = payload,
    model_path = model_path,
    test_cover_list = [],
    test_stego_list = adv_save_list[0:debug_num]
  )

if __name__ == '__main__':
  tf.app.run()