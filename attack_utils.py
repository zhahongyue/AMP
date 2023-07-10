# -*- coding: utf-8 -*-

"""
202206
zhahongyue
"""
import imageio
import numpy as np
import time
from model import Model,SRNet,YeNet,XuNet
from attack import AMP
from utils import get_img, get_stego

def attack( 
    model_type ,
    steg_type,
    payload,
    model_path,
    atk_image_list,
    atk_cover_list, 
    attack_name,
    advstego_save_list,
    adv_rhoplus1_save_list,
    adv_rhominus1_save_list,
    atk_rhoplus1_list = [],
    atk_rhominus1_list = [],
    advcover_save_list = [],
    **kwargs
  ):
    assert advstego_save_list!=[]
    if model_type =='SRNet':
        model_class = SRNet
    elif model_type =='YeNet':
        model_class = YeNet
    elif model_type =='XuNet':
        model_class = XuNet
    else:
        raise Exception("attack model_type invalid",model_type)
    
    if attack_name == 'AMP':
        attack_class = AMP
    else: 
        raise Exception("attack attack_name invalid",attack_name)
    
    if steg_type == 'HILL' or steg_type == 'hill':
        steg_type = 'hill'
    elif steg_type == 'uniward' or steg_type == 'suniward' or steg_type == 'SUNIWARD' or steg_type == 'UNIWARD':
        steg_type = 'uniward'
    else:
        raise Exception("attack steg_type invalid",attack_name)
    
    print('start attak')
    print('attak model type:', model_type,' steg type:',steg_type, ' payload:',payload, ' model path:', model_path, ' attack name:', attack_name)
    
    atk_label = np.ones(len(atk_image_list),dtype=np.int32)
    _attack = attack_class(attack_model_class = model_class,attack_steg_type = steg_type,attack_payload=payload,attack_model_path = model_path,**kwargs)

    for i in range(len(atk_image_list)):
        
        print("\nAttack number:",i+1)
        print("atk_cover path:",atk_cover_list[i:i+1])
        print("advstego save path:",advstego_save_list[i:i+1])
        
        atk_img = get_img(atk_image_list[i:i+1])
        atk_cover = get_img(atk_cover_list[i:i+1])
        
        if (attack_name == 'AMP' ) :
            if (atk_rhoplus1_list != []) & (atk_rhominus1_list != []):
                atk_rhoplus1 = np.load(atk_rhoplus1_list[i])
                atk_rhominus1 = np.load(atk_rhominus1_list[i])
                adv_img, adv_rho_plus1, adv_rho_minus1 = _attack._find_adv_img(atk_img,atk_cover,atk_label[i:i+1],atk_rhoplus1,atk_rhominus1)
            else:
                atk_rhoplus1 = None
                atk_rhominus1 = None
                adv_img, adv_rho_plus1, adv_rho_minus1 = _attack._find_adv_img(atk_img,atk_cover,atk_label[i:i+1])
            
        if kwargs.get('embed_from_rho') == 1:
            adv_img = get_stego(atk_cover,adv_rho_plus1,adv_rho_minus1,payload)
            
        adv_img = np.squeeze(adv_img)
        imageio.imwrite(advstego_save_list[i],adv_img)
        if adv_rhoplus1_save_list != []:    
            np.save(adv_rhoplus1_save_list[i],adv_rho_plus1)
        if adv_rhominus1_save_list != []:
            np.save(adv_rhominus1_save_list[i],adv_rho_minus1)
        if advcover_save_list != []:
            adv_cover = np.squeeze(adv_cover)
            imageio.imwrite(advcover_save_list[i],adv_cover)
    del _attack
    
def get_advrho_list_from_dir( dir,num_image):
    full_indexes = np.arange(1, num_image + 1)
    full_rhoplus1_path_list = []
    for i in range(num_image):
        img_path = dir + str(full_indexes[i])+'_rhoplus1.npy'
        full_rhoplus1_path_list.append(img_path)
    full_rhomius1_path_list = []
    for i in range(num_image):
        img_path = dir + str(full_indexes[i])+'_rhominus1.npy'
        full_rhomius1_path_list.append(img_path)     
    return full_rhoplus1_path_list,full_rhomius1_path_list

def get_adv_prop_list_from_dir( dir,num_image):
    full_indexes = np.arange(1, num_image + 1)
    full_rhoplus1_path_list = []
    for i in range(num_image):
        img_path = dir + str(full_indexes[i])+'_prop_plus1.npy'
        full_rhoplus1_path_list.append(img_path)
    full_rhomius1_path_list = []
    for i in range(num_image):
        img_path = dir + str(full_indexes[i])+'_prop_minus1.npy'
        full_rhomius1_path_list.append(img_path)     
    return full_rhoplus1_path_list,full_rhomius1_path_list