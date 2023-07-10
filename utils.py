# -*- coding: utf-8 -*-

"""
202206
zhahongyue
"""
import tensorflow as tf
import imageio
import numpy as np
import random
from glob import glob

class average_summary:
    # for summarising performance 
    # def __init__(self, 
    #              variable, 
    #              name, 
    #              num_variable):
    #     self.name = name
    #     # print("average_summary variable name:",self.name," num_variable:",num_variable)
    #     self.sum_variable = tf.Variable(tf.constant(0.0,dtype=tf.float32),
    #         trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
    #     with tf.control_dependencies([variable]):
    #         self.increment_op = tf.assign_add(self.sum_variable, variable)
    #     self.mean_variable = self.sum_variable / float(num_variable)
    #     self.summary = tf.summary.scalar(name, self.mean_variable)
    #     with tf.control_dependencies([self.summary]):
    #         self.reset_variable_op = tf.assign(self.sum_variable, 0)

    # def add_summary(self, sess, writer, step):
    #     mv,s, _ = sess.run([self.mean_variable,self.summary, self.reset_variable_op])   
    #     print(self.name,' ',mv) 
    #     writer.add_summary(s, step)
    #     return mv
    
    def __init__(self, 
                 variable, 
                 name, 
                 num_variable):
        self.name = name
        # print("average_summary variable name:",self.name," num_variable:",num_variable)
        self.sum_variable = tf.Variable(tf.constant(0.0,dtype=tf.float32),
            trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
        with tf.control_dependencies([variable]):
            self.increment_op = tf.assign_add(self.sum_variable, variable)
        self.mean_variable = self.sum_variable / np.asarray(num_variable,dtype=np.float32)
        self.reset_variable_op = tf.cast(tf.assign(self.sum_variable, 0.0),tf.float32)
    
    def add_summary(self, sess):
        mv = sess.run(self.mean_variable)
        print(self.name,' ',mv) 
        sess.run(self.reset_variable_op)   
        return mv

def get_img(img_list):
    """
    read the images in the img_list in the data format of NHWC
    """
    assert img_list!=[]
    img = imageio.imread(img_list[0])
    img_shape = img.shape
    img_batch = np.empty((len(img_list),img_shape[0],img_shape[1],1), dtype='uint8')
    for i in range(len(img_list)):
        tmp_img = imageio.imread(img_list[i])
        img_batch[i,:,:,0] = tmp_img # grayscale image only 
    return img_batch.astype(np.float32)

def get_rho(img_batch,steg_type='none'):
    """
    used in steg-adapt advattack
    return the minor value of rhop1 and rhom1
    """
    if steg_type == 'hill':
        rhop1_batch,rhom1_batch = hill(img_batch)
    elif steg_type == 'uniward':
        rhop1_batch,rhom1_batch = uniward(img_batch)
    else:
        raise Exception('Invalid steg function',steg_type)
    return np.asarray(rhop1_batch,dtype=np.float32),np.asarray(rhom1_batch,dtype=np.float32)

def get_stego(cover_batch,rhop1_batch,rhom1_batch,payload=0.5):
    """embedding simulator"""
    if payload==0:
        return cover_batch
    cover_batch = np.asarray(cover_batch,dtype=np.float32)
    stego_batch = np.copy(cover_batch)
    betap1_batch,betam1_batch = get_beta(rhop1_batch,rhom1_batch,payload)
    r_batch = np.random.rand(np.size(cover_batch)).astype(np.float32)
    r_batch = np.reshape(r_batch,cover_batch.shape)
    r = np.zeros(cover_batch.shape)
    r[r_batch<betap1_batch]=1
    r[r_batch>1-betam1_batch]=-1
    stego_batch += r
    stego_batch[stego_batch>255]=255
    stego_batch[stego_batch<0]=0
    return np.asarray(stego_batch,dtype=np.uint8)

def flip_and_rot(img_batch,randomseed=None):
    """flip and rot image batch for data augmentation""" 
    if randomseed == None:
        randomseed = random.random()    
    img_batch = np.asarray(img_batch,dtype=np.float32)
    img_batch_shape = img_batch.shape
    for n in range(img_batch_shape[0]):        
        for c in range(img_batch_shape[3]):
            random.seed(randomseed)
            tmp_img = img_batch[n,:,:,c]
            tmp_img = np.rot90(tmp_img,random.randint(0,3), axes=[0,1])
            if random.random()<0.5:
                tmp_img = np.flip(tmp_img, axis=0)
            if random.random()<0.5:
                tmp_img = np.flip(tmp_img, axis=1)
            img_batch[n,:,:,c] = tmp_img
    return img_batch

def get_beta(rhop1_batch,rhom1_batch,payload):
    """
    get beta batch from rho batch NHWC
    """ 
    rhop1_batch = np.asarray(rhop1_batch,dtype=np.float32)
    rhom1_batch = np.asarray(rhom1_batch,dtype=np.float32)
    rho_shape = rhop1_batch.shape
    img_num = rho_shape[0]
    img_channel = rho_shape[3]
    betap1_batch = np.zeros(rho_shape,dtype=np.float32)
    betam1_batch = np.zeros(rho_shape,dtype=np.float32)
    for n in range(img_num):
        for c in range(img_channel): 
            _lambda_upper = 1000.0
            _lambda_lower = 0 
            _lambda = np.copy(_lambda_upper)
            _rhop1 = rhop1_batch[n,:,:,c]
            _rhom1 = rhom1_batch[n,:,:,c]
            _size = np.size(_rhop1)
            _m_upper = np.float16(_size*(1.0001*payload))
            _m_lower = np.float16(_size*(0.9999*payload))
            exp_p1 = np.exp(-_lambda*_rhop1)
            exp_m1 = np.exp(-_lambda*_rhom1)
            exp_sum = 1+exp_p1+exp_m1
            exp_p1[np.isinf(exp_p1)] = 1e8
            exp_m1[np.isinf(exp_m1)] = 1e8
            exp_sum[np.isinf(exp_sum)] = 1e8
            
            _betap1 = exp_p1/exp_sum
            _betam1 = exp_m1/exp_sum            
            
            _m = get_payload_in_theory(_betap1,_betam1)
            iteration = 0 
            while _m>_m_upper or _m<_m_lower :    
                if _m > _m_upper:
                    if _lambda_upper == _lambda:
                        _lambda_upper = _lambda_upper*2
                        _lambda = _lambda_upper
                    else:
                        _lambda_lower = _lambda
                        _lambda = (_lambda_upper+_lambda_lower)/2
                if _m<_m_lower:
                    _lambda_upper = _lambda
                    _lambda = (_lambda_upper+_lambda_lower)/2
                exp_p1 = np.exp(-_lambda*_rhop1)
                exp_m1 = np.exp(-_lambda*_rhom1)
                exp_sum = 1+exp_p1+exp_m1
                exp_p1[np.isinf(exp_p1)] = 1e8
                exp_m1[np.isinf(exp_m1)] = 1e8
                exp_sum[np.isinf(exp_sum)] = 1e8
                _betap1 = exp_p1/exp_sum
                _betam1 = exp_m1/exp_sum  
                _m = get_payload_in_theory(_betap1,_betam1)
                iteration += 1
                if iteration >30:
                    break
            # print("original payload:",_m/_size)
            betap1_batch[n,:,:,c] = _betap1[:,:]
            betam1_batch[n,:,:,c] = _betam1[:,:]
    return np.asarray(betap1_batch,dtype=np.float32),np.asarray(betam1_batch,dtype=np.float32)

def get_payload_in_theory(betap1,betam1):
    betap1 = np.clip(np.asarray(betap1,dtype=np.float32),1e-8,1-1e-8)
    betap1 = betap1.flatten()
    betam1 = np.clip(np.asarray(betam1,dtype=np.float32),1e-8,1-1e-8)
    betam1 = betam1.flatten()
    p0 = np.clip(1-betap1-betam1,1e-8,1-1e-8)
    # betap1[betap1==0]=1e-8
    # betam1[betam1==0]=1e-8
    # p0[p0==0]=1e-8
    mp1 = -betap1*np.log2(betap1)
    mm1 = -betam1*np.log2(betam1)
    m0 = -p0*np.log2(p0)
    mp1[np.isnan(mp1)]=0
    mm1[np.isnan(mm1)]=0
    m0[np.isnan(m0)]=0
    payload_in_theory = np.sum(mp1)+np.sum(mm1)+np.sum(m0)
    return np.asarray(payload_in_theory,dtype=np.float32)

def get_sca_beta(img_batch,steg_type='none',payload = 0.5):
    """
    for SCAModel inputs
    """ 
    if steg_type == 'hill':
        rhop1_batch,rhom1_batch = hill(img_batch)
    elif steg_type == 'uniward':
        rhop1_batch,rhom1_batch = uniward(img_batch)
    else:
        raise Exception('Invalid steg function',steg_type)
    betap1_batch,betam1_batch = get_beta(rhop1_batch,rhom1_batch,payload)
    beta_batch = 0.5*betap1_batch + 0.5*betam1_batch
    return np.asarray(beta_batch,dtype=np.float32)



def get_minor_rho(img_batch,steg_type='none'):
    """
    used in steg-adapt advattack
    return the minor value of rhop1 and rhom1
    """
    if steg_type == 'hill':
        rhop1_batch,rhom1_batch = hill(img_batch)
    elif steg_type == 'uniward':
        rhop1_batch,rhom1_batch = uniward(img_batch)
    else:
        raise Exception('Invalid steg function',steg_type)
    rho_batch = np.copy(rhop1_batch)
    rho_batch[rho_batch>rhom1_batch] = rhom1_batch[rho_batch>rhom1_batch]
    return np.asarray(rho_batch,dtype=np.float32)

def uniward(img_batch, data_format ='NHWC'):
    """
    Calculate the uniward distortion
    """
    from scipy.signal import convolve2d
    img_batch = np.array(img_batch,dtype=np.float32)
    img_shape = img_batch.shape
    img_num = img_shape[0]
    channel_size = img_shape[3]    
    img_batch = img_batch.astype(np.float32)
    rhop1_batch = np.zeros(img_shape,dtype=np.float32)
    rhom1_batch = np.zeros(img_shape,dtype=np.float32)
    for n in range(img_num):
        for i in range(channel_size):
            temp_img = img_batch[n,:,:,i]
            sgm = 1
            hpdf = np.array([-0.0544158422, 0.3128715909, -0.6756307363, 0.5853546837, 0.0158291053, -0.2840155430, -0.0004724846, 0.1287474266, 0.0173693010, -0.0440882539,-0.0139810279, 0.0087460940, 0.0048703530, -0.0003917404, -0.0006754494, -0.0001174768])
            lpdf = np.multiply(((-1)**np.array(range(0, len(hpdf)))), np.flipud(hpdf))
            hpdf.shape = (1, len(hpdf))
            lpdf.shape = (1, len(lpdf))
            F1 = np.matmul(np.transpose(lpdf), hpdf)
            F2 = np.matmul(np.transpose(hpdf), lpdf)
            F3 = np.matmul(np.transpose(hpdf), hpdf)
            F = (F1, F2, F3)

            wetCost = 100000000  # 10**8
            xii = []
            for fIndex in range(0, 3):
                R = convolve2d(temp_img, F[fIndex], mode='same',boundary='symm')
                xi = convolve2d(1/(abs(R)+sgm), np.rot90(abs(F[fIndex]), 2), mode='same',boundary='symm')
                # if len(F[fIndex]) % 2 == 0:
                #     xi = np.roll(xi, 1, axis=0)
                # if len(F[fIndex][0]) % 2 == 0:
                #     xi = np.roll(xi, 1, axis=1)
                # xi = xi[int((len(xi)-len(temp_img))/2):int(len(xi)-(len(xi)-len(temp_img))/2), int((len(xi[0])-len(temp_img[0]))/2):int(len(xi[0])-(len(xi[0])-len(temp_img[0]))/2)]
                xii.append(xi)
            temp_rho = xii[0]+xii[1]+xii[2]
            temp_rho[temp_rho > wetCost] = wetCost
            temp_rho[np.isnan(temp_rho)] = wetCost
            temp_rhop1 = np.copy(temp_rho)
            temp_rhop1[temp_img==255]=wetCost
            temp_rhom1 = np.copy(temp_rho)
            temp_rhom1[temp_img==0]=wetCost
            rhop1_batch[n,:,:,i] = temp_rhop1[:,:]
            rhom1_batch[n,:,:,i] = temp_rhom1[:,:]
    return np.asarray(rhop1_batch,dtype=np.float32),np.asarray(rhom1_batch,dtype=np.float32)

def hill(img_batch, data_format = 'NHWC'):
    """
    Calculate the hill distortion
    """
    from scipy.signal import convolve2d
    img_batch = np.array(img_batch,dtype=np.float32)
    img_shape = img_batch.shape
    img_num = img_shape[0]
    channel_size = img_shape[3]    
    img_batch = img_batch.astype(np.float32)
    rhop1_batch = np.zeros(img_shape,dtype=np.float32)
    rhom1_batch = np.zeros(img_shape,dtype=np.float32)
    for n in range(img_num):
        for i in range(channel_size):
            temp_img = img_batch[n,:,:,i]
            wetCost = 10000000000  # 10**10
            sgm = 0.0000000001
            F = np.array([[-0.25, 0.5, -0.25],[0.5, -1, 0.5],[-0.25, 0.5, -0.25]])
            # compute residual
            R = convolve2d(temp_img, F, mode='same',boundary='symm')
            # compute suitability
            xi = convolve2d(abs(R),np.array([[1 for col in range(3)] for row in range(3)])/9.0, mode='same',boundary='symm')
            # compute embedding cost \rho
            with np.errstate(divide='ignore'):
                xi2 = 1.0/(xi + sgm)
            temp_rho = convolve2d(xi2, 1/225.0*np.array([[1 for col in range(15)] for row in range(15)]),mode='same',boundary='symm' )
            temp_rho[temp_rho > wetCost] = wetCost
            temp_rho[np.isnan(temp_rho)] = wetCost 
            temp_rhop1 = np.copy(temp_rho)
            temp_rhop1[temp_img==255]=wetCost
            temp_rhom1 = np.copy(temp_rho)
            temp_rhom1[temp_img==0]=wetCost
            rhop1_batch[n,:,:,i] = temp_rhop1[:,:]
            rhom1_batch[n,:,:,i] = temp_rhom1[:,:]
    return np.asarray(rhop1_batch,dtype=np.float32),np.asarray(rhom1_batch,dtype=np.float32)

def embed_dir(coverdir, stegodir, payload_rate, steg_type):
    print('coverdir:',coverdir)
    print('stegodir:',stegodir)    
    print('payload_rate:',payload_rate)
    print('steg_type:',steg_type)
    full_cover_list = sorted(glob(coverdir + '/*'))  
    print('coverdir_length:',len(full_cover_list))
    full_stego_list = []
    for _,list_i in enumerate(full_cover_list):
        img_name =  list_i[len(coverdir):len(list_i)]
        full_stego_list.append(stegodir + img_name)
    for i in range(len(full_cover_list)):
        cover_img = get_img([full_cover_list[i]])
        rho_p1,rho_m1 = get_rho(cover_img,steg_type)
        stego_img = get_stego(cover_img,rho_p1,rho_m1,payload_rate)
        stego_img = np.uint8(np.squeeze(stego_img))
        imageio.imwrite(full_stego_list[i],stego_img)

def get_rho_dir(coverdir, rhodir, steg_type):
    print('coverdir:',coverdir)
    print('rhodir:',rhodir)
    print('steg_type:',steg_type)
    full_cover_list = sorted(glob(coverdir + '/*'))  
    print('coverdir_length:',len(full_cover_list))
    full_rhoplus1_list = []
    full_rhominus1_list = []
    for _,list_i in enumerate(full_cover_list):
        img_name =  list_i[len(coverdir):-4]
        full_rhoplus1_list.append(rhodir + img_name + '_rhoplus1.npy')
        full_rhominus1_list.append(rhodir + img_name + '_rhominus1.npy')
    for i in range(len(full_cover_list)):
        cover_img = get_img([full_cover_list[i]])
        rho_p1,rho_m1 = get_rho(cover_img,steg_type)
        np.save(full_rhoplus1_list[i],rho_p1)
        np.save(full_rhominus1_list[i],rho_m1)
        
def save_to_npy(npy_path,item):
    item = np.array(item)
    np.save(npy_path,item)
    
def npy_to_list(npy_path):
    return (np.load(npy_path)).tolist()

def get_img_list_from_dir( dir,num_image):
    full_indexes = np.arange(1, num_image + 1)
    full_img_path_list = []
    for i in range(num_image):
        img_path = dir + str(full_indexes[i])+'.pgm'
        full_img_path_list.append(img_path)    
    return full_img_path_list

def compute_imageset_diff(imageset_dir1, imageset_dir2, num_image, debug = 0):
    image_list1 = get_img_list_from_dir(imageset_dir1, num_image)
    image_list2 = get_img_list_from_dir(imageset_dir2, num_image)
    l1_diff_sum = 0
    l0_diff_sum = 0
    diff_image_count = 0
    for i in range(num_image):
        image1 = imageio.imread(image_list1[i])
        image2 = imageio.imread(image_list2[i])
        imagediff = np.asarray(image1.flatten(),dtype=np.float32)-np.asarray(image2.flatten(),dtype=np.float32)
        current_l1_diff = np.linalg.norm(imagediff, ord=1)
        current_l0_diff = np.linalg.norm(imagediff, ord=0)
        l1_diff_sum += current_l1_diff
        l0_diff_sum += current_l0_diff
        if current_l0_diff!=0:
            diff_image_count += 1
        if debug:
            if current_l0_diff!=0:
                print('image_num:',i)
                print(image_list1[i])
                print(image_list2[i])
                print('l0_diff:',current_l0_diff, ' l1_diff:',current_l1_diff) 
                print('diff_image_count:',diff_image_count) 
    return l0_diff_sum/num_image, l1_diff_sum/num_image