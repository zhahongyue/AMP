import tensorflow as tf
import numpy as np
import time
from .AdvAttack import AdvAttack
from model import Model
from utils import get_rho,get_beta,get_payload_in_theory, get_stego

class AMP(AdvAttack):
    """
    Adversarial Modification Probabilities
    
    """
    def __init__(self,
                 attack_model_class=None,
                 attack_steg_type='hill', 
                 attack_payload=0.5,
                 attack_model_path=None, 
                 data_format='NCHW', 
                 confidence = 0.0001, 
                 payload_ral_diff = 1e-3, 
                 lambda_adv = 1, 
                 lambda_payload = 1, 
                 lambda_valid_prop = 1,
                #  lambda_boundary = 0,
                #  lamda_l0 = 0.0, 
                 lambda_double_tanh = 120, 
                 random_map_batchnum = 110, 
                #  iter_each_batch = 1,
                #  init_learning_rate = 0.001,
                #  nonstop_batchnum = 0,
                #  random_map_batchsize = 5,                
                 learning_rate_decay_rate = 1.0,
                 learning_rate_decay_batch = 2, 
                 tensorboard_path = '/public/zhahongyue/AMP/tensorboard/',
                 debug = 0,
                 **kwargs
                 ):
        """
        Args:
            attack_model_class ([type], optional): [the steganalysis model to attack]. Defaults to None.
            attack_steg_type (str, optional): [the steganographic distortion function name]. Defaults to 'none'.
            attack_payload (float, optional): [the payload for the model]. Defaults to 0.5.
            attack_model_path ([type], optional): [attack model checkpoint path]. Defaults to None.
            data_format (str, optional): [data format now only support 'NCHW' type]. Defaults to 'NCHW'.
            
        """
        lambda_l0 = kwargs.pop('amp_lambda_l0')
        nonstop_batchnum = kwargs.pop('amp_nonstop_batchnum')
        random_map_batchsize = kwargs.pop('amp_random_map_batchsize')
        iter_each_batch = kwargs.pop('amp_iter_each_batch')
        init_learning_rate = kwargs.pop('amp_init_learning_rate')
        debug = kwargs.pop('attack_debug')
        init_stddev = kwargs.pop('amp_init_stddev')
        
        if issubclass(attack_model_class,Model):
            self.attack_model = attack_model_class(False,data_format)
        self.attack_steg_type = attack_steg_type
        self.attack_payload = attack_payload
        self.attack_model_path = attack_model_path
        self.data_format = data_format
        self.random_map_batchsize = random_map_batchsize
        self.random_map_batchnum = random_map_batchnum
        self.iter_each_batch = iter_each_batch
        self.init_learning_rate = init_learning_rate
        self.tensorboard_path = tensorboard_path
        self.nonstop_batchnum = nonstop_batchnum
        print("attack_steg_type:",self.attack_steg_type)
        print("attack_payload:",self.attack_payload)
        print("attack_model_path:",self.attack_model_path)
        print("random_map_batchsize:",self.random_map_batchsize)
        print("random_map_batchnum:",self.random_map_batchnum)
        print("iter_each_batch:",self.iter_each_batch)
        print("init_learning_rate:",self.init_learning_rate)
        print("learning_rate_decay_rate:",learning_rate_decay_rate)
        print("learning_rate_decay_batch:",learning_rate_decay_batch)
        print("tensorboard_path:",self.tensorboard_path) 
        print("nonstop_batchnum:",self.nonstop_batchnum)
        
        print("init_learning_rate:",self.init_learning_rate)
        print("lambda_adv:",lambda_adv)
        print("lambda_payload:",lambda_payload)
        print("lambda_l0:",lambda_l0)
        print("lambda_valid_prop:",lambda_valid_prop)
        # print("lambda_boundary:",lambda_boundary)

        self.oa_l0_summary = 0
        self.oa_l1_summary = 0
        self.oa_l2_summary = 0
        self.oa_linf_summary = 0        
        
        self.total_count = 0
        self.success_count = 0
        self.attack_count = 0
        self.time_count = 0
        self.debug = debug
        self.wetCost = 1e8
        
        tf.reset_default_graph()
        ########## input image(in amp,  ori_img is not necessary )  ###########

        self.atk_cover_ph = tf.placeholder(tf.float64, shape = (1, 256, 256, 1), name ='img_placeholder')
        self.atk_cover = tf.Variable(tf.zeros_like(self.atk_cover_ph,dtype=tf.float64),trainable=False,name='atk_cover') 
        self.set_atk_cover_op = tf.assign(self.atk_cover,self.atk_cover_ph)
        self.batch_atk_cover = tf.tile(self.atk_cover,[self.random_map_batchsize,1,1,1])

        self.prop_plus1_mask = 1 - tf.sigmoid(1000000*(self.atk_cover-254.5))
        self.prop_minus1_mask = tf.sigmoid(1000000*(self.atk_cover-0.5))

        ###########  input +1/-1 modification probabilities map ############
        self.prop_plus1_ph = tf.placeholder(tf.float64, shape = (1, 256, 256, 1), name ='prop_plus1_placeholder')
        self.ori_prop_plus1 = tf.Variable(tf.zeros_like(self.prop_plus1_ph,dtype=tf.float64),trainable=False,name='ori_prop_plus1') 
        self.set_ori_prop_plus1_op = tf.assign(self.ori_prop_plus1,self.prop_plus1_ph)       

        self.adv_prop_plus1_tanh = tf.get_variable(name='adv_prop_plus1_tanh',shape=(1, 256, 256, 1),trainable = True, initializer = tf.truncated_normal_initializer(stddev=init_stddev),dtype=tf.float64)    
        self.adv_prop_plus1 = tf.clip_by_value(self.prop_plus1_mask*(self.ori_prop_plus1 + tf.math.tanh(self.adv_prop_plus1_tanh)),1/self.wetCost,1-1/self.wetCost)
        with tf.control_dependencies([tf.initialize_variables([self.adv_prop_plus1_tanh])]):
            self.adv_prop_plus1_tanh_weighted_init  = tf.assign(self.adv_prop_plus1_tanh,self.adv_prop_plus1_tanh*self.ori_prop_plus1*3)
        self.batch_adv_prop_plus1 = tf.tile(self.adv_prop_plus1,[self.random_map_batchsize,1,1,1]) 

        self.prop_minus1_ph = tf.placeholder(tf.float64, shape = (1, 256, 256, 1), name ='prop_minus1_placeholder')
        self.ori_prop_minus1 = tf.Variable(tf.zeros_like(self.prop_minus1_ph,dtype=tf.float64),trainable=False,name='ori_prop_minus1') 
        self.set_ori_prop_minus1_op = tf.assign(self.ori_prop_minus1,self.prop_minus1_ph)   

        self.adv_prop_minus1_tanh = tf.get_variable(name='adv_prop_minus1_tanh',shape=(1, 256, 256, 1),trainable = True, initializer = tf.truncated_normal_initializer(stddev=init_stddev),dtype=tf.float64)     
        self.adv_prop_minus1 = tf.clip_by_value(self.prop_minus1_mask*(self.ori_prop_minus1 + tf.math.tanh(self.adv_prop_minus1_tanh)),1/self.wetCost,1-1/self.wetCost)
        with tf.control_dependencies([tf.initialize_variables([self.adv_prop_minus1_tanh])]):
            self.adv_prop_minus1_tanh_weighted_init = tf.assign(self.adv_prop_minus1_tanh,self.adv_prop_minus1_tanh*self.ori_prop_minus1*3)
        self.batch_adv_prop_minus1 = tf.tile(self.adv_prop_minus1,[self.random_map_batchsize,1,1,1]) 

        ########### calculate the 0-modification probabilities map  ###########
        self.ori_prop_0 = tf.clip_by_value(1-self.ori_prop_plus1-self.ori_prop_minus1,1/self.wetCost,1-1/self.wetCost)
        self.adv_prop_0 = tf.clip_by_value(1-self.adv_prop_plus1-self.adv_prop_minus1,1/self.wetCost,1-1/self.wetCost)
        
        ########### calculate the payload ratio (bits per pixel) according to above probabilities  ############
        # self.ori_payload = -tf.reduce_mean(
        #     tf.multiply(self.ori_prop_plus1,tf.log(self.ori_prop_plus1)/tf.log(2.0))+
        #     tf.multiply(self.ori_prop_minus1,tf.log(self.ori_prop_minus1)/tf.log(2.0))+
        #     tf.multiply(self.ori_prop_0,tf.log(self.ori_prop_0)/tf.log(2.0))
        #     )
        
        self.adv_payload = -tf.reduce_mean(
            tf.multiply(self.adv_prop_plus1,tf.log(self.adv_prop_plus1)/tf.log(2.0))+
            tf.multiply(self.adv_prop_minus1,tf.log(self.adv_prop_minus1)/tf.log(2.0))+
            tf.multiply(self.adv_prop_0,tf.log(self.adv_prop_0)/tf.log(2.0))
            )
              
        ########## embedding simulator by double-tanh modular (batch case) ############
        
        self.random_map_ph = tf.placeholder(tf.float64, shape = (self.random_map_batchsize, 256, 256, 1), name ='random_map_placeholder')
        self.random_map = tf.Variable(tf.zeros_like(self.random_map_ph,dtype=tf.float64),trainable=False,name='random_map') 
        self.set_random_map_op = tf.assign(self.random_map,self.random_map_ph)
        
        self.batch_modification = 0.5*tf.tanh(lambda_double_tanh*(self.batch_adv_prop_plus1-self.random_map))-0.5*tf.tanh(lambda_double_tanh*(self.batch_adv_prop_minus1-1+self.random_map))
        
        ########## calculate the adv_img (batch case)  ############
        self.inference_only = tf.Variable(0.0,trainable=False,name='inference_only_ind')
        self.enable_inference_only = tf.assign(self.inference_only,1.0)
        self.disable_inference_only = tf.assign(self.inference_only,0.0)

        self.inference_img_ph = tf.placeholder(tf.float64, shape = (self.random_map_batchsize, 256, 256, 1), name ='inference_img_placeholder')
        self.inference_img = tf.Variable(tf.zeros_like(self.inference_img_ph,dtype=tf.float64),trainable=False,name='inference_img') 
        self.set_inference_img_op = tf.assign(self.inference_img,self.inference_img_ph)
        
        self.batch_adv_img = (1.0-self.inference_only)*tf.clip_by_value(self.batch_modification + self.batch_atk_cover,0,255) + self.inference_only * self.inference_img
        
        ############ steganalysis network  ###############
        self.attack_model._build_model(tf.cast(self.batch_adv_img,tf.float64))
        
        ########## losses #############
        self.adv_loss = lambda_adv * tf.reduce_sum(tf.abs(-self.attack_model.logits[:,0]+self.attack_model.logits[:,1]))/self.random_map_batchnum
        
        # self.l0_loss = lambda_l0 * tf.reduce_sum(tf.abs(self.batch_adv_img-self.batch_atk_cover))/self.random_map_batchnum

        self.l0_loss = lambda_l0 * tf.reduce_mean(
            tf.abs(self.adv_prop_plus1-self.ori_prop_plus1)+
            tf.abs(self.adv_prop_minus1-self.ori_prop_minus1)+
            tf.abs(self.adv_prop_0-self.ori_prop_0)
            )
        
        self.valid_prop_loss =   tf.reduce_sum(tf.maximum( lambda_valid_prop * (self.adv_prop_plus1+self.adv_prop_minus1-1+1/self.wetCost),0))/self.random_map_batchnum
        
        # self.upper_boundary_loss = lambda_boundary * tf.reduce_sum(tf.maximum(self.batch_modification + self.batch_atk_cover-255,0))/self.random_map_batchnum
        
        # self.lower_boundary_loss = lambda_boundary * tf.reduce_sum(tf.maximum(-self.batch_modification - self.batch_atk_cover,0))/self.random_map_batchnum
        
        self.payload_loss =  lambda_payload * tf.maximum( tf.abs(self.adv_payload - self.attack_payload)/self.attack_payload - payload_ral_diff ,0)
        
        self.loss =  self.adv_loss +  self.payload_loss  + self.valid_prop_loss + self.l0_loss
        
        # +  self.upper_boundary_loss +  self.lower_boundary_loss
        
        self.global_iter = tf.Variable(tf.constant(0,dtype=tf.int64),trainable=False,
        name='global_iter')
        self.reset_global_iter = tf.assign(self.global_iter,0)
        
        self.learning_rate = tf.train.exponential_decay(self.init_learning_rate,self.global_iter,learning_rate_decay_batch*self.iter_each_batch,learning_rate_decay_rate)
        
        variables_to_restore = tf.contrib.slim.get_variables_to_restore(exclude=['atk_cover','ori_prop_plus1','adv_prop_plus1_tanh','ori_prop_minus1','adv_prop_minus1_tanh','random_map','inference_only_ind','inference_img','global_iter'])
        
        if self.debug == 1:
            # tf.summary.scalar('adv_loss', self.adv_loss)
            # tf.summary.scalar('payload_loss', self.payload_loss)
            # tf.summary.scalar('l0_loss', self.l0_loss)
            # tf.summary.scalar('valid_prop_loss', self.valid_prop_loss)
            # tf.summary.scalar('boundary_loss',lambda_boundary * self.upper_boundary_loss + lambda_boundary * self.lower_boundary_loss)
            # tf.summary.scalar('loss',self.loss)
            # tf.summary.scalar('learning_rate',self.learning_rate)        
            # self.merge_summary = tf.summary.merge_all()        
            print("variables_to_restore:",variables_to_restore)
                
        self.minimizing_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss,self.global_iter,var_list=[self.adv_prop_plus1_tanh,self.adv_prop_minus1_tanh])
        
        ############## initialization ##################
        
        self.init_op = tf.group(tf.global_variables_initializer(),
                                tf.local_variables_initializer())  
        self.saver = tf.train.Saver(variables_to_restore)
        self.sess = tf.Session()
        self.sess.run(self.init_op)
        self.saver.restore(self.sess,self.attack_model_path)
        print("restore success!")
        # if self.debug == 1:
            # self.writer = tf.summary.FileWriter(tensorboard_path,self.sess.graph)
        
    def _find_adv_img(self,ori_img,atk_cover,ori_label,atk_rhoplus1=None,atk_rhominus1=None,**kwargs):        
        start_time = time.time()
        self.total_count = self.total_count+1
        self.attack_count = self.attack_count+1 
        if self.debug:
            print("\nStart attack num: ",self.attack_count) 
            
        adv_stego = atk_cover.copy()  
        if np.sum(atk_rhoplus1 == None) or np.sum(atk_rhominus1 == None) :
            ori_rho_plus1, ori_rho_minus1 = get_rho(atk_cover,self.attack_steg_type)
        else:
            ori_rho_plus1 = atk_rhominus1
            ori_rho_minus1 = atk_rhominus1
        ori_prop_plus1, ori_prop_minus1 = get_beta(ori_rho_plus1,ori_rho_minus1,self.attack_payload)
        
        if self.debug == 1:
            self.print_img_info(atk_cover,'atk_cover') 
            ori_payload_rate = get_payload_in_theory(ori_prop_plus1, ori_prop_minus1)/(256*256)
            print("target payload rate:",self.attack_payload," original payload rate:",ori_payload_rate)
        
        adv_rho_plus1 = ori_rho_plus1.copy()
        adv_rho_minus1 = ori_rho_minus1.copy()
        
        self.sess.run([self.set_atk_cover_op,self.set_ori_prop_plus1_op,self.set_ori_prop_minus1_op],feed_dict={self.atk_cover_ph:atk_cover, self.prop_plus1_ph:ori_prop_plus1,self.prop_minus1_ph:ori_prop_minus1})
        
        self.sess.run([self.adv_prop_plus1_tanh_weighted_init,
                       self.adv_prop_minus1_tanh_weighted_init,
                       self.reset_global_iter,
                       self.disable_inference_only])
        
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        
        # if self.debug:
        #     # last_adv_prop_0 =  np.array(self.sess.run(self.adv_prop_0)).flatten()
        #     # last_adv_prop_plus1 = np.array(self.sess.run(self.adv_prop_plus1)).flatten()
        #     # last_adv_prop_minus1 = np.array(self.sess.run(self.adv_prop_minus1)).flatten()
            
        for i in range(self.random_map_batchnum):
            np.random.seed(np.int(time.time()))
            random_map = np.random.rand(self.random_map_batchsize,256,256,1)
            self.sess.run(self.set_random_map_op,feed_dict={self.random_map_ph:random_map})
            
            with tf.control_dependencies(update_ops):
                # if self.debug == 1:
                    # adv_prop_plus1 = self.sess.run(self.adv_prop_plus1)
                    # adv_prop_minus1 = self.sess.run(self.adv_prop_minus1)
                    # adv_payload_in_theory = get_payload_in_theory(adv_prop_plus1,adv_prop_minus1)/(256*256)
                    # print('adv_payload_in_theory:',adv_payload_in_theory)
                    # adv_payload = self.sess.run(self.adv_payload)
                    # print('adv_payload:',adv_payload)
                loss = self.sess.run(self.loss)
                adv_loss = self.sess.run(self.adv_loss)
                valid_prop_loss = self.sess.run(self.valid_prop_loss)
                payload_loss = self.sess.run(self.payload_loss)
                l0_loss = self.sess.run(self.l0_loss)
                
                if self.debug == 1:
                    print("\nStart of batch ",i,"\tloss:",format(loss,'.4f'),
                          "\tadv_loss:",format(adv_loss,'.4f'),
                          "\tvalid_prop_loss",format(valid_prop_loss,'.4f'),
                          "\tpayload_loss",format(payload_loss,'.4f'),
                          "\tl0_loss",format(l0_loss,'.4f')
                          )
                
            # if (i>=self.nonstop_batchnum) or ((adv_loss<1e-4) & (valid_prop_loss<1e-4)):
            # if (i>=self.nonstop_batchnum):
            #     with tf.control_dependencies(update_ops):  
                    # adv_prop_plus1 = self.sess.run(self.adv_prop_plus1)
                    # adv_prop_minus1 = self.sess.run(self.adv_prop_minus1)
                    # adv_prop_0 = self.sess.run(self.adv_prop_0)
                    # adv_rho_plus1 = np.log(adv_prop_0/adv_prop_plus1)
                    # adv_rho_minus1 = np.log(adv_prop_0/adv_prop_minus1)
                    # adv_rho_plus1 = np.clip(adv_rho_plus1,-self.wetCost,self.wetCost)
                    # adv_rho_minus1 = np.clip(adv_rho_minus1,-self.wetCost,self.wetCost)
                    
                    # self.sess.run(self.enable_inference_only) 
                    # advstego_from_advrho = get_stego(atk_cover,adv_rho_plus1,adv_rho_minus1,self.attack_payload)    
                    # batch_advstego_from_advrho = np.tile(advstego_from_advrho,(self.random_map_batchsize,1,1,1))       
                            
                    # batch_advstego_from_advrho = get_stego(
                        # np.tile(atk_cover,(self.random_map_batchsize,1,1,1)),
                        # np.tile(adv_rho_plus1,(self.random_map_batchsize,1,1,1)),
                        # np.tile(adv_rho_minus1,(self.random_map_batchsize,1,1,1))
                        # )
                    # self.sess.run(self.set_inference_img_op,feed_dict={self.inference_img_ph:batch_advstego_from_advrho}) 
                    # with tf.control_dependencies(update_ops): 
                    #     batch_advstego_from_rho_forward = self.sess.run(self.attack_model.forward)
                    #     self.sess.run(self.disable_inference_only)
                    #     if self.debug == 1:
                    #         print("Start of batch ",i,"\tbatch_advstego_from_rho_forward:",batch_advstego_from_rho_forward)
                    #     for k in range(self.random_map_batchsize):
                    #         if batch_advstego_from_rho_forward[k] == 0:
                    #             adv_stego = np.asarray([batch_advstego_from_advrho[k,:,:,:]])
                    #             if self.debug == 1:
                    #                 self.print_img_info(adv_stego,'adv_stego') 
                    #                 print("adv_stego atk_cover l0:",np.sum(abs(adv_stego-atk_cover)))
                    #             break
                    # if np.sum(adv_stego!=atk_cover)!=0:
                    #     break
                        
                    # batch_adv_stego = self.sess.run(self.batch_adv_img) 
                    # batch_adv_forward = self.sess.run(self.attack_model.forward) 
                    # if self.debug == 1:
                        # self.print_img_info(self.sess.run(self.batch_atk_cover),'batch_atk_cover') 
                        # print('inference_only:',self.sess.run(self.inference_only))
                        # self.print_img_info(batch_adv_stego,'batch_adv_stego')  
                        # print('l0_loss:',self.sess.run(self.l0_loss))
                        # temp = np.asarray(self.sess.run(self.batch_modification))
                        # self.print_img_info(temp,"modification")
                        # print("batch ",i," batch_adv_forward:",batch_adv_forward)
                    
                
                # if np.asarray(batch_adv_forward==0).any():
                #     clipped_batch_adv_stego = np.clip(np.round(np.asarray(batch_adv_stego)),0,255)
                #     # if self.debug == 1:
                #         # self.print_img_info(clipped_batch_adv_stego,'clipped_batch_adv_stego')
                #     self.sess.run(self.enable_inference_only)
                #     self.sess.run(self.set_inference_img_op,feed_dict={self.inference_img_ph:clipped_batch_adv_stego}) 
                #     with tf.control_dependencies(update_ops):                   
                #         clipped_batch_adv_forward = self.sess.run(self.attack_model.forward)
                #         if self.debug == 1:
                #             # print("clipped batch ",i," adv_forward:",clipped_batch_adv_forward)
                #             # print('l0_loss:',self.sess.run(self.l0_loss))
                #             print("batch ",i," clipped_batch_adv_forward:",clipped_batch_adv_forward)
                #         self.sess.run(self.disable_inference_only)
                #     for k in range(self.random_map_batchsize):
                #         if clipped_batch_adv_forward[k] == 0:
                #             adv_stego = np.asarray([clipped_batch_adv_stego[k,:,:,:]])
                #             if self.debug == 1:
                #                 self.print_img_info(adv_stego,'adv_stego') 
                #                 print("adv_stego atk_cover l0:",np.sum(abs(adv_stego-atk_cover)))
                #             if np.sum(adv_stego!=atk_cover)!=0:
                #                 adv_prop_plus1 = self.sess.run(self.adv_prop_plus1)
                #                 adv_prop_minus1 = self.sess.run(self.adv_prop_minus1)
                #                 adv_prop_0 = self.sess.run(self.adv_prop_0)
                #                 adv_rho_plus1 = np.log(adv_prop_0/adv_prop_plus1)
                #                 adv_rho_minus1 = np.log(adv_prop_0/adv_prop_minus1)
                #                 adv_rho_plus1 = np.clip(adv_rho_plus1,-self.wetCost,self.wetCost)
                #                 adv_rho_minus1 = np.clip(adv_rho_minus1,-self.wetCost,self.wetCost)
                #                 break
            # if self.debug:
            #     last_adv_prop_0 =  np.array(self.sess.run(self.adv_prop_0)).flatten()
            #     last_adv_prop_plus1 = np.array(self.sess.run(self.adv_prop_plus1)).flatten()
            #     last_adv_prop_minus1 = np.array(self.sess.run(self.adv_prop_minus1)).flatten()
            
            for j in range(self.iter_each_batch):
                with tf.control_dependencies(update_ops):
                    batch_adv_forward = np.asarray(self.sess.run(self.attack_model.forward)) 
                    if self.debug == 1:
                        print("batch ",i," iter",j," batch_adv_forward:",batch_adv_forward)
                    if (i>=self.nonstop_batchnum):
                        acc_batch = (np.sum(batch_adv_forward==0))/(np.size(batch_adv_forward))
                        if acc_batch >=0.45 and acc_batch<=0.55:
                            adv_prop_plus1 = self.sess.run(self.adv_prop_plus1)
                            adv_prop_minus1 = self.sess.run(self.adv_prop_minus1)
                            adv_prop_0 = self.sess.run(self.adv_prop_0)
                            adv_rho_plus1 = np.log(adv_prop_0/adv_prop_plus1)
                            adv_rho_minus1 = np.log(adv_prop_0/adv_prop_minus1)
                            adv_rho_plus1 = np.clip(adv_rho_plus1,-self.wetCost,self.wetCost)
                            adv_rho_minus1 = np.clip(adv_rho_minus1,-self.wetCost,self.wetCost)
                            adv_stego = get_stego(atk_cover,adv_rho_plus1,adv_rho_minus1,payload = self.attack_payload)
                            break
                    self.sess.run(self.minimizing_op)

            if self.debug: 
                loss = self.sess.run(self.loss)
                adv_loss = self.sess.run(self.adv_loss)
                valid_prop_loss = self.sess.run(self.valid_prop_loss)
                payload_loss = self.sess.run(self.payload_loss)
                l0_loss = self.sess.run(self.l0_loss)
                print("End of batch ",i,"\tloss:",format(loss,'.4f'),
                          "\tadv_loss:",format(adv_loss,'.4f'),
                          "\tvalid_prop_loss",format(valid_prop_loss,'.4f'),
                          "\tpayload_loss",format(payload_loss,'.4f'),
                          "\tl0_loss",format(l0_loss,'.4f')
                          )   
                # self.writer.add_summary(self.sess.run(self.merge_summary) ,i)  
                current_adv_prop_0 =  np.array(self.sess.run(self.adv_prop_0)).flatten()
                current_adv_prop_plus1 = np.array(self.sess.run(self.adv_prop_plus1)).flatten()
                current_adv_prop_minus1 = np.array(self.sess.run(self.adv_prop_minus1)).flatten()

                # print('prop_0 changes max:',np.max(np.abs(current_adv_prop_0-last_adv_prop_0)))
                # print('prop_p1 changes max:',np.max(np.abs(current_adv_prop_plus1-last_adv_prop_plus1)))
                # print('prop_m1 changes max:',np.max(np.abs(current_adv_prop_minus1-last_adv_prop_minus1)))
                
                print('prop_p1 changes max:',np.max(np.abs(current_adv_prop_plus1-np.array(ori_prop_plus1).flatten())),'\tprop_m1 changes max:',np.max(np.abs(current_adv_prop_minus1-np.array(ori_prop_minus1).flatten())))   
                
                argmax_index = np.argmax(current_adv_prop_minus1)
                print('[propm1 prop_0 prop_p1] argmax prop_m1: [',current_adv_prop_minus1[argmax_index],current_adv_prop_0[argmax_index],current_adv_prop_plus1[argmax_index],']')
                argmax_index = np.argmax(current_adv_prop_0)
                print('[propm1 prop_0 prop_p1] argmax prop_0: [',current_adv_prop_minus1[argmax_index],current_adv_prop_0[argmax_index],current_adv_prop_plus1[argmax_index],']')
                argmax_index = np.argmax(current_adv_prop_plus1)
                print('[propm1 prop_0 prop_p1] argmax prop_p1: [',current_adv_prop_minus1[argmax_index],current_adv_prop_0[argmax_index],current_adv_prop_plus1[argmax_index],']')
            
            if np.sum(adv_stego!=atk_cover)!=0:
                break
            
        if np.sum(adv_stego!=atk_cover)!=0:
            if self.debug == 1:
                print("success")
            oa_perturbation = adv_stego - atk_cover
            oa_perturbation = np.asarray(oa_perturbation).flatten()
            oa_l0 = np.linalg.norm(oa_perturbation,ord=0)
            oa_l1 = np.linalg.norm(oa_perturbation,ord=1)
            oa_l2 = np.linalg.norm(oa_perturbation,ord=2)
            oa_linf = np.linalg.norm(oa_perturbation,ord=np.inf)
            if self.debug:
                print("oa_l0:",oa_l0,"\toa_l1:",oa_l1,"\toa_l2:",oa_l2,"\toa_linf:",oa_linf)
            assert np.any(oa_l0!=0)
            self.success_count = self.success_count + 1
            self.oa_l0_summary += oa_l0
            self.oa_l1_summary += oa_l1
            self.oa_l2_summary += oa_l2
            self.oa_linf_summary += oa_linf        
        else:
            if self.debug:
                print("fail")  
        end_time = time.time()
        if self.debug:
            print("attack time: %.2f"%(end_time - start_time))
        self.time_count = self.time_count + end_time - start_time        
        return np.asarray(adv_stego,dtype=np.uint8),np.asarray(adv_rho_plus1,dtype=np.float64),np.asarray(adv_rho_minus1,dtype=np.float64)
    
    def print_img_info(self,img,img_name = 'temp'):
        img = np.asarray(img)
        print(img_name," size:", np.shape(img))
        print(img_name," max value:",np.max(img))
        print(img_name," min value:",np.min(img))
        print(img_name," non-zero number:",np.sum(img!=0))
    
    def __del__(self):
        print("total count:",self.total_count)
        print("attack count:",self.attack_count)
        print("success count:",self.success_count)
        if self.attack_count!=0:
            print("attack success rate:",(self.success_count)/float(self.attack_count))
            print("average attack time:",self.time_count/float(self.attack_count))
            if self.success_count!=0:
                print("average oa_l0 loss:",self.oa_l0_summary/(float(self.success_count)))
                print("average oa_l1 loss:",self.oa_l1_summary/(float(self.success_count)))
                print("average oa_l2 loss:",self.oa_l2_summary/(float(self.success_count)))
                print("average oa_linf loss:",self.oa_linf_summary/(float(self.success_count)))
        else:
            print("no attack launched")
        # if self.debug:
        #     self.writer.close()
        self.sess.close()