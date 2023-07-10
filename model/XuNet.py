# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from .Model import Model

SRM_Kernels = np.load('./SRM_Kernels.npy')

class XuNet(Model):
    def __init__(self,is_training=None,data_format = 'NCHW',bn_epsilon=0.0001,reuse=None):
        super(XuNet,self).__init__(is_training=is_training,data_format=data_format)
        self.bn_epsilon = bn_epsilon
        self.reuse = reuse
    
    def _build_model(self,inputs):
        assert inputs.dtype == tf.float32
        if self.data_format == 'NCHW':
            _inputs = tf.transpose(inputs, [0, 3, 1, 2])
        else:
            _inputs = inputs
  
        z_c = tf.nn.conv2d(_inputs, SRM_Kernels, strides=[1, 1, 1, 1], padding='SAME',data_format=self.data_format)        
        #print("z_c shape" , z_c.get_shape().as_list())

        with tf.variable_scope("xunet",reuse=self.reuse):
            with tf.variable_scope("conv1"):
                f_conv2 = my_conv_layer(bn_epsilon=self.bn_epsilon, in1=z_c, filter_height=5,filter_width= 5, size_in = 30,size_out= 8,pooling_size= 5,stride_size= 1,active= 1,fabs= 1, padding_type='SAME',is_training = self.is_training,arg_scope= "xunet"+"1", reuse=self.reuse,data_format=self.data_format)
                #print("f_conv2 shape" , f_conv2.get_shape().as_list())
            with tf.variable_scope("conv2"):
                f_conv3 = my_conv_layer(bn_epsilon=self.bn_epsilon, in1=f_conv2,filter_height= 3, filter_width=3, size_in =8, size_out=16, pooling_size=5,stride_size=  2, active=1,fabs= 0,  padding_type='SAME', is_training = self.is_training,arg_scope="xunet"+"2", reuse=self.reuse,data_format=self.data_format)
                #print("f_conv3 shape" , f_conv3.get_shape().as_list())
            with tf.variable_scope("conv3"):
                f_conv4 = my_conv_layer(bn_epsilon=self.bn_epsilon,in1= f_conv3, filter_height=1, filter_width=1,size_in= 16, size_out=32, pooling_size=5, stride_size=2,active= 0, fabs= 0, padding_type='SAME', is_training = self.is_training, arg_scope="xunet"+"3", reuse=self.reuse,data_format=self.data_format)
                #print("f_conv4 shape" , f_conv4.get_shape().as_list())
            with tf.variable_scope("conv4"):
                f_conv5 = my_conv_layer(self.bn_epsilon, f_conv4, 1, 1, 32, 64, 5, 2, 0, 0, 'SAME', self.is_training, "xunet"+"4",  self.reuse,self.data_format)
                #print("f_conv5 shape" , f_conv5.get_shape().as_list())
            with tf.variable_scope("conv5"):
                f_conv6 = my_conv_layer(self.bn_epsilon, f_conv5, 1, 1, 64, 128, 5, 2, 0, 0, 'SAME',  self.is_training, "xunet"+"5",self.reuse,self.data_format)
                #print("f_conv6 shape" , f_conv6.get_shape().as_list())
            with tf.variable_scope("conv6"):
                f_conv7 = my_conv_layer(self.bn_epsilon, f_conv6, 1, 1, 128, 256, 16, 1, 0, 0, 'VALID', self.is_training,"xunet"+"6", self.reuse,self.data_format)
                #print("f_conv7 shape" , f_conv7.get_shape().as_list())
            with tf.variable_scope("conv7"):
                f_conv8 = tf.contrib.layers.flatten(f_conv7)
                logits = tf.contrib.layers.fully_connected(inputs=f_conv8,num_outputs=2, 
                        activation_fn=None, normalizer_fn=None, 
                        weights_initializer=tf.random_normal_initializer(mean=0., stddev=0.01), 
                        biases_initializer=tf.constant_initializer(0.), scope="conv7")
                
        self.logits = tf.cast(logits,tf.float32)
        self.softmax = tf.nn.softmax(self.logits,axis=1)
        self.forward = tf.argmax(self.logits, 1, output_type=tf.int32)
        self.softmax_before_fc = tf.nn.softmax(f_conv8,axis=1)
        return self.logits


def my_conv_layer(bn_epsilon, in1, filter_height, filter_width, size_in, size_out, pooling_size, stride_size, active, fabs, padding_type, is_training,arg_scope, reuse,data_format):
    w_conv = tf.get_variable('conv_w', shape=[filter_height, filter_width, size_in, size_out], 
        initializer=tf.truncated_normal_initializer(stddev=0.1))
    z_conv = tf.nn.conv2d(input=in1, filter=w_conv, strides=[1, 1, 1, 1], padding='SAME',data_format=data_format)

    if fabs==1:
        z_conv = tf.abs(z_conv)

    bn_conv = bn(bn_epsilon, z_conv,is_training,  reuse,data_format)

    if active == 1:
        f_conv = tf.nn.tanh(bn_conv)
    else:
        f_conv = tf.nn.relu(bn_conv)

    # Average pooling
    out = tf.nn.avg_pool(f_conv, ksize=[1,1 , pooling_size, pooling_size], 
        strides=[1, 1, stride_size, stride_size], padding=padding_type,data_format=data_format)
    
    return out


def bn(bn_epsilon, x, is_training, reuse, use_bias=False, decay=0.9,data_format='NCHW'):
    param_initializers = {'beta': tf.zeros_initializer, 'gamma': tf.ones_initializer()}
    bn_conv = tf.contrib.layers.batch_norm(inputs=x, decay=decay, center=True, scale=True, epsilon=bn_epsilon, is_training = is_training ,reuse=reuse, param_initializers=param_initializers,  scope='batch_norm',fused = True,data_format=data_format)
    return bn_conv
