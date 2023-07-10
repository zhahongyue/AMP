import tensorflow as tf
from model import Model

class AdvAttack:
    def __init__(self, 
                 attack_model_class=None, 
                 attack_steg_type='none',
                 attack_model_path=None,
                 data_format='NCHW',**kwargs):
        """ class for adversarial attacks
        Args:
            attack_model_class ([type], optional): [steganalysis model type]. Defaults to None.
            attack_steg_type (str, optional): [steganography distortion function name]. Defaults to 'none'.
            attack_model_path ([type], optional): [steganalysis model checkpoint path]. Defaults to None.
            data_format (str, optional): [data formation]. Defaults to 'NCHW'.
        """
        self.attack_model = attack_model_class(False,data_format)
        self.attack_steg_type = attack_steg_type
        self.attack_model_path = attack_model_path
        self.data_format = data_format

    def _find_adv_img(self,**kwargs):
        raise NotImplementedError('Implement your adversarial method.')
    
    def self_concat(self, tensor, times=2, axis=0 ):
        tensor_list = []
        for i in range(times):
            tensor_list.append(tensor)
        return tf.concat(tensor_list,axis=axis)
    
