import tensorflow as tf

class Model:
    def __init__(self, 
                 is_training=None, 
                 data_format='NCHW',**kwargs):
        self.data_format = data_format
        if is_training is None:
            self.is_training = tf.get_variable('is_training', dtype=tf.bool,
                                    initializer=tf.constant_initializer(True),
                                    trainable=False)
        else:
            self.is_training = is_training

    def _build_model(self, inputs):
        raise NotImplementedError('Here is your model definition')

    def _build_losses(self, labels, lambda_reg = 1):
        with tf.variable_scope('loss'):
            oh = tf.one_hot(labels, 2, on_value=1.0,off_value=0.0)
            xen_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=oh,logits=self.logits))
            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            self.loss = tf.cast(tf.add_n([xen_loss] + lambda_reg * reg_losses),tf.float32)
            equal = tf.equal(self.forward, labels)
            self.accuracy = tf.reduce_mean(tf.cast(equal, tf.float32))
        return self.loss, self.accuracy

