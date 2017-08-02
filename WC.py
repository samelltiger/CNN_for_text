import tensorflow as tf
import numpy as np
class Weighted_CNN(object):
    """
    create parameters and step function
    """
    def __init__(self, incoming,max_height,channel, fsize, fnumber,):
        
        def orthogonal(self, shape,scale = 1.0):
            #https://github.com/Lasagne/Lasagne/blob/master/lasagne/init.py
            flat_shape = (shape[0], np.prod(shape[1:]))
            a = np.random.normal(0.0, 1.0, flat_shape)
            u, _, v = np.linalg.svd(a, full_matrices=False)
            q = u if u.shape == flat_shape else v
            q = q.reshape(shape) #this needs to be corrected to float32
            return tf.Variable(scale * q[:shape[0], :shape[1]], dtype=tf.float32,trainable=True)
        def weight_init(self, shape):
            initial = tf.random_uniform(shape,minval=-np.sqrt(5)*np.sqrt(1.0/shape[0]), maxval=np.sqrt(5)*np.sqrt(1.0/shape[0]))
            return tf.Variable(initial,trainable=True)
        
        # var
        self.incoming = incoming
        self.max_height = max_height
        self.channel = channel
        self.fsize = fsize
        self.fnumber = fnumber
        
        # context conv parameters
        W_C = self.weight_init([self.fsize, self.max_height, self.channel, self.fnumber])
        b_C = tf.Variable(tf.constant(0.001, shape=shape))
        # context conv
        C_conv  = tf.nn.relu(tf.nn.conv2d(self.incoming, W_C, strides=[1, 1, 1, 1], padding="VALID") + b_C)
            
        # weight conv parameters
        W_W = self.orthogonal([self.fsize, self.max_height, self.channel, self.fnumber])
        b_W = tf.Variable(tf.constant(0.001, shape=shape))
        # context conv
        W_conv  = tf.nn.softmax(tf.nn.conv2d(self.incoming, W_C, strides=[1, 1, 1, 1], padding="VALID") + b_C)   
        
        # weighted conv
        weighted_conv = C_conv*W_conv
        # weighted pooling
        print("using weighted average pooling CNN")
        weighted_conv_reshape = tf.reshape(weighted_conv, [-1, self.max_height-self.fsize+1, self.fnumber])
        self.output = tf.reduce_sum(weighted_conv_reshape,axis=1)
        # L2
        self.L2 = tf.nn.l2_loss(W_C)+tf.nn.l2_loss(W_W)
        
