import tensorflow as tf 
from tensorflow.contrib import layers 
import math 
import numpy as np 
from tensorflow.python.training.moving_averages import assign_moving_average
from tensorflow.keras import layers
from tensorflow.keras import regularizers

def conv2d(input_, output_dim, k_h, k_w, d_h, d_w, stddev=0.02, name="conv2d",
           initializer=tf.truncated_normal_initializer):
  with tf.variable_scope(name):
    w = tf.get_variable(
        "w", [k_h, k_w, input_.get_shape()[-1], output_dim],
        initializer=initializer(stddev=stddev))
    conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding="SAME")
    biases = tf.get_variable(
        "biases", [output_dim], initializer=tf.constant_initializer(0.0))
    return tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32, tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(bias_start))
        return tf.matmul(input_, matrix) + bias

def lrelu(input_, leak=0.2, name="lrelu"):
      return tf.maximum(input_, leak * input_, name=name)

def batch_norm(x, is_training, scope, eps=1e-5, decay=0.999, affine=True):
    def mean_var_with_update(moving_mean, moving_variance):
        if len(x.get_shape().as_list()) == 4:
            statistics_axis = [0, 1, 2]
        else:
            statistics_axis = [0]
        mean, variance = tf.nn.moments(x, statistics_axis, name='moments')
        with tf.control_dependencies([assign_moving_average(moving_mean, mean, decay), assign_moving_average(moving_variance, variance, decay)]):
            return tf.identity(mean), tf.identity(variance)

    with tf.name_scope(scope):
        with tf.variable_scope(scope + '_w'):
            params_shape = x.get_shape().as_list()[-1:]
            moving_mean = tf.get_variable('mean', params_shape, initializer=tf.zeros_initializer(), trainable=False)
            moving_variance = tf.get_variable('variance', params_shape, initializer=tf.ones_initializer, trainable=False)

            mean, variance = tf.cond(is_training, lambda: mean_var_with_update(moving_mean, moving_variance), lambda: (moving_mean, moving_variance))
            if affine:
                beta = tf.get_variable('beta', params_shape, initializer=tf.zeros_initializer())
                gamma = tf.get_variable('gamma', params_shape, initializer=tf.ones_initializer)
                return tf.nn.batch_normalization(x, mean, variance, beta, gamma, eps)
            else:
                return tf.nn.batch_normalization(x, mean, variance, None, None, eps)


def deconv2d(input_, output_shape, k_h, k_w, d_h, d_w, stddev=0.02, name="deconv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable("w", [k_h, k_w, output_shape[-1], input_.get_shape()[-1]], initializer=tf.random_normal_initializer(stddev=stddev))
        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])
        biases = tf.get_variable("biases", [output_shape[-1]], initializer=tf.constant_initializer(0.0))
    return tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

def downsample(x, out_dim, kernel_size, name, l2_reg=None):
    with tf.variable_scope(name):
        input_shape = x.get_shape().as_list()
        assert(len(input_shape) == 4)
        kernel_reg = None
        if l2_reg != None:
          kernel_reg = regularizers.l2(l2_reg)
        return (tf.keras.layers.Conv2D(out_dim, kernel_size, strides=2, padding='same',kernel_regularizer=kernel_reg)(x))

def upsample(x, out_dim, kernel_size, name, l2_reg=None):
    with tf.variable_scope(name):
        input_shape = x.get_shape().as_list()
        assert(len(input_shape) == 4)
        kernel_reg = None
        if l2_reg != None:
          kernel_reg = regularizers.l2(l2_reg)
        return (tf.keras.layers.Conv2DTranspose(out_dim, kernel_size, strides=2, padding='same',kernel_regularizer=kernel_reg)(x))

def res_block(x, out_dim, is_training, name, depth=2, kernel_size=3):
    with tf.variable_scope(name):
        y = x
        for i in range(depth):
            y = tf.nn.relu(batch_norm(y, is_training, 'bn'+str(i)))
            y = tf.layers.conv2d(y, out_dim, kernel_size, padding='same', name='layer'+str(i))
        s = tf.layers.conv2d(x, out_dim, kernel_size, padding='same', name='shortcut')
        return y + s 


def res_fc_block(x, out_dim, name, depth=2):
    with tf.variable_scope(name):
        y = x 
        for i in range(depth):
            y = tf.layers.dense(tf.nn.relu(y), out_dim, name='layer'+str(i))
        s = tf.layers.dense(x, out_dim, name='shortcut')
        return y + s 


def scale_block(x, out_dim, is_training, name, block_per_scale=1, depth_per_block=2, kernel_size=3):
    with tf.variable_scope(name):
        y = x 
        for i in range(block_per_scale):
            y = res_block(y, out_dim, is_training, 'block'+str(i), depth_per_block, kernel_size)
        return y 


def scale_fc_block(x, out_dim, name, block_per_scale=1, depth_per_block=2):
    with tf.variable_scope(name):
        y = x 
        for i in range(block_per_scale):
            y = res_fc_block(y, out_dim, 'block'+str(i), depth_per_block)
        return y 


### Model

import tensorflow as tf 
import math 
import numpy as np 
from tensorflow.python.training.moving_averages import assign_moving_average


class TwoStageVaeModel(object):
    def __init__(self, x, latent_dim=64, second_depth=3, second_dim=1024):
        self.raw_x = x
        self.x = tf.cast(self.raw_x, tf.float32) / 255.0 
        self.batch_size = x.get_shape().as_list()[0]
        self.latent_dim = latent_dim
        self.second_dim = second_dim
        self.img_dim = x.get_shape().as_list()[1]
        self.second_depth = second_depth

        self.is_training = tf.placeholder(tf.bool, [], 'is_training')

        self.gamma_x = tf.placeholder(tf.float32, [], 'gamma_x')
        self.gamma_z = tf.placeholder(tf.float32, [], 'gamma_z')

        self.__build_network()
        self.__build_loss()
        self.__build_summary()
        self.__build_optimizer()

    def __build_network(self):
        with tf.variable_scope('stage1'):
            self.build_encoder1()
            self.build_decoder1()
        with tf.variable_scope('stage2'):
            self.build_encoder2()
            self.build_decoder2()

    def __build_loss(self):
        HALF_LOG_TWO_PI = 0.91893
        k = (2*self.img_dim/self.latent_dim)**2
        self.kl_loss1 = tf.reduce_sum(tf.square(self.mu_z) + tf.square(self.sd_z) - 2 * self.logsd_z - 1) / 2.0 / float(self.batch_size)
        self.mse_loss1 = tf.losses.mean_squared_error(self.x, self.x_hat)
        self.loggamma_x = tf.log(self.gamma_x)
        self.gen_loss1 = tf.reduce_sum(tf.square((self.x - self.x_hat) / self.gamma_x) / 2.0 + self.loggamma_x + HALF_LOG_TWO_PI) / float(self.batch_size)
        self.loss1 = k*self.kl_loss1 + self.gen_loss1 

        self.loggamma_z = tf.log(self.gamma_z)
        self.kl_loss2 = tf.reduce_sum(tf.square(self.mu_u) + tf.square(self.sd_u) - 2 * self.logsd_u - 1) / 2.0 / float(self.batch_size)
        self.mse_loss2 = tf.losses.mean_squared_error(self.z, self.z_hat)
        self.gen_loss2 = tf.reduce_sum(tf.square((self.z - self.z_hat) / self.gamma_z) / 2.0 + self.loggamma_z + HALF_LOG_TWO_PI) / float(self.batch_size)
        self.loss2 = self.kl_loss2 + self.gen_loss2 

    def __build_summary(self):
        with tf.name_scope('stage1_summary'):
            self.summary1 = []
            self.summary1.append(tf.summary.scalar('gamma', self.gamma_x))
            self.summary1 = tf.summary.merge(self.summary1)

        with tf.name_scope('stage2_summary'):
            self.summary2 = []
            self.summary2.append(tf.summary.scalar('gamma', self.gamma_z))
            self.summary2 = tf.summary.merge(self.summary2)

    def __build_optimizer(self):
        all_variables = tf.global_variables()
        variables1 = [var for var in all_variables if 'stage1' in var.name]
        variables2 = [var for var in all_variables if 'stage2' in var.name]
        self.lr = tf.placeholder(tf.float32, [], 'lr')
        self.global_step = tf.get_variable('global_step', [], tf.int32, tf.zeros_initializer(), trainable=False)
        self.opt1 = tf.train.AdamOptimizer(self.lr).minimize(self.loss1, self.global_step, var_list=variables1)
        self.opt2 = tf.train.AdamOptimizer(self.lr).minimize(self.loss2, self.global_step, var_list=variables2)
        
    def build_encoder2(self):
        with tf.variable_scope('encoder'):
            t = self.z 
            for i in range(self.second_depth):
                t = tf.layers.dense(t, self.second_dim, tf.nn.relu, name='fc'+str(i))
            t = tf.concat([self.z, t], -1)
        
            self.mu_u = tf.layers.dense(t, self.latent_dim, name='mu_u')
            self.logsd_u = tf.layers.dense(t, self.latent_dim, name='logsd_u')
            self.sd_u = tf.exp(self.logsd_u)
            self.u = self.mu_u + self.sd_u * tf.random_normal([self.batch_size, self.latent_dim])
        
    def build_decoder2(self):
        with tf.variable_scope('decoder'):
            t = self.u 
            for i in range(self.second_depth):
                t = tf.layers.dense(t, self.second_dim, tf.nn.relu, name='fc'+str(i))
            t = tf.concat([self.u, t], -1)
            self.z_hat = tf.layers.dense(t, self.latent_dim, name='z_hat')

    def extract_posterior(self, sess, x):
        num_sample = np.shape(x)[0]
        num_iter = int(math.ceil(float(num_sample) / float(self.batch_size)))
        x_extend = np.concatenate([x, x[0:self.batch_size]], 0)
        mu_z, sd_z = [], []
        for i in range(num_iter):
            mu_z_batch, sd_z_batch = sess.run([self.mu_z, self.sd_z], feed_dict={self.raw_x: x_extend[i*self.batch_size:(i+1)*self.batch_size], self.is_training: False})
            mu_z.append(mu_z_batch)
            sd_z.append(sd_z_batch)
        mu_z = np.concatenate(mu_z, 0)[0:num_sample]
        sd_z = np.concatenate(sd_z, 0)[0:num_sample]
        return mu_z, sd_z

    def step(self, stage, input_batch, gamma, lr, sess, writer=None, write_iteration=600):
        if stage == 1:
            loss, summary, mse_loss,_ = sess.run([self.loss1, self.summary1, self.mse_loss1, self.opt1], feed_dict={self.raw_x: input_batch, self.gamma_x: gamma, self.lr: lr, self.is_training: True})
        elif stage == 2:
            loss, summary, mse_loss,_ = sess.run([self.loss2, self.summary2, self.mse_loss2, self.opt2], feed_dict={self.z: input_batch, self.gamma_z:gamma,self.lr: lr, self.is_training: True})
        else:
            raise Exception('Wrong stage {}.'.format(stage))
        global_step = self.global_step.eval(sess)
        if global_step % write_iteration == 0 and writer is not None:
            writer.add_summary(summary, global_step)
        return loss, mse_loss

    def reconstruct2(self, sess, z):
        #reconstruction of latent space by the second stage
        num_sample = np.shape(z)[0]
        num_iter = int(math.ceil(float(num_sample) / float(self.batch_size)))
        z_extend = np.concatenate([z, z[0:self.batch_size]], 0)
        recon_z = []
        for i in range(num_iter):
            recon_z_batch = sess.run(self.z_hat, feed_dict={self.z: z_extend[i*self.batch_size:(i+1)*self.batch_size], self.is_training: False})
            recon_z.append(recon_z_batch)
        recon_z = np.concatenate(recon_z, 0)[0:num_sample]
        return recon_z
    
    def reconstruct(self, sess, x):
        num_sample = np.shape(x)[0]
        num_iter = int(math.ceil(float(num_sample) / float(self.batch_size)))
        x_extend = np.concatenate([x, x[0:self.batch_size]], 0)
        recon_x = []
        mu_z_tot = []
        logsd_z_tot = []
        for i in range(num_iter):
            #get mu_z and logsd_z for every batch of data
            mu_z_batch, logsd_z_batch, recon_x_batch = sess.run([self.mu_z, self.logsd_z, self.x_hat], feed_dict={self.raw_x: x_extend[i*self.batch_size:(i+1)*self.batch_size], self.is_training: False}) 
            recon_x.append(recon_x_batch)
            mu_z_tot.append(mu_z_batch)
            logsd_z_tot.append(logsd_z_batch)
        recon_x = np.concatenate(recon_x, 0)[0:num_sample]
        mu_z_tot = np.concatenate(mu_z_tot, 0)[0:num_sample]
        logsd_z_tot = np.concatenate(logsd_z_tot, 0)[0:num_sample]
        #return recon_x
        return mu_z_tot, logsd_z_tot, recon_x

    def generate(self, sess, num_sample, stage=2, adjust2=None, adjust1=None):
        num_iter = int(math.ceil(float(num_sample) / float(self.batch_size)))
        gen_samples = []
        gen_z = []
        for i in range(num_iter):
            if stage == 2:
                # u ~ N(0, I)
                u = np.random.normal(0, 1, [self.batch_size, self.latent_dim])
                # z ~ N(f_2(u), \gamma_z I)
                z = sess.run(self.z_hat, feed_dict={self.u: u, self.is_training: False})
                if type(adjust2) == np.float32: #np.ndarray
                    #print("normalizing 2")
                    rescale = adjust2/np.mean(np.std(z,axis=0))
                    #print(rescale)
                    z = z*rescale
            else:
                z = np.random.normal(0, 1, [self.batch_size, self.latent_dim])
                if type(adjust2) == np.float32: #np.ndarray
                   z = z*adjust2
            # x = f_1(z)
            x = sess.run(self.x_hat, feed_dict={self.z: z, self.is_training: False})
            gen_z.append(z)
            gen_samples.append(x)
        gen_z = np.concatenate(gen_z, 0)
        gen_samples = np.concatenate(gen_samples, 0)
        if type(adjust1) == np.float:
            rescale = adjust1/np.mean(np.std(gen_samples,axis=0))
            gmean = np.mean(gen_samples,axis=0)
            gen_samples = (gen_samples - gmean)*rescale + gmean
            #need to remain in range 0-1
            gen_samples = np.maximum(np.minimum(gen_samples,1),0)
        return (gen_samples[0:num_sample],gen_z[0:num_sample])

class Resnet(TwoStageVaeModel):
    def __init__(self, x, num_scale, block_per_scale=1, depth_per_block=2, kernel_size=3, base_dim=16, fc_dim=512, latent_dim=64, second_depth=3, second_dim=1024, l2_reg=.001):
        self.num_scale = num_scale
        self.block_per_scale = block_per_scale
        self.depth_per_block = depth_per_block
        self.kernel_size = kernel_size 
        self.base_dim = base_dim 
        self.fc_dim = fc_dim
        self.l2_reg = l2_reg
        super(Resnet, self).__init__(x, latent_dim, second_depth, second_dim)

    def build_encoder1(self):
        with tf.variable_scope('encoder'):
            dim = self.base_dim
            y = tf.layers.conv2d(self.x, dim, self.kernel_size, 1, 'same', name='conv0')
            for i in range(self.num_scale):
                y = scale_block(y, dim, self.is_training, 'scale'+str(i), self.block_per_scale, self.depth_per_block, self.kernel_size)

                if i != self.num_scale - 1:
                    dim *= 2
                    y = downsample(y, dim, self.kernel_size, 'downsample'+str(i), self.l2_reg)
            
            y = tf.reduce_mean(y, [1, 2])
            y = scale_fc_block(y, self.fc_dim, 'fc', 1, self.depth_per_block)
            
            self.mu_z = tf.layers.dense(y, self.latent_dim)
            self.logsd_z = tf.layers.dense(y, self.latent_dim)
            self.sd_z = tf.exp(self.logsd_z)
            self.z = self.mu_z + tf.random_normal([self.batch_size, self.latent_dim]) * self.sd_z 

    def build_decoder1(self):
        desired_scale = self.x.get_shape().as_list()[1]
        scales, dims = [], []
        current_scale, current_dim = 2, self.base_dim 
        while current_scale <= desired_scale:
            scales.append(current_scale)
            dims.append(current_dim)
            current_scale *= 2
            current_dim = min(current_dim*2, 1024)
        assert(scales[-1] == desired_scale)
        dims = list(reversed(dims))

        with tf.variable_scope('decoder'):
            y = self.z 
            data_depth = self.x.get_shape().as_list()[-1]

            fc_dim = 2 * 2 * dims[0]
            y = tf.layers.dense(y, fc_dim, name='fc0')
            y = tf.reshape(y, [-1, 2, 2, dims[0]])

            for i in range(len(scales)-1):
                y = upsample(y, dims[i+1], self.kernel_size, 'up'+str(i), self.l2_reg)
                y = scale_block(y, dims[i+1], self.is_training, 'scale'+str(i), self.block_per_scale, self.depth_per_block, self.kernel_size)
            
            y = tf.layers.conv2d(y, data_depth, self.kernel_size, 1, 'same')
            self.x_hat = tf.nn.sigmoid(y)
