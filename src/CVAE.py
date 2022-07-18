from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf #could sub this
import numpy as np #also probably

class CVAE(keras.Model):
    def __init__(self, encoder, decoder, beta, shape, category_count, **kwargs):
        super(CVAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta
        self.shape = shape
        self.category_count = category_count
        self.total_loss_tracker = keras.metrics.Mean(name="loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.total_loss_no_weights_tracker = keras.metrics.Mean(name="loss_no_weights")
        #
        self.val_total_loss_tracker = keras.metrics.Mean(name="val_loss")
        self.val_reconstruction_loss_tracker = keras.metrics.Mean(
            name="val_reconstruction_loss")
        self.val_kl_loss_tracker = keras.metrics.Mean(name="val_kl_loss")
        self.val_total_loss_no_weights_tracker = keras.metrics.Mean(name="val_loss_no_weights")

    
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.val_total_loss_tracker,
            self.val_reconstruction_loss_tracker,
            self.val_kl_loss_tracker,
            self.total_loss_no_weights_tracker,
            self.val_total_loss_no_weights_tracker,
        ]
       
    def call(self, inputs):
        _, input_label, conditional_input = self.conditional_input(inputs)
        z_mean, z_log_var = self.encoder(conditional_input)
        z_cond = self.sampling(z_mean, z_log_var, input_label)
        return self.decoder(z_cond)


    def conditional_input(self, inputs):
        image_size = [self.shape[0], self.shape[1], self.shape[2]]
        input_img = layers.InputLayer(input_shape=image_size,
                                    dtype ='float32')(inputs[0])
        input_label = layers.InputLayer(input_shape=(self.category_count, ),
                                        dtype ='float32')(inputs[1])
        labels = tf.reshape(inputs[1], [-1, 1, 1, self.category_count])
        labels = tf.cast(labels, dtype='float32')
        ones = tf.ones([inputs[0].shape[0]] + image_size[0:-1] + [self.category_count])
        labels = ones * labels

        conditional_input = layers.Concatenate(axis=3)([input_img, labels]) 
        return  input_img, input_label, conditional_input

    def sampling(self, z_mean, z_log_var, input_label):
        if len(input_label.shape) == 1:
            input_label = np.expand_dims(input_label, axis=0)

        eps = tf.random.normal(tf.shape(z_log_var), dtype=tf.float32,
                               mean=0., stddev=1.0, name='epsilon')
        z = z_mean + tf.exp(z_log_var / 2) * eps
        z_cond = tf.concat([z, input_label], axis=1)
        return z_cond

    @tf.function
    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            input_img, input_label, conditional_input = self.conditional_input(data)
            z_mean, z_log_var = self.encoder(conditional_input)
            z_cond = self.sampling(z_mean, z_log_var, input_label)
            reconstruction = self.decoder(z_cond)

            reconstruction_loss = tf.reduce_sum(
                 keras.losses.MSE(input_img, # removed np.prod(self.shape) *
                                    reconstruction), axis=(1, 2))            
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean)
                      - tf.exp(z_log_var))

            kl_loss = tf.reduce_sum(kl_loss, axis=1) 
            total_loss_no_weights = reconstruction_loss + kl_loss
            total_loss_no_weights = tf.reduce_mean(total_loss_no_weights)
            kl_loss = self.beta * kl_loss
            total_loss = reconstruction_loss + kl_loss
            total_loss = tf.reduce_mean(total_loss) 
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.total_loss_no_weights_tracker.update_state(total_loss_no_weights)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "loss_no_weights": self.total_loss_no_weights_tracker.result(),

        }

    @tf.function
    def test_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        
        input_img, input_label, conditional_input = self.conditional_input(data)
        z_mean, z_log_var = self.encoder(conditional_input)
        z_cond = self.sampling(z_mean, z_log_var, input_label)
        reconstruction = self.decoder(z_cond)
        reconstruction_loss = tf.reduce_sum(
                 keras.losses.MSE(input_img,
                                    reconstruction), axis=(1, 2))   
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean)
                  - tf.exp(z_log_var))
        kl_loss = tf.reduce_sum(kl_loss, axis=1)
        total_loss_no_weights = reconstruction_loss + kl_loss
        total_loss_no_weights = tf.reduce_mean(total_loss_no_weights)
        kl_loss = self.beta * kl_loss
        total_loss =  reconstruction_loss + kl_loss
        total_loss = tf.reduce_mean(total_loss)
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.total_loss_no_weights_tracker.update_state(total_loss_no_weights)
        return{
            'loss': self.total_loss_tracker.result(),
            'reconstruction_loss': self.reconstruction_loss_tracker.result(),
            'kl_loss': self.kl_loss_tracker.result(),
            "loss_no_weights": self.total_loss_no_weights_tracker.result()
        }


# class TwoStageVaeModel(CVAE):

#     def __init__(self, encoder, decoder, beta, shape, category_count, **kwargs):
#         super(TwoStageVaeModel, self).__init__(encoder2, decoder2, second_depth, second_dim)

#     def extract_posterior(self, sess, x):
#         #num_sample = np.shape(x)[0] lenght of dataset
#         #num_iter = math.ceil(float(num_sample) / float(self.batch_size)) 
#         num_iter = 1000 #fixed for now
#         x_extend = np.concatenate([x, x[0:self.batch_size]], 0)
#         mu_z, sd_z = [], []
#         for i in range(num_iter):
#             mu_z_batch, sd_z_batch = sess.run([self.mu_z, self.sd_z], feed_dict={self.raw_x: x_extend[i*self.batch_size:(i+1)*self.batch_size], self.is_training: False})
#             mu_z.append(mu_z_batch)
#             sd_z.append(sd_z_batch)
#         mu_z = np.concatenate(mu_z, 0)[0:num_sample]
#         sd_z = np.concatenate(sd_z, 0)[0:num_sample]
#         return mu_z, sd_z



# self.is_training = tf.placeholder(tf.bool, [], 'is_training')