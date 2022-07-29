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
        self.gamma_x = tf.Variable(1.)
        self.loggamma_x = tf.Variable(1.)
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
       # ones = tf.ones([100] + image_size[0:-1] + [self.category_count])
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
            HALF_LOG_TWO_PI = tf.constant(0.91893,  dtype=tf.float32)# added
            reconstruction_loss = reconstruction_loss / self.gamma_x + self.loggamma_x + HALF_LOG_TWO_PI #added
            if self.reconstruction_loss_tracker.result() > 0:
                reconstruction_loss = tf.minimum(self.reconstruction_loss_tracker.result(), self.reconstruction_loss_tracker.result()*.99 + reconstruction_loss *.01) #min between cumulated reconstruction loss and this batch.
            total_loss = reconstruction_loss + kl_loss
            total_loss = tf.reduce_mean(total_loss)

        self.gamma_x.assign( tf.sqrt(tf.reduce_mean(reconstruction_loss)))#added
        self.loggamma_x.assign( tf.math.log(self.gamma_x)) #ådded

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.total_loss_no_weights_tracker.update_state(total_loss_no_weights)
        #tf.print(self.gamma_x)
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


class SecondStage(CVAE):

    def __init__(self, encoder, decoder, beta, shape, category_count, **kwargs):
        super(SecondStage, self).__init__(encoder2, decoder2, second_depth, second_dim)

    # def extract_posterior(self, sess, x):
    #     #num_sample = np.shape(x)[0] lenght of dataset
    #     #num_iter = math.ceil(float(num_sample) / float(self.batch_size)) 
    #     num_iter = 1000 #fixed for now
    #     x_extend = np.concatenate([x, x[0:self.batch_size]], 0)
    #     mu_z, sd_z = [], []
    #     for i in range(num_iter):
    #         mu_z_batch, sd_z_batch = sess.run([self.mu_z, self.sd_z], feed_dict={self.raw_x: x_extend[i*self.batch_size:(i+1)*self.batch_size], self.is_training: False})
    #         mu_z.append(mu_z_batch)
    #         sd_z.append(sd_z_batch)
    #     mu_z = np.concatenate(mu_z, 0)[0:num_sample]
    #     sd_z = np.concatenate(sd_z, 0)[0:num_sample]
        # return mu_z, sd_z

    def loss2(self):
        HALF_LOG_TWO_PI = 0.91893
        self.loggamma_z = tf.log(self.gamma_z)
        self.kl_loss2 = tf.reduce_sum(tf.square(self.mu_u) + tf.square(self.sd_u) - 2 * self.logsd_u - 1) / 2.0 / float(self.batch_size)
        self.mse_loss2 = tf.losses.mean_squared_error(self.z, self.z_hat)
        self.gen_loss2 = tf.reduce_sum(tf.square((self.z - self.z_hat) / self.gamma_z) / 2.0 + self.loggamma_z + HALF_LOG_TWO_PI) / float(self.batch_size)
        self.loss2 = self.kl_loss2 + self.gen_loss2 



#add: (1st stage)
# initialize gamma_x = 1
# initalize mseloss = 1

# mseloss = min(mseloss, mseloss*.99+ reconstruction_loss*.01) #bmseloss: batch (mean) reconstruction error
# gamma_x = np.sqrt(mseloss) 

# -> reconstruction_loss = tf.reduce_sum(tf.square((self.x - self.x_hat) / self.gamma_x) / 2.0 + self.loggamma_x + HALF_LOG_TWO_PI) / float(self.batch_size)

#add: (2nd stage)
# mseloss2 = np.mean(np.square(mu_z - z_hat), axis = (0,1))
# gamma_z = np.sqrt(mseloss2)

# encoder2: takes as input z, gives u_mean and u_log_var
# decoder2: takes as input u, gives z_hat (same dimension as mu_z)

#extract_posterior: takes x as input, gives mu_z (z_mean) and sd_z (z_log_var)

#
#self.z == z_cond