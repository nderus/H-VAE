# generations
import tensorflow as tf

from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
import wandb

class Generations:
    def __init__(self, model, encoded_dim, category_count, input_shape, labels):
        self.model = model
        self.encoded_dim = encoded_dim
        self.category_count = category_count
        self.input_shape = input_shape
        self.labels = labels

    def reparametrization(self, z_mean, z_log_var, input_label):
        """ Performs the riparametrization trick"""

        eps = tf.random.normal(shape = (input_label.shape[0], self.encoded_dim),
                                mean = 0.0, stddev = 1.0)       
        z = z_mean + tf.math.exp(z_log_var * .5) * eps
        z_cond = tf.concat([z, input_label], axis=1) # (batch_size, label_dim + latent_dim)

        return z_cond


    def generations_class(self, digit_label=1, image_count = 10):
        _, axs = plt.subplots(2, image_count, figsize=(20, 4))
        for i in range(image_count):
            digit_label_one_hot = to_categorical(digit_label, self.category_count).reshape(1,-1)
            a = tf.convert_to_tensor(digit_label_one_hot)
            b = tf.concat([a, a], axis=0) # with 1 dimension, it fails...
            z_cond = self.reparametrization(z_mean=0, z_log_var=0, input_label = b) # TO DO: sub this with the sampling CVAE function
            decoded_x = self.model.decoder.predict(z_cond)
            digit_0 = decoded_x[0].reshape(self.input_shape) 
            digit_1 = decoded_x[1].reshape(self.input_shape) 
            axs[0, i].imshow(digit_0)
            axs[0, i].axis('off')
            if len(self.labels) <= 10:
                axs[0, i].set_title(self.labels[digit_label])
            axs[1, i].imshow(digit_1)
            axs[1, i].axis('off')
            if len(self.labels) <= 10:
                axs[1, i].set_title(self.labels[digit_label])
            #wandb.log({"Generations: {}".format(digit_label): wandb.Image(plt)})
        wandb.log({"Generations": wandb.Image(plt, caption="Class:{}_{}".format(digit_label, self.labels[digit_label])) }) #

    def generations_celeba(self, target_attr):
        image_count = 10

        _, axs = plt.subplots(2, image_count, figsize=(12, 3))
        for j in range(2):
            for i in range(image_count):

                attr_vect = np.zeros(40)
                for attr in target_attr:
                    attr_vect[attr] = 1

                random_sample = tf.random.normal(shape = (1, self.encoded_dim))
                digit_label_one_hot= np.array([attr_vect], dtype='float32')


                decoded_x = self.modeldecoder.predict([random_sample,digit_label_one_hot])
                digit = decoded_x[0].reshape(self.input_shape)
                axs[j, i].imshow(digit)
                axs[j, i].axis('off')

                attributes = str(self.labels[target_attr].tolist())
        #wandb.log({"Generations:_{}".format(attributes): wandb.Image(plt)})
        wandb.log({"Generations": wandb.Image(plt, caption="Attributes:{}".format( attributes)) }) #

    
    def latent_space_interpolation(self, digit_label=1):
        n = 10 # number of images per row and column
        limit=3 # random values are sampled from the range [-limit,+limit]
        digit_label_one_hot = to_categorical(digit_label, self.category_count).reshape(1,-1)
        a = tf.convert_to_tensor(digit_label_one_hot)
        grid_x = np.linspace(-limit,limit, n) 
        grid_y = np.linspace(limit,-limit, n)

        generated_images=[]
        for i, yi in enumerate(grid_y):
            single_row_generated_images=[]
            for j, xi in enumerate(grid_x):
                random_sample = np.array([[xi, yi]])
                digit_label_one_hot = to_categorical(digit_label, self.category_count).reshape(1,-1)
                a = tf.convert_to_tensor(digit_label_one_hot)
                b = tf.concat([a, a], axis=0) # with 1 dimension, it fails...
              # z_cond = self.reparametrization(z_mean=random_sample, z_log_var=0.0, input_label = b) 
                z_cond = tf.concat([random_sample, b], axis=1) # changed, need testing
                decoded_x = self.model.decoder.predict(z_cond)
                single_row_generated_images.append(decoded_x[0].reshape(self.input_shape)) # changed, need testing
                generated_images.append(single_row_generated_images)      
        plot_generated_images(generated_images,n,n,True)
    
    def __call__(self):

        if (self.category_count <= 10):
            for i in range(self.category_count):
                self.generations_class(i)

        else:
            print('generations for celeba beta')
            self.generations_celeba(self, [0, 8, 15, 20])
            self.generations_celeba(self, [2, 9, 12, 21, 26, 27, 31, 39])



def plot_generated_images(generated_images, nrows, ncols, digit_label,
                          no_space_between_plots=False, figsize=(10, 10)):
  _, axs = plt.subplots(nrows, ncols,figsize=figsize,squeeze=False)

  for i in range(nrows):
    for j in range(ncols):
      axs[i,j].axis('off')
      axs[i,j].imshow(generated_images[i][j], cmap='gray')

  if no_space_between_plots:
    plt.subplots_adjust(wspace=0,hspace=0)
  
  #wandb.log({"Latent_interpolation_class: {}".format(digit_label): wandb.Image(plt)})
  wandb.log({"Latent_interpolation": wandb.Image(plt, caption="Class:{}".format( digit_label)) }) #

  plt.show()