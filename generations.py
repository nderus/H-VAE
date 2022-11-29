# generations
import tensorflow as tf

from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
import wandb

class Generations:
    def __init__(self, model, encoded_dim, category_count, input_shape, labels, model2 = None, second_stage = False):
        self.model = model
        self.encoded_dim = encoded_dim
        self.category_count = category_count
        self.input_shape = input_shape
        self.labels = labels
        self.model2 = model2
        self.second_stage = second_stage

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
            if self.second_stage:
                z_cond = self.model2.posterior(z_cond, b)
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
        if self.second_stage:
            wandb.log({"Generations 2nd stage": wandb.Image(plt, caption="Class:{}_{}".format(digit_label, self.labels[digit_label])) })
        else:
            wandb.log({"Generations": wandb.Image(plt, caption="Class:{}_{}".format(digit_label, self.labels[digit_label])) }) #
            

    def generations_celeba(self, target_attr, batch_size = 100):
        image_count = 10

        _, axs = plt.subplots(2, image_count, figsize=(20, 4))
        #for j in range(2):
        for i in range(image_count):

            attr_vect = np.zeros(40)
            for attr in target_attr:
                attr_vect[attr] = 1

            labels = np.tile(attr_vect, reps = [batch_size, 1])
            
            a = tf.convert_to_tensor(labels,dtype="float")
            b = tf.concat([a, a], axis=0) # with 1 dimension, it fails...
            z_cond = self.reparametrization(z_mean=0, z_log_var=0.3, input_label = b)
            if self.second_stage:
                z_cond = self.model2.posterior(z_cond, b)
            decoded_x = self.model.decoder.predict(z_cond)
            digit_0 = decoded_x[0].reshape(self.input_shape) 
            digit_1 = decoded_x[1].reshape(self.input_shape) 
            axs[0, i].imshow(digit_0)
            axs[0, i].axis('off')
            axs[1, i].imshow(digit_1)
            axs[1, i].axis('off')

        attributes = str(self.labels[target_attr].tolist())
        if self.second_stage:
            wandb.log({"Generations 2nd stage": wandb.Image(plt, caption="Attributes:{}".format( attributes)) }) #
        else:
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
            self.generations_celeba( [0, 8, 20])
            self.generations_celeba([2, 9, 12, 21, 26, 27, 31, 39])



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

class Generations_filters:
    
    def __init__(self, model, encoded_dim, category_count, input_shape, labels, model2 = None, second_stage = False):
        self.model = model
        self.encoded_dim = encoded_dim
        self.category_count = category_count
        self.input_shape = input_shape
        self.labels = labels
        self.model2 = model2
        
    
    def sampling(self, z_mean, z_log_var, input_label):
        if len(input_label.shape) == 1:
            input_label = np.expand_dims(input_label, axis=0)
        eps = tf.random.normal(tf.shape(z_log_var), dtype=tf.float32,
                               mean=0., stddev=1.0, name='epsilon')
        z = z_mean + tf.exp(z_log_var / 2) * eps
        image_size = [tf.shape(z_log_var)[1], tf.shape(z_log_var)[2], tf.shape(z_log_var)[3]]
        labels = tf.reshape(input_label, [-1, 1, 1, self.category_count])
        labels = tf.cast(labels, dtype='float32')
        ones = tf.ones([tf.shape(z_log_var)[0]] + image_size[0:-1] + [self.category_count])
        input_label = ones * labels
        z_cond = tf.concat([z, input_label], axis=3)
        return z_cond
    
    def generations_class(self, digit_label=1, image_count = 10):
        _, axs = plt.subplots(2, image_count, figsize=(20, 4))
        for i in range(image_count):
            digit_label_one_hot = to_categorical(digit_label, self.category_count).reshape(1,-1)
            a = tf.convert_to_tensor(digit_label_one_hot)
            b = tf.concat([a, a], axis=0) # with 1 dimension, it fails...
            z_cond = self.sampling(z_mean=0, z_log_var=0, input_label = b) # TO DO: sub this with the sampling CVAE function
            if self.second_stage:
                z_cond = self.model2.posterior(z_cond, b)
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
                
    def __call__(self):

        if (self.category_count <= 10):
            for i in range(self.category_count):
                self.generations_class(i)

        else:
            print('generations for celeba beta')
            self.generations_celeba( [0, 8, 20])
            self.generations_celeba([2, 9, 12, 21, 26, 27, 31, 39])

