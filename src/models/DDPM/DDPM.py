import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
from training.ddpm_train_utils import KID
import wandb

def ResidualBlock(width):
    def apply(x):
        input_width = x.shape[3]
        if input_width == width:
            residual = x
        else:
            residual = layers.Conv2D(width, kernel_size=1)(x)
        x = layers.BatchNormalization(center=False, scale=False)(x)
        x = layers.Conv2D(
            width, kernel_size=3, padding="same", activation=keras.activations.swish
        )(x)
        x = layers.Conv2D(width, kernel_size=3, padding="same")(x)
        x = layers.Add()([x, residual])
        return x

    return apply

def DownBlock(width, block_depth):
    def apply(x):
        x, skips = x
        for _ in range(block_depth):
            x = ResidualBlock(width)(x)
            skips.append(x)
        x = layers.AveragePooling2D(pool_size=2)(x)
        return x

    return apply

def UpBlock(width, block_depth):
    def apply(x):
        x, skips = x
        x = layers.UpSampling2D(size=2, interpolation="bilinear")(x)
        for _ in range(block_depth):
            x = layers.Concatenate()([x, skips.pop()])
            x = ResidualBlock(width)(x)
        return x

    return apply

def sinusoidal_embedding(x, embedding_max_frequency = 1000.0, embedding_dims=128): # was 512
        embedding_min_frequency = 1.0
        frequencies = tf.exp(
            tf.linspace(
                tf.math.log(embedding_min_frequency),
                tf.math.log(embedding_max_frequency),
                embedding_dims // 2,
            )
        )
        angular_speeds = 2.0 * math.pi * frequencies
        embeddings = tf.concat(
            [tf.sin(angular_speeds * x), tf.cos(angular_speeds * x)], axis=3 )
        return embeddings
 
def get_network(image_size, widths, block_depth, embedding_dims):
    noisy_images = keras.Input(shape=(image_size, image_size, 3))
    noise_variances = keras.Input(shape=(1, 1, 1))

   # e = layers.Lambda(sinusoidal_embedding)(noise_variances)
    e = layers.Lambda(lambda x: sinusoidal_embedding(x, embedding_max_frequency=1000.0, embedding_dims=embedding_dims))(noise_variances)
    e = layers.UpSampling2D(size=image_size, interpolation="nearest")(e)

    x = layers.Conv2D(widths[0], kernel_size=1)(noisy_images)
    x = layers.Concatenate()([x, e])

    skips = []
    for width in widths[:-1]:
        x = DownBlock(width, block_depth)([x, skips])

    for _ in range(block_depth):
        x = ResidualBlock(widths[-1])(x)

    for width in reversed(widths[:-1]):
        x = UpBlock(width, block_depth)([x, skips])

    layer = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=128, attention_axes=(2, 3)) #added, was , num_heads=8, key_dim=16
    output_tensor = layer(x, x) #added
    x = layers.Conv2D(3, kernel_size=1, kernel_initializer="zeros")(output_tensor)


    return keras.Model([noisy_images, noise_variances], x, name="residual_unet") #changed output

class DiffusionModel(keras.Model):
    def __init__(self, image_size, widths, block_depth, batch_size, min_signal_rate, max_signal_rate,
                 cvae, kid_diffusion_steps, plot_diffusion_steps, ema, encoded_dim, kid_image_size, embedding_dims):
        super().__init__()
        self.image_size = image_size
        self.batch_size = batch_size
        self.min_signal_rate = min_signal_rate
        self.max_signal_rate = max_signal_rate
        self.cvae = cvae
        self.kid_diffusion_steps = kid_diffusion_steps
        self.plot_diffusion_steps = plot_diffusion_steps
        self.ema = ema
        self.encoded_dim = encoded_dim
        self.kid_image_size = kid_image_size
        self.embedding_dims = embedding_dims
        self.normalizer = layers.Normalization()
        self.network = get_network(image_size, widths, block_depth, self.embedding_dims)
        self.ema_network = keras.models.clone_model(self.network)

    def compile(self, **kwargs):
        super().compile(**kwargs)
        self.noise_loss_tracker = keras.metrics.Mean(name="n_loss")
        self.image_loss_tracker = keras.metrics.Mean(name="i_loss")
        self.kid = KID(name="kid", image_size=self.image_size, kid_image_size=self.kid_image_size)

    @property
    def metrics(self):
        return [self.noise_loss_tracker, self.image_loss_tracker, self.kid]

    def denormalize(self, images):
        # convert the pixel values back to 0-1 range
        images = self.normalizer.mean + images * self.normalizer.variance**0.5
        return tf.clip_by_value(images, 0.0, 1.0)

    def diffusion_schedule(self, diffusion_times):
        # diffusion times -> angles
        start_angle = tf.acos(self.max_signal_rate)
        end_angle = tf.acos(self.min_signal_rate)

        diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

        # angles -> signal and noise rates
        signal_rates = tf.cos(diffusion_angles)
        noise_rates = tf.sin(diffusion_angles)
        # note that their squared sum is always: sin^2(x) + cos^2(x) = 1

        return noise_rates, signal_rates

    def denoise(self, noisy_images, noise_rates, signal_rates, reconstructions, training):
        # the exponential moving average weights are used at evaluation
        if training:
            network = self.network
        else:
            network = self.ema_network

        # predict noise component and calculate the image component using it
        pred_noises = network([noisy_images, noise_rates**2], training=training)
        #pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates
        pred_images = (noisy_images - reconstructions - pred_noises * noise_rates) / signal_rates #x_0_hat
        
        return pred_noises, pred_images

    def reverse_diffusion(self, initial_noise, diffusion_steps, reconstructions):
        # reverse diffusion = sampling
        num_images = initial_noise.shape[0]
        step_size = 1.0 / diffusion_steps

        # important line:
        # at the first sampling step, the "noisy image" is pure noise
        # but its signal rate is assumed to be nonzero (min_signal_rate)
        next_noisy_images = initial_noise
        for step in range(diffusion_steps):
            noisy_images = next_noisy_images

            # separate the current noisy image to its components
            diffusion_times = tf.ones((num_images, 1, 1, 1)) - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates, reconstructions, training=False
            )
            # network used in eval mode

            # remix the predicted components using the next signal and noise rates
            next_diffusion_times = diffusion_times - step_size

            next_noise_rates, next_signal_rates = self.diffusion_schedule(
                next_diffusion_times
            )
            coeff = 1 - next_noise_rates / noise_rates

            next_noisy_images = (
               next_signal_rates * pred_images + next_noise_rates * (pred_noises + reconstructions / noise_rates ) + coeff * reconstructions
                
            )

        return pred_images


    def train_step(self, images):

        if isinstance(images, tuple):
            images = images[0]

        # normalize images to have standard deviation of 1, like the noises
        images = self.normalizer(images, training=True)
        noises = tf.random.normal(shape=(self.batch_size, self.image_size, self.image_size, 3))
        condition = tf.convert_to_tensor(np.array([1, 0], dtype='float32'))
        condition = tf.reshape(condition, shape=(1,2))
        condition = tf.repeat(condition, repeats = [self.batch_size], axis=0)
        reconstructions = self.cvae([images, condition])

        # sample uniform random diffusion times
        diffusion_times = tf.random.uniform(
            shape=(self.batch_size, 1, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        # mix the images with noises accordingly
        noisy_images = signal_rates * images + noise_rates * noises + reconstructions

        with tf.GradientTape() as tape:
            # train the network to separate noisy images to their components
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates, reconstructions, training=True
            )

            noise_loss = self.loss(noises, pred_noises)  # used for training
            image_loss = self.loss(images, pred_images)  # only used as metric
        #pred_images = pred_images - reconstructions # added, at t=0, remove cond 
        gradients = tape.gradient(noise_loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        self.noise_loss_tracker.update_state(noise_loss)
        self.image_loss_tracker.update_state(image_loss)

        # track the exponential moving averages of weights
        for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)

        # KID is not measured during the training phase for computational efficiency
        return {m.name: m.result() for m in self.metrics[:-1]}

    def test_step(self, images):
        if isinstance(images, tuple):
          images = images[0]
        # normalize images to have standard deviation of 1, like the noises
        images = self.normalizer(images, training=False)
        noises = tf.random.normal(shape=(self.batch_size, self.image_size, self.image_size, 3))
        condition = tf.convert_to_tensor(np.array([1, 0], dtype='float32'))
        condition = tf.reshape(condition, shape=(1,2))
        condition = tf.repeat(condition, repeats = [self.batch_size], axis=0)
        reconstructions = self.cvae([images, condition])
      
        # sample uniform random diffusion times
        diffusion_times = tf.random.uniform(
            shape=(self.batch_size, 1, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        # mix the images with noises accordingly
        noisy_images = signal_rates * images + noise_rates * noises + reconstructions

        # use the network to separate noisy images to their components
        pred_noises, pred_images = self.denoise(
            noisy_images, noise_rates, signal_rates, reconstructions, training=False
        )

        noise_loss = self.loss(noises, pred_noises)
        image_loss = self.loss(images, pred_images)

        self.image_loss_tracker.update_state(image_loss)
        self.noise_loss_tracker.update_state(noise_loss)

        #pred_images = pred_images - reconstructions # added, at t=0, remove cond 

        # measure KID between real and generated images
        # this is computationally demanding, kid_diffusion_steps has to be small
        images = self.denormalize(images)

        generated_images = self.generate(
            num_images=self.batch_size, diffusion_steps=self.kid_diffusion_steps
        )
        self.kid.update_state(images, generated_images)

        return {m.name: m.result() for m in self.metrics}

    def plot_images(self, epoch=None, logs=None, num_rows=1, num_cols=6, save = True, wandb_log = False):
        # plot random generated images for visual evaluation of generation quality
        generated_images = self.generate(
            num_images=num_rows * num_cols,
            diffusion_steps=self.plot_diffusion_steps,
        )
        plt.figure(figsize=(num_cols * 2.0, num_rows * 2.0))
        for row in range(num_rows):
            for col in range(num_cols):
                index = row * num_cols + col
                plt.subplot(num_rows, num_cols, index + 1)
                plt.imshow(generated_images[index])
                plt.axis("off")
        plt.tight_layout()
        if save:
          plt.savefig('/content/drive/MyDrive/plot_images/plot_images_diffusion_conditional.png')
        if wandb_log:
          wandb.log({"Images": wandb.Image(plt, caption="Diffusion images") })

        plt.show()
        plt.close()

    def generate(self, num_images, diffusion_steps):
        # noise -> images -> denormalized images 
        initial_noise =  tf.random.normal(  mean=0., stddev=1.0, shape=(num_images, self.encoded_dim)) 
        condition = tf.convert_to_tensor(np.array([1, 0], dtype='float32'))
        condition = tf.reshape(condition, shape=(1,2))
        condition = tf.repeat(condition, repeats = [num_images], axis=0)
        initial_noise = layers.Concatenate()([initial_noise, condition])

        y = self.cvae.decoder(initial_noise)
        #y = (y - tf.experimental.numpy.mean(y)) / tf.experimental.numpy.std(y)
        x_t = tf.random.normal(mean = y, shape=(num_images, self.image_size, self.image_size, 3))
  
        generated_images = self.reverse_diffusion(x_t, diffusion_steps, y)
        generated_images = self.denormalize(generated_images)
        return generated_images