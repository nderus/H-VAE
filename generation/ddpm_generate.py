"""
Generate synthetic datasets from the VAE backbone + DDPM refiner.
"""
import argparse
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
import numpy as np
from src.models.DDPM.DDPM import DiffusionModel
from generation.ddpm_generate_utils import ddpm_defaults
from generation.ddpm_generate_utils import add_dict_to_argparser
from generation.ddpm_generate_utils import preprocess_image


def main(cvae, cvae_encoded_dim):
  
    args = create_argparser().parse_args()
    builder = tfds.builder_from_directory('datasets/=/histo/1.0.0')
    # Construct the tf.data.Dataset pipeline
    train_ds, val_ds = builder.as_dataset(split = ["cancer[:60000]",
                                                  "cancer[60000:70000]"],
                                                  as_supervised=False, shuffle_files=False)
    
    train_ds = train_ds.map(preprocess_image, num_parallel_calls = tf.data.AUTOTUNE).cache()
    train_ds = train_ds.repeat(args.dataset_repetitions).shuffle(10 * args.batch_size)
    train_ds = train_ds.batch(args.batch_size, drop_remainder = True).prefetch(buffer_size = tf.data.AUTOTUNE)
    val_ds = val_ds.map(preprocess_image, num_parallel_calls = tf.data.AUTOTUNE).cache()
    val_ds = val_ds.repeat(args.dataset_repetitions).shuffle(10 * args.batch_size)
    val_ds = val_ds.batch(args.batch_size, drop_remainder = True).prefetch(buffer_size = tf.data.AUTOTUNE)
    # create and compile the model
    model = DiffusionModel(image_size = args.image_size,
                           widths = args.widths, 
                           block_depth = args.block_depth, 
                           batch_size = args.batch_size,
                           min_signal_rate = args.min_signal_rate, 
                           max_signal_rate = args.max_signal_rate, 
                           cvae = cvae,
                           kid_diffusion_steps = args.kid_diffusion_steps, 
                           plot_diffusion_steps = args.plot_diffusion_steps,
                           encoded_dim = cvae_encoded_dim,
                           ema = args.ema, 
                           kid_image_size= args.kid_image_size
                           )
  
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=args.learning_rate,
        ),
        loss=keras.losses.mean_absolute_error,
    )

    # calculate mean and variance of training dataset for normalization
    model.normalizer.adapt(train_ds)
    model.load_weights(args.checkpoint_path)

    batches = args.num_samples // args.batch_size

    result = [model.generate(args.batch_size, args.plot_diffusion_steps) for _ in range(batches)]
    result.append(model.generate(args.num_samples % args.batch_size, args.plot_diffusion_steps))

    print('Generated {} images in {} batches of {} + a minibatch of {}'.format(args.num_samples, 
                                                                               batches,
                                                                               args.batch_size,
                                                                               args.num_samples % args.batch_size))

    result = np.array(result, dtype=object)
    np.save('datasets/synthetic/ddpm_synthetic_dataset.npy', result)

def create_argparser():
    defaults = dict(
        dataset_name = 'histo',
        dataset_repetitions = 15,
        image_size = 48,
        kid_image_size = 75,
        kid_diffusion_steps = 5,
        plot_diffusion_steps = 1000,
        min_signal_rate = 0.15,
        max_signal_rate = 0.95,
        embedding_dims = 512,
        embedding_max_frequency = 1000.0,
        widths = [32, 64, 96, 128, 256],
        block_depth = 4,
        batch_size = 64,
        ema = 0.999,
        learning_rate = 1e-3,
        weight_decay = 1e-4,     
        checkpoint_path = "checkpoints/diffusion_model",
        num_samples = 100,
    )
    defaults.update(ddpm_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
  main()