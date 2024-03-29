"""
Train a DDPM model on images as refiner to VAEs.
"""
import argparse
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from src.models.DDPM.DDPM import DiffusionModel
from training.ddpm_train_utils import ddpm_defaults
from training.ddpm_train_utils import add_dict_to_argparser
from training.ddpm_train_utils import preprocess_image
from training.ddpm_train_utils import ddpm_sweep
from wandb.keras import WandbCallback

def main(cvae, cvae_encoded_dim):
  
    args = create_argparser().parse_args()
    builder = tfds.builder_from_directory('datasets/=/histo/1.0.0')
    # Construct the tf.data.Dataset pipeline
    train_ds, val_ds = builder.as_dataset(split = ["cancer[:60000]",
                                                  "cancer[60000:70000]"],
                                                  as_supervised=False,
                                                  shuffle_files=False)
    
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
                           kid_image_size= args.kid_image_size,
                           embedding_dims= args.embedding_dims,
                           )
  
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=args.learning_rate,
        ),
        loss=keras.losses.mean_absolute_error,
    )

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath = args.checkpoint_path,
        save_weights_only = True,
        monitor = "val_kid",
        mode = "min",
        save_best_only = True,
    )

    # calculate mean and variance of training dataset for normalization
    model.normalizer.adapt(train_ds)

    if args.train_from_checkpoint:
        model.load_weights(args.checkpoint_path)

    # run training and plot generated images periodically
    model.fit(
        train_ds,
        epochs = args.num_epochs,
        validation_data = val_ds,
        callbacks=[
            keras.callbacks.LambdaCallback(on_epoch_end = model.plot_images),
            checkpoint_callback,
            WandbCallback(save_model = False),
        ],
    )

    model.load_weights(args.checkpoint_path)
    model.plot_images(wandb_log = True)

def create_argparser():
    defaults = dict(
        dataset_name = 'histo',
        dataset_repetitions = 15,
        num_epochs = 100, 
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
        checkpoint_path = "/checkpoints/diffusion_model",
        train_from_checkpoint = False,  
    )
    defaults.update(ddpm_defaults())
    defaults.update(ddpm_sweep())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
  main()

