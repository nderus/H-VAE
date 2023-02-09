import argparse
import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
from tensorflow.keras import regularizers
from src.models.VAE.CVAE import CVAE
from src.models.VAE.encoders import EncoderResNet18, EncoderResNet34, EncoderResNet50, encoderCNN, EncoderMixNet18
from src.models.VAE.decoders import DecoderResNet18, DecoderResNet34, DecoderResNet50, decoderCNN
from src.datasets import data_loader
from src.models.VAE.generations import Generations
from generation.vae_generate_utils import vae_generate_defaults
from generation.vae_generate_utils import add_dict_to_argparser

def main():
  
    args = create_argparser().parse_args()
    data = data_loader(name = args.dataset_name, root_folder='datasets/')

    if 'resnet' in args.model_name:
        encoder = EncoderMixNet18(encoded_dim = args.encoded_dim)
        encoder = encoder.model(input_shape=(data['input_shape'][0], data['input_shape'][1], data['input_shape'][2] + data['category_count']))
    else:
        encoder = encoderCNN(data['input_shape'], data['category_count'], args.encoded_dim,  regularizer=regularizers.L2(.001))

    if 'resnet' in args.model_name:
        # decoder = DecoderResNet18( encoded_dim = encoded_dim, final_stride = 2)
        # decoder = decoder.model(input_shape=(encoded_dim + category_count,))
        decoder = decoderCNN(data['input_shape'], data['category_count'], args.encoded_dim, final_stride = 1, regularizer=regularizers.L2(.001))
    else:
        decoder = decoderCNN(data['input_shape'], data['category_count'], args.encoded_dim, final_stride = 1, regularizer=regularizers.L2(.001))

    cvae = CVAE(encoder, decoder, args.kl_coefficient, data['input_shape'], data['category_count'])
    cvae.built = True

    opt = keras.optimizers.Adam(learning_rate = args.learning_rate)    
    cvae.compile(optimizer = opt, run_eagerly = False)

    generated_images = []

    while len(generated_images) * args.batch_size < args.num_samples:
        initial_noise = tf.random.normal( mean=0., stddev=1.0, shape=(args.num_samples, args.encoded_dim)) 
        condition = tf.convert_to_tensor(np.array([1, 0], dtype='float32'))
        condition = tf.reshape(condition, shape=(1,2))
        condition = tf.repeat(condition, repeats = [args.num_samples], axis=0)
        initial_noise = layers.Concatenate()([initial_noise, condition])
        generated_images.append(cvae.decoder(initial_noise) )

def create_argparser():
    defaults = dict(
        dataset_name = 'histo',
        model_name = 'CVAE_resnet_l2',
        encoded_dim = 1024 + 2048,
        learning_rate = 0.0001,
        batch_size = 32,
        num_samples = 100,
    )
    defaults.update(vae_generate_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
  main()

