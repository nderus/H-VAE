"""
Train a VAE model on images.
"""
import os
import argparse
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras import regularizers
import wandb
from wandb.keras import WandbCallback

from training.vae_train_utils import vae_defaults
from training.vae_train_utils import add_dict_to_argparser
from training.vae_train_utils import args_to_dict
from training.vae_train_utils import str2bool

from src.models.VAE.encoders import EncoderResNet18, EncoderResNet34, EncoderResNet50, encoderCNN, EncoderMixNet18
from src.models.VAE.decoders import DecoderResNet18, DecoderResNet34, DecoderResNet50, decoderCNN
from src.datasets import data_loader
from src.models.VAE.embeddings import embedding
from src.models.VAE.reconstructions import reconstructions
from src.models.VAE.generations import Generations
from src.models.VAE.activations import VisualizeActivations
from src.models.VAE.gradcam import GradCam
from src.models.VAE.CVAE import CVAE


def main():
  
  args = create_argparser().parse_args()

  data = data_loader(name = args.dataset_name, root_folder='datasets/')

  # (TO DO: wandb wrapper)
  wandb.init(project = args.dataset_name, entity="nrderus",
    config = {
    "dataset": args.dataset_name,
    "model": args.model_name,
    "encoded_dim": args.encoded_dim,
    "kl_coefficient": args.kl_coefficient,
    "learning_rate": args.learning_rate,
    "epochs": args.epoch_count,
    "batch_size": args.batch_size,
    "patience": args.patience,
    })
  # (TO DO: function load model)
  #cvae = create_model(**args_to_dict(args, vae_defaults().keys()))
  
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
  cvae_input = cvae.encoder.input[0]
  cvae_output = cvae.decoder.output
  mu = cvae.encoder.get_layer('mu').output
  log_var = cvae.encoder.get_layer('log_var').output

  opt = keras.optimizers.Adam(learning_rate = args.learning_rate)    
  cvae.compile(optimizer = opt, run_eagerly = False)
  early_stop = keras.callbacks.EarlyStopping(monitor = 'val_loss',
             patience = args.patience, restore_best_weights = False)

  # (TO DO: function fit model)
  #NB: save_weights_only -> ValueError: Unable to create dataset (name a
  history = cvae.fit([data['train_x'], data['train_y_one_hot']],
                   validation_data = ([data['val_x'], data['val_y_one_hot']],None),
                   epochs = args.epoch_count,
                   batch_size = args.batch_size,
                   callbacks=[early_stop, WandbCallback(save_model = False) ]) 
  
  _, input_label_train, train_input = cvae.conditional_input([data['train_x'][:1000], data['train_y_one_hot'][:1000]])
  _, input_label_test, test_input = cvae.conditional_input([data['test_x'][:1000], data['test_y_one_hot'][:1000]])
  _, input_label_val, val_input = cvae.conditional_input([data['val_x'][:1000], data['val_y_one_hot'][:1000]])

  train_x_mean, train_log_var = cvae.encoder.predict(train_input)
  test_x_mean, test_log_var = cvae.encoder.predict(test_input)
  val_x_mean, val_log_var = cvae.encoder.predict(val_input)
  
  if args.embeddings:
    embedding(args.encoded_dim, data['category_count'], train_x_mean, test_x_mean, val_x_mean, data['train_y'], data['test_y'], data['val_y'], 
              train_log_var, test_log_var, val_log_var, data['labels'], quantity = 1000, avg_latent=True)
    
  if args.reconstructions:
    reconstructions(cvae, data['train_x'], data['train_y'], train_x_mean, train_log_var, input_label_train, data['labels'], set = 'train')
    reconstructions(cvae, data['train_x'], data['train_y'], train_x_mean, train_log_var, input_label_train, data['labels'], set = 'test')

  if args.activations:
    activations_encoder = VisualizeActivations(cvae, cvae.encoder, data['test_x'], data['test_y_one_hot'])
    activations_encoder()
    activations_decoder = VisualizeActivations(cvae, cvae.decoder, data['test_x'], data['test_y_one_hot'])
    activations_decoder()

  if args.generations:
    generator = Generations(cvae, args.encoded_dim, data['category_count'], data['input_shape'], data['labels'])
    generator()
   
  if args.gradcam:
    if 'resnet' in args.model_name:
      target_layer = "layer4"
    else:
      target_layer = "block3_conv2"
    gc = GradCam(cvae, data['test_x'], data['test_y_one_hot'], HQ = False, target_layer = target_layer)
    gc.gradcam()
    
  if args.gradcamHQ:
    if 'resnet' in args.model_name:
      target_layer = "layer4"
    else:
      target_layer = "block3_conv2"
    gc = GradCam(cvae, data['test_x'], data['test_y_one_hot'], HQ = True, target_layer = target_layer)
    gc.gradcam()

  wandb.finish(exit_code=0, quiet = True)

  if args.ddpm_refiner:
    #cvae.encoder.load_weights('checkpoints/VAE/encoder_weights2.h5')
    #cvae.decoder.load_weights('checkpoints/VAE/decoder_weights2.h5')
    from training.ddpm_train import main as ddpm
    ddpm(cvae, args.encoded_dim)



def create_argparser():
    defaults = dict(
        dataset_name = 'histo',
        model_name = 'CVAE_resnet_l2',
        kl_coefficient = .03,
        encoded_dim = 1024 + 2048,
        learning_rate = 0.0001,
        epoch_count = 100,
        batch_size = 100,
        patience = 10,
        wandb_logging = False,
        embeddings = False,
        reconstructions = False,
        generations = False,
        activations = False,
        gradcam = False,
        gradcamHQ = False,
        ddpm_refiner = False,
    )
    defaults.update(vae_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
  main()

