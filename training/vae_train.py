"""
Train a VAE model on images.
"""
import argparse

from training.vae_train_utils import vae_defaults
from training.vae_train_utils import add_dict_to_argparser
from training.vae_train_utils import args_to_dict
from training.vae_train_utils import str2bool

from src.encoders import EncoderResNet18, EncoderResNet34, EncoderResNet50, encoderCNN, EncoderMixNet18
from src.decoders import DecoderResNet18, DecoderResNet34, DecoderResNet50, decoderCNN
from datasets import data_loader
from src.embeddings import embedding
from src.reconstructions import reconstructions
from src.generations import Generations
from activations import VisualizeActivations
from src.gradcam import GradCam
from src.CVAE import CVAE
from datasets import data_loader

def main():
  
  args = create_argparser().parse_args()
  #args to dict -> make a dict to feed into a function i.e. make_model()
  #args = args_to_dict(args, vae_defaults().keys())

  data = data_loader(name = args.dataset_name, root_folder='/content/')
  
  # (TO DO: function load model)
  early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',
             patience=patience, restore_best_weights=False)

  # (TO DO: function fit model)
  #NB: save_weights_only -> ValueError: Unable to create dataset (name a
  history = cvae.fit([train_x, train_y_one_hot],
                   validation_data = ([val_x, val_y_one_hot],None),
                   epochs = epoch_count,
                   batch_size = batch_size,
                   callbacks=[early_stop, WandbCallback(save_model = False) ]) 
  
  _, input_label_train, train_input = cvae.conditional_input([train_x[:1000], train_y_one_hot[:1000]])
  _, input_label_test, test_input = cvae.conditional_input([test_x[:1000], test_y_one_hot[:1000]])
  _, input_label_val, val_input = cvae.conditional_input([val_x[:1000], val_y_one_hot[:1000]])

  train_x_mean, train_log_var = cvae.encoder.predict(train_input)
  test_x_mean, test_log_var = cvae.encoder.predict(test_input)
  val_x_mean, val_log_var = cvae.encoder.predict(val_input)
  
  if args.embeddings:
    embedding(encoded_dim, category_count, train_x_mean, test_x_mean, val_x_mean, train_y, test_y, val_y, 
              train_log_var, test_log_var, val_log_var, labels, quantity = 1000, avg_latent=True)
    
  if args.reconstructions:
    reconstructions(cvae, train_x, train_y, train_x_mean, train_log_var, input_label_train, labels, set = 'train')
    reconstructions(cvae, train_x, train_y, train_x_mean, train_log_var, input_label_train, labels, set = 'test')
    
  if args.generations:
    generator = Generations(cvae, encoded_dim, category_count, input_shape, labels)
    generator()
   
  if args.gradcam:
    if 'resnet' in model_name:
      target_layer = "layer4"
    else:
      target_layer = "block3_conv2"
    gc = GradCam(cvae, test_x, test_y_one_hot, HQ = False, target_layer = target_layer)
    gc.gradcam()
    
  if args.gradcamHQ:
  if 'resnet' in model_name:
    target_layer = "layer4"
  else:
    target_layer = "block3_conv2"
  gc = GradCam(cvae, test_x, test_y_one_hot, HQ = True, target_layer = target_layer)
  gc.gradcam()
  
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
    )
    defaults.update(vae_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
  main()

