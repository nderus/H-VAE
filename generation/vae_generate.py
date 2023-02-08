import argparse
from tensorflow import keras
from tensorflow.keras import regularizers
from src.models.VAE.CVAE import CVAE
from src.models.VAE.encoders import EncoderResNet18, EncoderResNet34, EncoderResNet50, encoderCNN, EncoderMixNet18
from src.models.VAE.decoders import DecoderResNet18, DecoderResNet34, DecoderResNet50, decoderCNN
from src.datasets import data_loader
from src.models.VAE.generations import Generations
from training.vae_train_utils import vae_defaults
from training.vae_train_utils import add_dict_to_argparser

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

  generator = Generations(cvae, args.encoded_dim, data['category_count'], data['input_shape'], args.labels)
  generator()

def create_argparser():
    defaults = dict(
        dataset_name = 'histo',
        model_name = 'CVAE_resnet_l2',
        encoded_dim = 1024 + 2048,
        learning_rate = 0.0001,
        wandb_logging = False,
    )
    defaults.update(vae_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
  main()

