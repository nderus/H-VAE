"""
Train a VAE model on images.
"""
import argparse

from training.vae_train_utils import vae_defaults
from training.vae_train_utils import add_dict_to_argparser
from training.vae_train_utils import args_to_dict
from training.vae_train_utils import str2bool
from datasets import data_loader

def main():
  
  args = create_argparser().parse_args()
  #args = args_to_dict(args, vae_defaults().keys())

  data = data_loader(name = args.dataset_name, root_folder='/content/')
  
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
    )
    defaults.update(vae_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
  main()

