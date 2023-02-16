import argparse
import wandb

def vae_defaults():
    """
    Defaults for VAE models.
    """
    return dict(
        dataset_name = 'histo',
        model_name = 'CVAE_resnet_l2',
        kl_coefficient = .03,
        encoded_dim = 1024 + 2048,
        learning_rate = 0.0001,
        epoch_count = 1,
        batch_size = 100,
        patience = 1,
        wandb_logging = False,
        embeddings = False,
        reconstructions = False,
        generations = False,
        activations = False,
        gradcam = False,
        gradcamHQ = False,
        ddpm_refiner = False,
        
    )

def vae_sweep():
    """
    Input config options for wandb sweep.
    """
    # sweep_configuration = {
    #     'method': 'random',
    #     'name': 'sweep',
    #     'metric': {'goal': 'minimize', 'name': 'val_loss'},
    #     'parameters': 
    #     {   
    #         'kl_coefficient' : {"values": [0, 0.0005, 0.05, 1]},
    #         'encoded_dim' : {"values": [1024, 2048, 1024 + 2048]},
    #         'learning_rate' : {"values": [0.0001]},
    #         'epoch_count' : {"values": [1]},
    #         'batch_size' :  {"values": [100]},
    #         'patience' :  {"values": [1]},
    #     }

    # }
    wandb.init(entity='nrderus', project='VAE-sweep')
    #wandb.init()
    #sweep_id = wandb.sweep(sweep=sweep_configuration, entity='nrderus', project='VAE-sweep')
    #import os
    #os.environ["SWEEP_ID"] = sweep_id
    print(wandb.config)

    return dict(
        kl_coefficient = wandb.config.kl_coefficient,
        encoded_dim = wandb.config.encoded_dim,
        learning_rate = wandb.config.learning_rate,
        epoch_count = wandb.config.epoch_count,
        batch_size = wandb.config.batch_size,
        patience = wandb.config.patience,
    )

def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)

def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}

def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
