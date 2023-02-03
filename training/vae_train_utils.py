import argparse

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
        epoch_count = 100,
        batch_size = 100,
        patience = 10,
        wandb_logging = False,
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