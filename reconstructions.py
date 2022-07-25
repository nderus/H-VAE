#reconstructions
import random
import matplotlib.pyplot as plt
import wandb
import numpy as np

def reconstructions(model, train_x, train_y, train_x_mean, train_log_var, input_label_train, labels, random_reconstructions=False, set ='train'):


    z_cond_train = model.sampling(train_x_mean, train_log_var, input_label_train)

    reconstruction_train = model.decoder(z_cond_train[:20])

    if reconstruction_train.shape[3] == 1:
        reconstruction_train = reconstruction_train.squeeze()

    image_count = 10

    if random_reconstructions:

        _, axs = plt.subplots(2, image_count, figsize=(20, 4))
        for i in range(image_count):
            random_idx = random.randint(0, reconstruction_train.shape[0]-1)
            fixed_idx = range(0, image_count)
            axs[0, i].imshow(train_x[random_idx])
            axs[0, i].axis('off')
            if len(labels) <= 10:
                axs[0, i].set_title( labels[int(train_y[random_idx])]  )
            axs[1, i].imshow(reconstruction_train[random_idx])
            axs[1, i].axis('off')
        wandb.log({"Reconstructions": wandb.Image(plt)})
    
    else:

        _, axs = plt.subplots(2, image_count, figsize=(20, 4))
        for i in range(image_count):
            axs[0, i].imshow(train_x[i])
            axs[0, i].axis('off')
            if len(labels) <= 10:
                axs[0, i].set_title( labels[int(train_y[i])]  )
            axs[1, i].imshow(reconstruction_train[i])
            axs[1, i].axis('off')

    #wandb.log({"Reconstructions_{}".format(set): wandb.Image(plt)})
    wandb.log({"Reconstructions": wandb.Image(plt, caption="Set:{}".format(set)) }) #


####

