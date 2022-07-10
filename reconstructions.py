#reconstructions
import random
import matplotlib.pyplot as plt
import wandb


def reconstructions(model, train_x, train_y, train_x_mean, train_log_var, input_label_train, labels):
    z_cond_train = model.sampling(train_x_mean, train_log_var, input_label_train)


    reconstruction_train = model.decoder(z_cond_train[:200])


    image_count = 10

    _, axs = plt.subplots(2, image_count, figsize=(20, 4))
    for i in range(image_count):
        random_idx = random.randint(0, reconstruction_train.shape[0]-1)
        axs[0, i].imshow(train_x[random_idx])
        axs[0, i].axis('off')

        axs[0, i].set_title( labels[int(train_y[random_idx])]  )
        axs[1, i].imshow(reconstruction_train[random_idx])
        axs[1, i].axis('off')
    wandb.log({"Reconstructions": wandb.Image(plt)})