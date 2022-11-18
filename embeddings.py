import matplotlib.pyplot as plt
import wandb
import numpy as np
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

# TO DO: add log_var visualization maybe?

def embedding(encoded_dim, category_count, train_x_mean, test_x_mean, val_x_mean, train_y, test_y, val_y,
             train_log_var, test_log_var, val_log_var, labels, quantity = 5000, avg_latent=True):



    if encoded_dim == 2:
        plot_2d_data( [train_x_mean[:quantity], test_x_mean[:quantity], val_x_mean[:quantity]],
                    [train_y[:quantity], test_y[:quantity] ,val_y[:quantity]],
                    ['Train','Test', 'Validation'], figsize = (12,4))
        if category_count <= 10:
          plot_2d_data_categorical( [train_x_mean[:quantity], test_x_mean[:quantity], val_x_mean[:quantity]],
                      [train_y[:quantity], test_y[:quantity] ,val_y[:quantity]], labels,
                      ['Train','Test', 'Validation'], (12, 4 * category_count), category_count)

    else:
        from sklearn import manifold
        tsne = manifold.TSNE(n_components = 2, init='pca', random_state=0)
        train_x_tsne = tsne.fit_transform(train_x_mean[:quantity])
        test_x_tsne = tsne.fit_transform(test_x_mean[:quantity])
        val_x_tsne = tsne.fit_transform(val_x_mean[:quantity])
        plot_2d_data( [train_x_tsne, test_x_tsne, val_x_tsne],
                [train_y[:quantity], test_y[:quantity], val_y[:quantity]],
                ['Train','Test', 'Validation'], (12,4) )
        if category_count <= 10:
          plot_2d_data_categorical( [train_x_tsne, test_x_tsne, val_x_tsne],
                  [train_y[:quantity], test_y[:quantity] ,val_y[:quantity]], labels,
                  ['Train','Test', 'Validation'], (12, 4 * category_count), category_count)

    if avg_latent:
      import tensorflow as tf
      avg_variance_train = tf.reduce_mean(np.exp(train_log_var), axis=0)
      avg_variance_test = tf.reduce_mean(np.exp(test_log_var), axis=0)
      avg_variance_val = tf.reduce_mean(np.exp(val_log_var), axis=0)

      avg_mean_train = tf.reduce_mean(train_x_mean, axis=0)
      avg_mean_test = tf.reduce_mean(test_x_mean, axis=0)
      avg_mean_val = tf.reduce_mean(val_x_mean, axis=0)

      latent_variance = [avg_variance_train, avg_variance_test, avg_variance_val]
      latent_mean = [avg_mean_train, avg_mean_test, avg_mean_val]

      plot_latent_variables(latent_variance, latent_mean)
    
def plot_2d_data(data_2d, y, titles=None, figsize = (7, 7)):
  _, axs = plt.subplots(1, len(data_2d), figsize = figsize)

  for i in range(len(data_2d)):
    
    if (titles != None):
      axs[i].set_title(titles[i])

    scatter=axs[i].scatter(data_2d[i][:, 0], data_2d[i][:, 1],
                            s = 1,  c = plt.cm.tab10(y[i])) # removed c = y[i], cmap = plt.cm.tab10


    # axs[i].set_xlim([-xy_lim, xy_lim])
    # axs[i].set_ylim([-xy_lim, xy_lim])
    axs[i].legend(*scatter.legend_elements())
    
  wandb.log({"Embdedding": wandb.Image(plt)})




def plot_2d_data_categorical(data_2d, y, labels, titles=None, figsize = (7, 7), category_count = 10):
  _, axs = plt.subplots(category_count, len(data_2d), figsize = figsize )
  
  if category_count == 10:
    colors = np.array(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])
  
  elif cateogry_count == 2:
    colors = np.array(['#1f77b4', '#ff7f0e'])
  
  else:
    print('colors to be implemented')

  for i in range(len(data_2d)):
      for k in range(category_count):

        index = find_indices(y[i], lambda e: e == k)

        data_2d_k = data_2d[i][index, ]

        if (titles != None):
          axs[k,i].set_title("{} - Class: {}".format(titles[i], labels[k]))

        scatter = axs[k, i].scatter(data_2d_k[:, 0], data_2d_k[:, 1],
                                s = 1, c = colors[k], cmap = plt.cm.tab10)
        axs[k, i].legend(*scatter.legend_elements())
        # axs[k, i].set_xlim([-xy_lim, xy_lim])
        # axs[k, i].set_ylim([-xy_lim, xy_lim])
        
  wandb.log({"Embdedding_classes": wandb.Image(plt)})

def plot_latent_variables(latent_variance, latent_mean, titles = ['Train','Test', 'Validation'], figsize = (12,8)):
  _, axs = plt.subplots(2, len(latent_variance), figsize = (12,8))
    
  for i in range(len(latent_variance)):
      axs[0, i].set_title("{} - Average latent mean".format(titles[i]))
      axs[1, i].set_title("{} - Average latent variance".format(titles[i]))
      axs[0, i].hist(latent_mean[i], bins=25)
      axs[1, i].hist(latent_variance[i], bins=25)
      
  wandb.log({"Latent_variables": wandb.Image(plt)})

def find_indices(lst, condition):
    return np.array([i for i, elem in enumerate(lst) if condition(elem)])
    
