from json import encoder
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import models
import numpy as np
import math
import wandb

class VisualizeActivations():
    def __init__(self, main_model, encoder, decoder, test_x, test_y_one_hot):
        self.main_model = main_model
        self.encoder = encoder
        self.decoder = decoder
        self.test_x = test_x
        self.test_y_one_hot = test_y_one_hot

    def visualize_activations(self, main_model, model):
        test = self.test_x[0]
        plt.imshow(test)
        test = np.expand_dims(test, axis=0)
        test.shape
        test_label = self.test_y_one_hot[0]
        img_tensor = [test, test_label]

        # Extracts the outputs of the top 8 layers:
        layer_outputs = []
        layer_names = []
        for layer in model.layers[1:]:
            
            try: 
                layer_outputs.append(layer.get_output_at(0)) #N: this depends on the architecture, resnet = 1 , CNN = 0
                layer_names.append(layer.name)
            
            except:
                layer_outputs.append(layer.output)
                layer_names.append(layer.name)

        # Creates a model that will return these outputs, given the model input:
        activation_model = models.Model(inputs= model.input, outputs=layer_outputs)
        
        # This will return a list of 5 Numpy arrays:
        # one array per layer activation
        if 'encoder' in model.name:
            _, input_label, conditional_input = main_model.conditional_input(img_tensor) # TO DO: input the cvae as parameter and make it clear is different from encoder,decoder
            activations = activation_model.predict(conditional_input) #for encoder

        if 'decoder' in model.name:
            _, input_label, conditional_input = main_model.conditional_input(img_tensor)
            input_label = np.expand_dims(input_label, axis=0)
            z_mean, z_log_var = main_model.encoder(conditional_input)
            z_cond = main_model.sampling(z_mean, z_log_var, input_label)
            
            activations = activation_model.predict(z_cond) #for decoder
        
        for activation, name in zip(activations[0:], layer_names[0:]):
            print(name)
            print(activation.shape)
        
        for _, (activation, name) in enumerate(zip(activations[0:], layer_names[0:])):
            print(name)
            print(model.name)
            self.plot_filters(activation, name, model_name = model.name)

    def plot_filters(self, activation_layer, layer_name, model_name):

        if len(activation_layer.shape) == 2: # if flat layer
            print('flat')
            return None
        n = math.floor(np.sqrt(activation_layer.shape[3]))

        if int(n + 0.5) ** 2 == activation_layer.shape[3]:
            m = n
        else:
            m = math.floor(activation_layer.shape[3] / n)

        if activation_layer.shape[3] == 1:
            fig, ax = plt.subplots(1, 1, sharex='col', sharey='row',
                                    figsize=(15, 15))
            fig.suptitle(layer_name)

            ax.imshow(activation_layer[0,:, :, 0], cmap='viridis')
            wandb.log({"Activations": wandb.Image(plt, caption="{}_{}".format(model_name, layer_name)) })
    
            return None   

        if n == 1:
            fig, ax = plt.subplots(1, 3, sharex='col', sharey='row',figsize=(15, 15))
            fig.suptitle(layer_name)
            for i in range(3):
                ax[i].imshow(activation_layer[0,:, :, i], cmap='viridis')
            wandb.log({"Activations": wandb.Image(plt, caption="{}_{}".format(model_name, layer_name)) })
            return None   

        fig, ax = plt.subplots(n, m, sharex='col', sharey='row',figsize=(15, 15))
        fig.suptitle(layer_name)

        filter_counter = 0
        for i in range(n):
            for j in range(m):
                ax[i, j].imshow(activation_layer[0,:, :, filter_counter], cmap='viridis')
                filter_counter += 1
                if filter_counter == (activation_layer.shape[3] ):
                    break

        wandb.log({"Activations": wandb.Image(plt, caption="{}_{}".format(model_name, layer_name)) })
        return None

    def __call__(self):
        self.visualize_activations(self.main_model, self.encoder)
        self.visualize_activations(self.main_model, self.decoder)