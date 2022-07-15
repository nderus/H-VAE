#https://github.com/ismailuddin/gradcam-tensorflow-2/blob/master/notebooks/GradCam.ipynb
# GRAD CAM

# TO DO:
# 1. generalize for resnets
# 2. repeat for more images?
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import numpy as np
import wandb

class GradCam:
    def __init__(self, cvae, test_x, test_y_one_hot, HQ, target_layer):
        self.cvae = cvae
        self.model = cvae.encoder
        self.image = test_x[0]
        self.test_y_one_hot = test_y_one_hot[0]
        self.HQ = HQ
        self.target_layer = target_layer

    def gradcam(self):

        last_conv_layer = self.model.get_layer(self.target_layer) #last conv layer
        last_conv_layer_model = tf.keras.Model(self.model.inputs, last_conv_layer.output)

        classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
        x = classifier_input
        names = self.find_names_after(self.target_layer)
        #['batch_normalization_2', 'leaky_re_lu_2', 'flatten', 'dense', 'mu']
        for layer_name in names: # only for CNN rn, need to normalize names
            x = self.model.get_layer(layer_name)(x)
        classifier_model = tf.keras.Model(classifier_input, x)

        with tf.GradientTape() as tape:
            inputs = self.image[np.newaxis, ...]
            _, _, z_cond = self.cvae.conditional_input([inputs, self.test_y_one_hot])
            last_conv_layer_output = last_conv_layer_model(z_cond)
            tape.watch(last_conv_layer_output)
            preds = classifier_model(last_conv_layer_output)
            top_pred_index = tf.argmax(preds[0])
            top_class_channel = preds[:, top_pred_index]
        
        grads = tape.gradient(top_class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        last_conv_layer_output = last_conv_layer_output.numpy()[0]
        pooled_grads = pooled_grads.numpy()
        for i in range(pooled_grads.shape[-1]):
            last_conv_layer_output[:, :, i] *= pooled_grads[i]

        # Average over all the filters to get a single 2D array
        gradcam = np.mean(last_conv_layer_output, axis=-1)
        # Clip the values (equivalent to applying ReLU)
        # and then normalise the values
        gradcam = np.clip(gradcam, 0, np.max(gradcam)) / np.max(gradcam)
        gradcam = cv2.resize(gradcam, (64, 64))
        wandb.log({"Gradcam": wandb.Image(gradcam) })
        plt.imshow(self.image)
        plt.imshow(gradcam, alpha=0.5)

    def guided_gradcam(self):

        #GUIDED GRAD CAM
        last_conv_layer = self.model.get_layer(self.target_layer) #last conv layer
        last_conv_layer_model = tf.keras.Model(self.model.inputs, last_conv_layer.output)
        classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
        x = classifier_input
        names = self.find_names_after(self.target_layer)
        for layer_name in names: # only for CNN rn, need to normalize names
            x = self.model.get_layer(layer_name)(x)
        classifier_model = tf.keras.Model(classifier_input, x)

        with tf.GradientTape() as tape:
            _, _, z_cond = self.cvae.conditional_input([self.image[np.newaxis, ...], self.test_y_one_hot])
            last_conv_layer_output = last_conv_layer_model(z_cond)
            tape.watch(last_conv_layer_output)
            preds = classifier_model(last_conv_layer_output)
            top_pred_index = tf.argmax(preds[0])
            top_class_channel = preds[:, top_pred_index]
            grads = tape.gradient(top_class_channel, last_conv_layer_output)[0]
            last_conv_layer_output = last_conv_layer_output[0]
            guided_grads = (
                tf.cast(last_conv_layer_output > 0, "float32")
                * tf.cast(grads > 0, "float32")
                * grads
            )
            pooled_guided_grads = tf.reduce_mean(guided_grads, axis=(0, 1))
            guided_gradcam = np.ones(last_conv_layer_output.shape[:2], dtype=np.float32)
            for i, w in enumerate(pooled_guided_grads):
                guided_gradcam += w * last_conv_layer_output[:, :, i]
            guided_gradcam = cv2.resize(guided_gradcam.numpy(), (64, 64))
            guided_gradcam = np.clip(guided_gradcam, 0, np.max(guided_gradcam))
            guided_gradcam = (guided_gradcam - guided_gradcam.min()) / (
                    guided_gradcam.max() - guided_gradcam.min()
            )
            wandb.log({"Gradcam_guided": wandb.Image(guided_gradcam) })
            plt.imshow(self.image)
            plt.imshow(guided_gradcam, alpha=0.5)

            if self.HQ:
                gb = GuidedBackprop(self.cvae, self.test_y_one_hot, self.model, self.target_layer)
                saliency_map = gb.guided_backprop(self.image[np.newaxis, ...]).numpy()
                saliency_map = saliency_map * np.repeat(guided_gradcam[..., np.newaxis], 3, axis=2)
                saliency_map -= saliency_map.mean()
                saliency_map /= saliency_map.std() + tf.keras.backend.epsilon()
                saliency_map *= 0.25
                saliency_map += 0.5
                saliency_map = np.clip(saliency_map, 0, 1)
                saliency_map *= (2 ** 8) - 1
                saliency_map = saliency_map.astype(np.uint8)
                wandb.log({"Gradcam_guided_HQ": wandb.Image(saliency_map) })
                plt.imshow(saliency_map)

    def find_names_after(self, target_layer):
        names = []
        flag = False

        for layer in self.model.layers:
            #print(layer.name)
            if flag:
                names.append(layer.name)

            if layer.name == target_layer:
                flag = True
        return names[:-1]
    

class GuidedBackprop:
    def __init__(self, cvae,  test_y_one_hot, model, layer_name: str):
        self.cvae = cvae
        self.test_y_one_hot = test_y_one_hot
        self.model = model
        self.layer_name = layer_name
        self.gb_model = self.build_guided_model()

    def build_guided_model(self):
        gb_model = tf.keras.Model(
            self.model.inputs, self.model.get_layer(self.layer_name).output
        )
        layers = [
            layer for layer in gb_model.layers[1:] if hasattr(layer, "activation")
        ]
        for layer in layers:
        
            if layer.activation == tf.keras.activations.relu:
                layer.activation = self.guided_relu
        return gb_model

    def guided_backprop(self, image: np.ndarray):
        with tf.GradientTape() as tape:
            inputs = tf.cast(image, tf.float32)
            tape.watch(inputs)
            _, _, z_cond = self.cvae.conditional_input([inputs, self.test_y_one_hot])

            outputs = self.gb_model(z_cond)
        grads = tape.gradient(outputs, inputs)[0]
        return grads


@tf.custom_gradient
def guided_relu(x):
    def grad(dy):
        return tf.cast(dy > 0, "float32") * tf.cast(x > 0, "float32") * dy

    return tf.nn.relu(x), grad


