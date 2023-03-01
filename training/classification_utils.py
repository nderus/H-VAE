import argparse
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np

def cnn_defaults():
    """
    Defaults for CNN models.
    """
    return dict(
        dataset_name = 'histo',
        model_name = 'CNN',
        class_names = ('non-cancer','cancer'),
        batch_size = 64, #was 250
        epoch_count = 100,
        learning_rate =  10e-4,
        vae_synthetic_data = True,  
        ddpm_synthetic_data = False,
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

def Train_Val_Plot(acc, val_acc, loss, val_loss, auc, val_auc, precision, val_precision, f1, val_f1):
    
    fig, (ax1, ax2,ax3,ax4,ax5) = plt.subplots(1,5, figsize= (16,4))
    fig.suptitle(" MODEL'S METRICS VISUALIZATION ")

    ax1.plot(range(1, len(acc) + 1), acc)
    ax1.plot(range(1, len(val_acc) + 1), val_acc)
    ax1.set_title('History of Accuracy')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    #ax1.legend(['training', 'validation'])


    ax2.plot(range(1, len(loss) + 1), loss)
    ax2.plot(range(1, len(val_loss) + 1), val_loss)
    ax2.set_title('History of Loss')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    #ax2.legend(['training', 'validation'])
    
    ax3.plot(range(1, len(auc) + 1), auc)
    ax3.plot(range(1, len(val_auc) + 1), val_auc)
    ax3.set_title(' History of AUC ')
    ax3.set_xlabel(' Epochs ')
    ax3.set_ylabel('AUC')
    #ax3.legend(['training', 'validation'])
    
    ax4.plot(range(1, len(precision) + 1), precision)
    ax4.plot(range(1, len(val_precision) + 1), val_precision)
    ax4.set_title('History of Precision')
    ax4.set_xlabel('Epochs')
    ax4.set_ylabel('Precision')
    #ax4.legend(['training', 'validation'])
    
    ax5.plot(range(1, len(f1) + 1), f1)
    ax5.plot(range(1, len(val_f1) + 1), val_f1)
    ax5.set_title('History of F1-score')
    ax5.set_xlabel('Epochs')
    ax5.set_ylabel('F1 score')
    #ax5.legend(['training', 'validation'])


    plt.show()
    
def show_confusion_matrix(conf_matrix,class_names,figsize=(10,10)):
  fig, ax = plt.subplots(figsize=figsize)
  img=ax.matshow(conf_matrix)
  tick_marks = np.arange(len(class_names))
  _=plt.xticks(tick_marks, class_names,rotation=45)
  _=plt.yticks(tick_marks, class_names)
  _=plt.ylabel('Real')
  _=plt.xlabel('Predicted')
  
  for i in range(len(class_names)):
    for j in range(len(class_names)):
        text = ax.text(j, i, '{0:.1%}'.format(conf_matrix[i, j]),
                       ha='center', va='center', color='w')
        
        
def CNN(input_shape=(48, 48, 3), output_class_count=2):
                
    inputs = layers.Input(shape=input_shape,name='Input')

    x = base_model.get_layer('block1_conv1')(inputs)
    x.trainable=True

    x = base_model.get_layer('block1_conv2')(x)
    x.trainable=True


    x = layers.Conv2D(filters=6, kernel_size=5, strides=1,padding='valid',name='conv1_1')(x)
    #x = layers.Conv2D(filters=6, kernel_size=5, strides=1,padding='same',name='conv1_2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=2, strides=2,name='S1')(x)
    

# layer 2
    x = layers.Conv2D(filters=16, kernel_size=5,strides=1,padding='valid',name='conv2_1')(x)
    #x = layers.Conv2D(filters=16, kernel_size=5,strides=1,padding='same',name='conv2_2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=2, strides=2,name='S2')(x)
    

# layer 3
    x = layers.Conv2D(filters=120, kernel_size=5, strides=1,padding='valid',name='conv3_1')(x)
   # x = layers.Conv2D(filters=120, kernel_size=5, strides=1,padding='same',name='conv3_2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Flatten()(x)

    x = layers.Dropout(0.1)(x)
    x = layers.Dense(84, activation='relu',name='F6')(x)
    outputs = layers.Dense(units=output_class_count,activation='softmax',name='Output')(x)
    
    model = keras.Model(inputs, outputs)
    return model
        
def CNN2(input_shape=(48, 48, 3), output_class_count=2):
    
    inputs = layers.Input(shape=input_shape,name='Input')
    #block 1 - pretrained
    x = base_model.get_layer('block1_conv1')(inputs)
    x.trainable=True

    x = base_model.get_layer('block1_conv2')(x)
    x.trainable=True

    # block 2
    x = layers.Conv2D(32, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv1', kernel_regularizer=regularizers.L2(.01))(x)

    x = layers.Conv2D(32, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv2', kernel_regularizer=regularizers.L2(.01))(x)
    
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    x = layers.BatchNormalization()(x)

# Block 3
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv1', kernel_regularizer=regularizers.L2(.01))(x)
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv2', kernel_regularizer=regularizers.L2(.01))(x)
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv3', kernel_regularizer=regularizers.L2(.01))(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    x = layers.BatchNormalization()(x)

      # Block 4
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv1', kernel_regularizer=regularizers.L2(.01))(x)
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv2', kernel_regularizer=regularizers.L2(.01))(x)
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv3', kernel_regularizer=regularizers.L2(.01))(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    x = layers.BatchNormalization()(x)

    # classifier
    x = layers.Flatten()(x)
    
    x = layers.Dense(84, activation='relu',name='dense1')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(84, activation='relu', name='dense2')(x)
    outputs = layers.Dense(units=output_class_count,activation='softmax',name='Output')(x)

    model = keras.Model(inputs, outputs)
    return model

def f1_score(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

base_model = keras.applications.VGG19(
    weights='imagenet',  # Load weights pre-trained on ImageNet.
    input_shape=(48, 48, 3),
    include_top=False)  # Do not include the ImageNet classifier at the top.

base_model.trainable = False

METRICS = [
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),  
      tf.keras.metrics.AUC(name='auc'),
      f1_score,
]