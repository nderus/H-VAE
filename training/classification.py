"""
Classification of histopathological images.
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras.backend as K
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from src.datasets import data_loader

batch_size = 250
epoch_count = 20
class_names = ('non-cancer','cancer')

def Train_Val_Plot(acc,val_acc,loss,val_loss,auc,val_auc,precision,val_precision,f1,val_f1):
    
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
    #block 1 - pretrained
    x = base_model.get_layer('block1_conv1')(inputs)
    x.trainable=False

    x = base_model.get_layer('block1_conv2')(x)
    x.trainable=False

    # block 2
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv1')(x)

    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv2')(x)
    
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    x = layers.BatchNormalization()(x)

# Block 3
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv1')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv2')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    x = layers.BatchNormalization()(x)

      # Block 4
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv1')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv2')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    x = layers.BatchNormalization()(x)

    # classifier
    x = layers.Flatten()(x)
    
    x = layers.Dense(120, activation='relu',name='dense1')(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(120, activation='relu', name='dense2')(x)
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

METRICS = [
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),  
      tf.keras.metrics.AUC(name='auc'),
      f1_score,
]

data = data_loader(name = 'histo', root_folder='datasets/')
train_y_label = np.argmax(data['train_y'], axis=1) # from one-hot encoding to integer
test_y_label = np.argmax(data['test_y'], axis=1)
val_y_label = np.argmax(data['val_y'], axis=1)

base_model = keras.applications.VGG19(
    weights='imagenet',  # Load weights pre-trained on ImageNet.
    input_shape=(48, 48, 3),
    include_top=False)  # Do not include the ImageNet classifier at the top.

base_model.trainable = False

model = CNN((48, 48, 3), 2)
model.summary()

optimizer=keras.optimizers.Adam()
model.compile(optimizer = optimizer, loss='binary_crossentropy', metrics=METRICS)

history = model.fit(data['train_x'], data['train_y'], batch_size, epoch_count,
         validation_data = (data['val_x'], data['val_y']))

Train_Val_Plot(history.history['accuracy'],history.history['val_accuracy'],
               history.history['loss'],history.history['val_loss'],
               history.history['auc'],history.history['val_auc'],
               history.history['precision'],history.history['val_precision'],
               history.history['f1_score'],history.history['val_f1_score'])

test_conf_pred = model.predict(data['test_x'])
print('Output predictions shape: ',test_conf_pred.shape)

test_y_pred = np.argsort(test_conf_pred, axis=1)[:,-1]
print('Class predictions shape: ',test_y_pred.shape)

test_conf_pred = model.predict(data['val_x'])
print('Output predictions shape: ',test_conf_pred.shape)

val_y_pred = np.argsort(test_conf_pred, axis=1)[:,-1]
print('Class predictions shape: ',val_y_pred.shape)


conf_matrix = confusion_matrix(test_y_label, test_y_pred, normalize='all')
print(conf_matrix)
show_confusion_matrix(conf_matrix,class_names)

conf_matrix = confusion_matrix(val_y_label, val_y_pred, normalize=None)
print(conf_matrix)
show_confusion_matrix(conf_matrix,class_names)

tn, fp, fn, tp = conf_matrix.ravel()

sensitivity = tp / (conf_matrix[1][0] + conf_matrix[1][1])
sensitivity

specificity = 1 - fp / (conf_matrix[0][0]+ conf_matrix[0][1])  # 1 - FPR; FPR = FP / N
specificity

BA = (sensitivity + specificity) / 2