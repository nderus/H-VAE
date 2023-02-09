"""
Classification of histopathological images.
"""

import argparse
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.metrics import confusion_matrix
import numpy as np

from src.datasets import data_loader
from training.classification_utils import cnn_defaults
from training.classification_utils import add_dict_to_argparser

from training.classification_utils import CNN
from training.classification_utils import base_model
from training.classification_utils import Train_Val_Plot
from training.classification_utils import f1_score
from training.classification_utils import show_confusion_matrix
from training.classification_utils import METRICS

def main():
  
    args = create_argparser().parse_args()

    data = data_loader(name = 'histo', root_folder='datasets/')

    if args.synthetic_data:
        synthetic_x = np.load('datasets/synthetic/vae_synthetic_dataset.npy', allow_pickle=True)
        synthetic_x = np.concatenate(synthetic_x, axis=0)
        data['train_x'] = np.concatenate([data['train_x'], synthetic_x], axis=0)
        synthetic_y = np.repeat(1, len(synthetic_x))
        data['train_y'] = np.concatenate([data['train_y'], synthetic_y], axis=0)

    train_y_label = np.argmax(data['train_y'], axis=1) # from one-hot encoding to integer
    test_y_label = np.argmax(data['test_y'], axis=1)
    val_y_label = np.argmax(data['val_y'], axis=1)

    model = CNN((48, 48, 3), 2)
    model.summary()

    optimizer=keras.optimizers.Adam()
    model.compile(optimizer = optimizer, loss='binary_crossentropy', metrics=METRICS)

    history = model.fit(data['train_x'], data['train_y'], args.batch_size, args.epoch_count,
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
    show_confusion_matrix(conf_matrix, args.class_names)

    conf_matrix = confusion_matrix(val_y_label, val_y_pred, normalize=None)
    print(conf_matrix)
    show_confusion_matrix(conf_matrix, args.class_names)

    tn, fp, fn, tp = conf_matrix.ravel()

    sensitivity = tp / (conf_matrix[1][0] + conf_matrix[1][1])
    sensitivity

    specificity = 1 - fp / (conf_matrix[0][0]+ conf_matrix[0][1])  # 1 - FPR; FPR = FP / N
    specificity

    BA = (sensitivity + specificity) / 2

def create_argparser():
    defaults = dict(
        dataset_name = 'histo',
        model_name = 'CNN',
        class_names = ('non-cancer','cancer'),
        batch_size = 250,
        epoch_count = 20,
        synthetic_data = True,   
    )
    defaults.update(cnn_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
  main()

