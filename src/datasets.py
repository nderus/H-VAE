import os
import numpy as np
import pandas as pd
import random
import tensorflow as tf
import tensorflow_datasets as tfds
import cv2
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from glob import glob

# TO DO: same method for loading, maybe TFRecord datasets

def data_loader(name, root_folder):
    """
    For a dataset, load training, test and validation set, along with info
    regarding the shape of the images and the number of categories.
    :param name: the name of the dataset
    :param root folder: a dataset directory.
    """
    if name.lower() == 'mnist':
        category_count = 10
        (train_x, train_y), (test_x, test_y) = keras.datasets.mnist.load_data()
        train_x, val_x, train_y, val_y = train_test_split(train_x, train_y,
                        test_size = 10000, shuffle=False, random_state=11)
        train_x = np.expand_dims(train_x, axis=3)
        val_x = np.expand_dims(val_x, axis=3)
        test_x = np.expand_dims(test_x, axis=3)
        train_y_one_hot = to_categorical(train_y, category_count)
        val_y_one_hot = to_categorical(val_y, category_count)
        test_y_one_hot = to_categorical(test_y, category_count)
        input_shape = (28, 28, 1)
        labels = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

    elif name.lower() == 'fashion_mnist':
        category_count = 10
        (train_x, train_y), (test_x, test_y) = keras.datasets.fashion_mnist.load_data()
        train_x, val_x, train_y, val_y = train_test_split(train_x, train_y,
                test_size = 10000, shuffle=False, random_state=11)
        train_x = np.expand_dims(train_x, axis=3)
        val_x = np.expand_dims(val_x, axis=3)
        test_x = np.expand_dims(test_x, axis=3)
        train_y_one_hot = to_categorical(train_y, category_count)
        val_y_one_hot = to_categorical(val_y, category_count)
        test_y_one_hot = to_categorical(test_y, category_count)
        input_shape = (28, 28, 1)

        labels = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

    elif name.lower() == 'cifar10':
        category_count = 10
        (train_x, train_y), (test_x, test_y) = keras.datasets.cifar10.load_data()
        train_x, val_x, train_y, val_y = train_test_split(train_x, train_y,
                test_size = 10000, shuffle=False, random_state=11)
        train_y_one_hot = to_categorical(train_y, category_count)
        val_y_one_hot = to_categorical(val_y, category_count)
        test_y_one_hot = to_categorical(test_y, category_count)
        input_shape = (32, 32, 3)
        labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    elif name.lower() == 'celeba':
        category_count = 40
        data_folder = os.path.join(root_folder, name)
        df = pd.read_csv(os.path.join(data_folder, "list_attr_celeba.csv") , header = 0, index_col = 0).replace(-1,0)

        train_x = np.load(os.path.join(data_folder, 'train.npy'))
        val_x = np.load(os.path.join(data_folder, 'val.npy'))
        test_x = np.load(os.path.join(data_folder, 'test.npy'))

        train_x = np.array(train_x, dtype = "float")
        val_x = np.array(val_x, dtype = "float")
        test_x = np.array(test_x, dtype = "float")

        train_y = np.array(df.iloc[0: 162770], dtype = "float32") #TO DO: check the splits
        val_y = np.array(df.iloc[162770: 182637], dtype = "float32")
        test_y = np.array(df.iloc[182637: 202599], dtype = "float32")
        train_x = train_x[:100000,:, :, :] # TO DO: solve batch_size issue
        train_y = train_y[:100000, :] 

        test_x = test_x[:19000,:, :, :]
        test_y = test_y[:19000, :]

        val_x = val_x[:19000, :, :, :]
        val_y = val_y[:19000, :]
        train_y_one_hot = train_y
        val_y_one_hot = val_y
        test_y_one_hot = test_y
        input_shape = (64, 64, 3)
     
        labels = df.columns
    
    elif name.lower() == 'histo':
        category_count = 2 
        imagePatches = glob(root_folder +'/breast-histopathology/IDC_regular_ps50_idx5/**/*.png', recursive=True)
        class0 = [] # 0 = no cancer
        class1 = [] # 1 = cancer

        for filename in imagePatches:
            if filename.endswith("class0.png"):
                class0.append(filename)
            else:
                class1.append(filename)

        random.seed(11)
        sampled_class0 = random.sample(class0, 78000) # TO DO: use whole dataset
        random.seed(11)
        sampled_class1 = random.sample(class1, 78000)
        class0_array = get_image_arrays(sampled_class0, 0)
        class1_array = get_image_arrays(sampled_class1, 1)
        combined_data = np.concatenate((class0_array, class1_array))

        X = []
        y = []

        for features, label in combined_data:
            X.append(features)
            y.append(label) 
        
        X = np.array(X).reshape(-1, 48, 48, 3) #was 64
        y = np.array(y)
        train_x, test_x, train_y, test_y = train_test_split(X, y, test_size = 0.2, random_state=11, shuffle=False)
        train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size = 0.25, random_state=11, shuffle=False)
        train_y_one_hot = to_categorical(train_y, category_count)
        test_y_one_hot = to_categorical(test_y, category_count)
        val_y_one_hot = to_categorical(val_y, category_count) 
        input_shape = (48, 48, 3) #was 64
                              
        labels = ['non-cancer','cancer']
    
    elif name.lower() == 'experimental':
  
        (train_ds, val_ds), info = tfds.load("histo", split=[ "cancer[:30000] + non-cancer[:30000]", "cancer[30000:40000] + non-cancer[30000:40000]"], as_supervised=True, shuffle_files=False, with_info=True)

        train_ds = train_ds.map(
                normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.map(
                normalize_img, num_parallel_calls=tf.data.AUTOTUNE)

        train_ds = train_ds.batch(100).map(lambda x, y: (x, tf.one_hot(y, depth=2))).repeat()
        train_ds = train_ds.cache()
        train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

        val_ds = val_ds.batch(100).map(lambda x, y: (x, tf.one_hot(y, depth=2))).repeat()
        val_ds = val_ds.cache()
        val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
        input_shape = (64, 64, 3)
        labels = ['non-cancer','cancer']
        category_count = 2
        return train_ds, val_ds, input_shape, category_count, labels

    else:
        raise Exception('No such dataset called {}.'.format(name))

    return dict(train_x = resize(train_x),
               test_x = resize(test_x),
               val_x = resize(val_x),
               train_y = train_y,
               test_y = test_y,
               val_y = val_y,
               train_y_one_hot = train_y_one_hot,
               test_y_one_hot = test_y_one_hot,
               val_y_one_hot = val_y_one_hot,
               input_shape = input_shape,
               category_count = category_count,
               labels = labels
    )

def get_image_arrays(data, label):
    img_arrays = []
    for i in data:
        if i.endswith('.png'):
            img = cv2.imread(i ,cv2.IMREAD_COLOR)
            img_sized = cv2.resize(img, (48, 48), #was (64,64) 
                        interpolation=cv2.INTER_LINEAR)
            img_arrays.append([img_sized, label]) 
    return img_arrays

def resize(data):
    data = data / 255.0
    return(data)

def normalize_img(image, label):
    return tf.cast(image, tf.float32) / 255., label