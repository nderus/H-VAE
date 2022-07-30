from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers


class EncoderResBlock(keras.Model):
    def __init__(self, filters, downsample):
        super().__init__()
        if downsample:
            self.conv1 = layers.Conv2D(filters, 3, 2, padding='same')
            self.shortcut = keras.Sequential([
                layers.Conv2D(filters, 1, 2),
                layers.BatchNormalization()
            ])
        else:
            self.conv1 = layers.Conv2D(filters, 3, 1, padding='same')
            self.shortcut = keras.Sequential()
 
        self.conv2 = layers.Conv2D(filters, 3, 1, padding='same')
        
    def __call__(self, input):
        
        shortcut = self.shortcut(input)

        input = self.conv1(input)
        input = layers.BatchNormalization()(input)
        input =  layers.LeakyReLU(0.2)(input)
        #input = layers.LeakyReLU(0.2)(input)
        input = self.conv2(input)
        input = layers.BatchNormalization()(input)
        input =  layers.LeakyReLU(0.2)(input)
        #input = layers.LeakyReLU(0.2)(input)

        input= input + shortcut
        return  layers.LeakyReLU(0.2)(input)


class EncoderResNet(keras.Model):
    def __init__(self, resblock, repeat, encoded_dim):
   
        super().__init__()
        
        self.layer0 = keras.Sequential([
            layers.Conv2D(64, 3, 2, padding='same'),
            layers.MaxPool2D(pool_size=3, strides=1, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2)
        ], name='layer0')

        self.layer1 = keras.Sequential([
            resblock(64, downsample=False) for _ in range(repeat[0])
        ], name='layer1')

        self.layer2 = keras.Sequential([
            resblock(128, downsample=True)
        ] + [
            resblock(128, downsample=False) for _ in range(1, repeat[1])
        ], name='layer2')

        self.layer3 = keras.Sequential([
            resblock(256, downsample=True)
        ] + [
            resblock(256, downsample=False) for _ in range(1, repeat[2])
        ], name='layer3')

        self.layer4 = keras.Sequential([
            resblock(512, downsample=True)
        ] + [
            resblock(512, downsample=False) for _ in range(1, repeat[3])
        ], name='layer4')

        self.flat = layers.Flatten(name = 'flatten')
        self.bottleneck = layers.Dense(encoded_dim * 2, name='encoder_bottleneck')
        self.mu = layers.Dense(encoded_dim, name='mu')
        self.log_var = layers.Dense(encoded_dim, name='log_var')
 

    def call(self, input):
        input = self.layer0(input)
        input = self.layer1(input)
        input = self.layer2(input)
        input = self.layer3(input)
        input = self.layer4(input)
        input = self.flat(input)
        input = self.bottleneck(input)
        mu = self.mu(input)
        log_var = self.log_var(input)

        return [mu, log_var]

    def get_config(self):
        return super().get_config()


class ResBottleneckBlock(keras.Model): #check this
    def __init__(self, filters, downsample):
        super().__init__()
        self.downsample = downsample
        self.filters = filters
        self.conv1 = layers.Conv2D(filters, 1, 1, padding = 'same')
        if downsample:
            self.conv2 = layers.Conv2D(filters, 3, 2, padding='same')
        else:
            self.conv2 = layers.Conv2D(filters, 3, 1, padding='same')
        self.conv3 = layers.Conv2D(filters*4, 1, 1, padding = 'same')

    def build(self, input_shape):
        if (self.downsample) or (self.filters * 4 != input_shape[-1]):
            self.shortcut = keras.Sequential([
                layers.Conv2D(
                    self.filters*4, 1, 2 if self.downsample else 1, padding='same'),
                layers.BatchNormalization()
            ])
        else:
            self.shortcut = keras.Sequential()

    def __call__(self, input):
        shortcut = self.shortcut(input)

        input = self.conv1(input)
        input = layers.BatchNormalization()(input)
        #input = layers.LeakyReLU(0.2)(input)
        input =  layers.LeakyReLU(0.2)(input)

        input = self.conv2(input)
        input = layers.BatchNormalization()(input)
        #input = layers.LeakyReLU(0.2)(input)
        input =  layers.LeakyReLU(0.2)(input)

        input = self.conv3(input)
        input = layers.BatchNormalization()(input)
        #input = layers.LeakyReLU(0.2)(input)
        input =  layers.LeakyReLU(0.2)(input)

        input = input + shortcut
        return  layers.LeakyReLU(0.2)(input)

class EncoderResNet18(EncoderResNet):
    def __init__(self, encoded_dim):
        super().__init__(EncoderResBlock, [2, 2, 2, 2], encoded_dim)

    def call(self, input):
        return super().call(input)

    def model(self, input_shape):
        x = layers.Input(input_shape, name='input', dtype='float32')
        return keras.models.Model(x, self.call(x), name='encoder')

class EncoderResNet34(EncoderResNet):
    def __init__(self, encoded_dim):
        super().__init__(EncoderResBlock,  [3, 4, 6, 3], encoded_dim)

    def call(self, input):
        return super().call(input)

    def model(self, input_shape):
        x = layers.Input(input_shape, name='input', dtype='float32')
        return keras.models.Model(x, self.call(x), name='encoder')

class EncoderResNet50(EncoderResNet):
    def __init__(self, encoded_dim):
        super().__init__(ResBottleneckBlock,  [3, 4, 6, 3], encoded_dim)

    def call(self, input):
        return super().call(input)

    def model(self, input_shape):
        x = layers.Input(input_shape, name='input', dtype='float32')
        return keras.models.Model(x, self.call(x), name='encoder')

def encoderCNN( input_shape = (28, 28, 1),  label_size=2, encoded_dim = 2, regularizer = regularizers.L2(.001)):   

    inputs = layers.Input(shape=(input_shape[0],
                input_shape[1], input_shape[2] + label_size), dtype='float32',
                name='Input')

    x = layers.Conv2D(16, (3, 3),
                      padding='same',
                      name='block1_conv1', kernel_regularizer=regularizer)(inputs)
    x = layers.Conv2D(16, (3, 3),
                      padding='same',
                      name='block1_conv2', kernel_regularizer=regularizer)(x)
    x = bn_relu(x)
    # block 2
    x = layers.Conv2D(32, (3, 3),
                      padding='same',
                      name='block2_conv1', kernel_regularizer=regularizer)(x)
    x = layers.Conv2D(32, (3, 3),
                      padding='same',
                      name='block2_conv2', kernel_regularizer=regularizer)(x)

    x = bn_relu(x)
    x = layers.MaxPool2D(pool_size=2, strides=2,name='S4')(x)

    # block 3
    x = layers.Conv2D(64, (3, 3),
                      padding='same',
                      name='block3_conv1', kernel_regularizer=regularizer)(x)
    x = layers.Conv2D(64, (3, 3),
                padding='same',
                name='block3_conv2', kernel_regularizer=regularizer)(x)    
    x = bn_relu(x)            
    x = layers.Flatten()(x)
    y = layers.Dense(encoded_dim )(x) #removed 2*
    mu = layers.Dense(encoded_dim, name='mu')(y)
    log_var = layers.Dense(encoded_dim, name='log_var')(y)
    

    model = keras.Model(inputs, [mu, log_var], name='encoder')
    
    return model


def bn_relu(inputs):
    bn = layers.BatchNormalization()(inputs)
    relu =  layers.LeakyReLU(0.2)(bn)
    return(relu)


def bn_swish(inputs):
    bn = layers.BatchNormalization()(inputs)
    swish = layers.Activation('swish')(inputs)
    return(swish)

#############
# To DO. maxpool is in different place
class EncoderMixNet(keras.Model):
    def __init__(self, resblock, repeat, encoded_dim, regularizer = regularizers.L2(.001)):
   
        super().__init__()
        
        self.layer0 = keras.Sequential([
            layers.Conv2D(16, 3,  padding='same',  kernel_regularizer=regularizer),
            layers.MaxPool2D(pool_size=3, strides=1, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2)
        ], name='layer0')

        self.layer1 = keras.Sequential([
            layers.Conv2D(16, 3,  padding='same',  kernel_regularizer=regularizer),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2)
        ], name='layer1')
 
        self.layer2 = keras.Sequential([
            layers.Conv2D(32, 3, padding='same',  kernel_regularizer=regularizer),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2)
        ], name='layer2')

        self.layer3 = keras.Sequential([
            layers.Conv2D(32, 3,  padding='same',  kernel_regularizer=regularizer),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2)
        ], name='layer3')

        self.layer4 = keras.Sequential([
            resblock(64, downsample=True) for _ in range(repeat[0])
        ], name='layer4')

        self.flat = layers.Flatten(name = 'flatten')
        self.bottleneck = layers.Dense(encoded_dim * 2, name='encoder_bottleneck')
        self.mu = layers.Dense(encoded_dim, name='mu')
        self.log_var = layers.Dense(encoded_dim, name='log_var')
 

    def call(self, input):
        input = self.layer0(input)
        input = self.layer1(input)
        input = self.layer2(input)
        input = self.layer3(input)
        input = self.layer4(input)
        input = self.flat(input)
        input = self.bottleneck(input)
        mu = self.mu(input)
        log_var = self.log_var(input)

        return [mu, log_var]

    def get_config(self):
        return super().get_config()

class EncoderMixNet18(EncoderMixNet):
    def __init__(self, encoded_dim):
        super().__init__(EncoderResBlock, [1, 2, 2, 2], encoded_dim, regularizer = None)

    def call(self, input):
        return super().call(input)

    def model(self, input_shape):
        x = layers.Input(input_shape, name='input', dtype='float32')
        return keras.models.Model(x, self.call(x), name='encoder')


#
import tensorflow as tf
def encoder2(encoded_dim, category_count, second_dim, second_depth):
    z_cond = layers.Input(shape=(encoded_dim + category_count,), dtype='float32',
            name='Input')
    t = layers.Dense(second_dim, name='fc')(z_cond)
    t = layers.LeakyReLU(0.2)(t)
    for i in range(second_depth - 1):
        t = layers.Dense(second_dim, name='fc'+str(i))(t)
        t = layers.LeakyReLU(0.2)(t)
    t = layers.Concatenate(axis=-1)([z_cond, t]) 

    mu_u = layers.Dense(encoded_dim, name='mu_u')(t)
    logsd_u = layers.Dense( encoded_dim, name='logsd_u')(t)
    sd_u = tf.exp(logsd_u)
    model = keras.Model(z_cond, [mu_u, logsd_u, sd_u], name='encoder2')
    return model