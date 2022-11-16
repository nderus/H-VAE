from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers

#NB: original had relu and upsample(4)
class DecoderResBlock(keras.Model):
    def __init__(self, filters, upsample):
        super().__init__()
        if upsample:
            self.conv1 = layers.Conv2DTranspose(filters, 3, 2, padding='same', use_bias = False)
            self.shortcut = keras.Sequential([
                layers.Conv2DTranspose(filters, 1, 2, use_bias = False),
                layers.BatchNormalization()
            ])
        else:
            self.conv1 = layers.Conv2DTranspose(filters, 3, 1, padding='same', use_bias = False)
            self.shortcut = keras.Sequential()
 
        self.conv2 = layers.Conv2DTranspose(filters, 3, 1, padding='same', use_bias = False)

    def __call__(self, input):
        shortcut = self.shortcut(input)

        input = self.conv1(input)
        input = layers.BatchNormalization()(input)
        input =  layers.LeakyReLU(0.2)(input)

        input = self.conv2(input)
        input = layers.BatchNormalization()(input)
        input =  layers.LeakyReLU(0.2)(input)


        input = input + shortcut
        return  layers.LeakyReLU(0.2)(input)


class DecoderResNet(keras.Model):
    def __init__(self, resblock, repeat, encoded_dim, final_stride=2):
        super().__init__()
        
        self.layer5 = keras.Sequential([
            resblock(512, upsample=False)
        ] + [
            resblock(512, upsample=False)  for _ in range(1, repeat[3])
        ], name='layer5')


        self.layer6 = keras.Sequential([
            resblock(256, upsample=True)
        ] + [
            resblock(256, upsample=False) for _ in range(1, repeat[2])
        ], name='layer6')


        self.layer7 = keras.Sequential([
            resblock(128, upsample=True)
        ] + [
            resblock(128, upsample=False) for _ in range(1, repeat[1])
        ], name='layer7')
        

        self.layer8 =  keras.Sequential([ 
            resblock(64, upsample=True)
        ] + [
            resblock(64, upsample=False) for _ in range(repeat[0]) 
        ], name='layer8')

        self.layer9 = keras.Sequential([
                layers.Conv2DTranspose(64, 3, 1, padding='same', use_bias = False), 
                #layers.MaxPool2D(pool_size=3, strides=2, padding='same'),
                layers.BatchNormalization(),
                 layers.LeakyReLU(0.2)
            ], name='layer9')
          
        self.bottleneck = layers.Dense(encoded_dim * 2, name='bottleneck')
        self.pre_reshape = layers.Dense(4*4*512, name='pre_reshape')
        self.reshape = layers.Reshape(target_shape=(4, 4, 512), name = 'reshape')
        #self.upsample = layers.UpSampling2D(2)
        self.output_layer = layers.Conv2DTranspose(filters = 3, kernel_size=2, strides=final_stride, activation='sigmoid',padding='same', name='outputs')


    def call(self, input):
        #input = self.bottleneck(input)
        input = self.pre_reshape(input)
        input = self.reshape(input)
        input = self.layer5(input)
        input = self.layer6(input)
        input = self.layer7(input)
        input = self.layer8(input)
        input = self.layer9(input)
        #input = self.upsample(input)
        out = self.output_layer(input)
        return out

    def get_config(self):
        return super().get_config()

class ResBottleneckBlock(keras.Model): #check this
    def __init__(self, filters, upsample):
        super().__init__()
        self.upsample = upsample
        self.filters = filters
        self.conv1 = layers.Conv2DTranspose(filters, 1, 1)
        if upsample:
            self.conv2 = layers.Conv2DTranspose(filters, 3, 2, padding='same')
        else:
            self.conv2 = layers.Conv2DTranspose(filters, 3, 1, padding='same')
        self.conv3 = layers.Conv2DTranspose(filters*4, 1, 1)

    def build(self, input_shape):
        if self.upsample or self.filters * 4 != input_shape[-1]:
            self.shortcut = keras.Sequential([
                layers.Conv2DTranspose(
                    self.filters*4, 1, 2 if self.upsample else 1, padding='same'),
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

class DecoderResNet18(DecoderResNet):
    def __init__(self, encoded_dim, final_stride):
        super().__init__(DecoderResBlock, [2, 2, 2, 2], encoded_dim, final_stride)

    def call(self, input):
        return super().call(input)

    def model(self, input_shape):
        x = keras.Input(input_shape, name='input')
        return keras.models.Model(x, self.call(x), name='decoder')

class DecoderResNet34(DecoderResNet):
    def __init__(self, encoded_dim, final_stride):
        super().__init__(DecoderResBlock, [3, 4, 6, 3], encoded_dim, final_stride)

    def call(self, input):
        return super().call(input)

    def model(self, input_shape):
        x = keras.Input(input_shape, name='input')
        return keras.models.Model(x, self.call(x), name='decoder')


class DecoderResNet50(DecoderResNet):
    def __init__(self, encoded_dim, final_stride):
        super().__init__(ResBottleneckBlock,  [3, 4, 6, 3], encoded_dim, final_stride)

    def call(self, input):
        return super().call(input)

    def model(self, input_shape):
        x = layers.Input(input_shape, name='input', dtype='float32')
        return keras.models.Model(x, self.call(x), name='decoder')


def decoderCNN(input_shape, label_size=10, encoded_dim = 2, final_stride = 2, regularizer=None): 

    decoder_inputs = layers.Input(shape=(encoded_dim + label_size,),
                                 name='decoder_input')
    x = layers.Dense(encoded_dim)

    x = layers.Dense(encoded_dim * 2 )
 
    x = layers.Dense(input_shape[0]/2 * input_shape[1]/2 *64)(decoder_inputs)
   
    x = layers.Reshape(target_shape=(int(input_shape[0]/2),
                     int(input_shape[1]/2), 64))(x)
    x = bn_relu(x) 
    x = layers.Conv2DTranspose(64, (3, 3),
                      padding='same',
                      name='up_block4_conv1', kernel_regularizer=regularizer)(x) #regularizers.L2(.001)
    x = layers.Conv2DTranspose(64, (3, 3),
                    padding='same',
                    name='up_block4_conv2',  kernel_regularizer=regularizer)(x)  
    x = bn_relu(x) 
    # block 2
    x = layers.Conv2DTranspose(32, (3, 3),
                      padding='same',
                      name='up_block5_conv1',  kernel_regularizer=regularizer)(x)
    x = layers.Conv2DTranspose(32, (3, 3),
                      padding='same',
                      name='up_block5_conv2',  kernel_regularizer=regularizer)(x)
    x = bn_relu(x) 
    x = layers.UpSampling2D()(x)
    
    # block 3
    x = layers.Conv2DTranspose(16, (3, 3),
                      padding='same',
                      name='up_block6_conv1',  kernel_regularizer=regularizer)(x)

    x = layers.Conv2DTranspose(16, (3, 3),
                    padding='same',
                    name='up_block6_conv2',  kernel_regularizer=regularizer)(x)
    x = bn_relu(x)                                
    outputs = layers.Conv2DTranspose(filters=input_shape[-1], kernel_size=1, #was 3 (!)
                             strides=final_stride, activation='sigmoid',padding='same')(x)

    model = keras.Model(decoder_inputs, outputs, name='decoder')
    return model

def decoderVGG19(input_shape, label_size=10, encoded_dim = 2, final_stride = 2, regularizer=None): 

    decoder_inputs = layers.Input(shape=(encoded_dim + label_size,),
                                 name='decoder_input')
    x = layers.Dense(encoded_dim)

    x = layers.Dense(encoded_dim * 2 )
 
    x = layers.Dense(3 * 3 * 512)(decoder_inputs)
   
    x = layers.Reshape(target_shape=(3, 3, 512))(x)
 
    x = layers.Conv2DTranspose(512, (3, 3),
                      padding='same',
                      activation = 'relu',
                      name='up_block4_conv1', kernel_regularizer=regularizer)(x) #regularizers.L2(.001)
    x = layers.Conv2DTranspose(512, (3, 3),
                    padding='same',
                     activation = 'relu',
                    name='up_block4_conv2',  kernel_regularizer=regularizer)(x)  
    x = layers.Conv2DTranspose(512, (3, 3),
                padding='same',
                 activation = 'relu',
                name='up_block4_conv3',  kernel_regularizer=regularizer)(x)  
    x = layers.Conv2DTranspose(512, (3, 3),
                padding='same',
                 activation = 'relu',
                name='up_block4_conv4',  kernel_regularizer=regularizer)(x)    
    x = layers.UpSampling2D()(x)
    # block 2
    x = layers.Conv2DTranspose(512, (3, 3),
                      padding='same',
                      activation = 'relu',
                      name='up_block5_conv1', kernel_regularizer=regularizer)(x) #regularizers.L2(.001)
    x = layers.Conv2DTranspose(512, (3, 3),
                    padding='same',
                     activation = 'relu',
                    name='up_block5_conv2',  kernel_regularizer=regularizer)(x)  
    x = layers.Conv2DTranspose(512, (3, 3),
                padding='same',
                 activation = 'relu',
                name='up_block5_conv3',  kernel_regularizer=regularizer)(x)  
    x = layers.Conv2DTranspose(512, (3, 3),
                padding='same',
                 activation = 'relu',
                name='up_block5_conv4',  kernel_regularizer=regularizer)(x)    
    x = layers.UpSampling2D()(x)

    # block 3
    x = layers.Conv2DTranspose(256, (3, 3),
                      padding='same',
                      activation = 'relu',
                      name='up_block6_conv1', kernel_regularizer=regularizer)(x) #regularizers.L2(.001)
    x = layers.Conv2DTranspose(256, (3, 3),
                    padding='same',
                     activation = 'relu',
                    name='up_block6_conv2',  kernel_regularizer=regularizer)(x)  
    x = layers.Conv2DTranspose(256, (3, 3),
                padding='same',
                 activation = 'relu',
                name='up_block6_conv3',  kernel_regularizer=regularizer)(x)  
    x = layers.Conv2DTranspose(256, (3, 3),
                padding='same',
                 activation = 'relu',
                name='up_block6_conv4',  kernel_regularizer=regularizer)(x)    
    x = layers.UpSampling2D()(x)   
    
    #block 4
    x = layers.Conv2DTranspose(128, (3, 3),
                      padding='same',
                      activation = 'relu',
                      name='up_block7_conv1', kernel_regularizer=regularizer)(x) #regularizers.L2(.001)
    x = layers.Conv2DTranspose(128, (3, 3),
                    padding='same',
                     activation = 'relu',
                      name='up_block7_conv2', kernel_regularizer=regularizer)(x)
    x = layers.UpSampling2D()(x)
    x = layers.Conv2DTranspose(64, (3, 3),
                      padding='same',
                      activation = 'relu',
                      name='up_block7_conv3', kernel_regularizer=regularizer)(x) #regularizers.L2(.001)
    x = layers.Conv2DTranspose(64, (3, 3),
                    padding='same',
                    activation = 'relu',
                    name='up_block7_conv4', kernel_regularizer=regularizer)(x)
    
    outputs = layers.Conv2DTranspose(filters=input_shape[-1], kernel_size=1, #was 3 (!)
                             strides=final_stride, activation='sigmoid',padding='same')(x)

    model = keras.Model(decoder_inputs, outputs, name='decoder')
    return model

def bn_relu(inputs):
    bn = layers.BatchNormalization()(inputs)
    relu = layers.LeakyReLU(0.2)(bn)
    return(relu)

#
def decoder2(encoded_dim, category_count, second_dim, second_depth):

    u_cond = layers.Input(shape=(encoded_dim + category_count,), dtype='float32',
                name='Input')

    u = layers.Dense(second_dim, name='fc')(u_cond)
    u = layers.LeakyReLU(0.2)(u)

    for i in range(second_depth - 1):
        u = layers.Dense(second_dim, name='fc'+str(i))(u)
        u = layers.LeakyReLU(0.2)(u)

    u = layers.Concatenate(axis=-1)([u_cond, u]) 
    z_hat = layers.Dense(encoded_dim + category_count, name='z_hat')(u)
    model = keras.Model(u_cond, z_hat, name='decoder2')
    return model
