from tensorflow import keras
from tensorflow.keras import layers

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
        input = layers.ReLU()(input)

        input = self.conv2(input)
        input = layers.BatchNormalization()(input)
        input = layers.ReLU()(input)

        input = input + shortcut
        return layers.ReLU()(input)

class DecoderResNet(keras.Model):
    def __init__(self, resblock, repeat, encoded_dim):
        super().__init__()
        
        self.layer5 = keras.Sequential([
            resblock(512, upsample=False)
        ] + [
            resblock(512, upsample=False)  for _ in range(1, repeat[0])
        ], name='layer5')


        self.layer6 = keras.Sequential([
            resblock(256, upsample=True)
        ] + [
            resblock(256, upsample=False) for _ in range(1, repeat[1])
        ], name='layer6')


        self.layer7 = keras.Sequential([
            resblock(128, upsample=True)
        ] + [
            resblock(128, upsample=False) for _ in range(1, repeat[2])
        ], name='layer7')
        
        # self.layer7 = keras.Sequential([ # TO DO: change back this into resblock, heigth/depth issue
        #     layers.Conv2DTranspose(128, 4, 1, padding='valid'),
        #     #layers.BatchNormalization(),
        #     layers.ReLU()
        # ], name='layer7')

        self.layer8 =  keras.Sequential([ 
            resblock(64, upsample=True)
        ] + [
            resblock(64, upsample=False) for _ in range(repeat[3])
        ], name='layer8')

        self.layer9 = keras.Sequential([
                layers.Conv2DTranspose(64, 7, 1, padding='same'),
                #layers.MaxPool2D(pool_size=3, strides=2, padding='same'),
                layers.BatchNormalization(),
                layers.ReLU()
            ], name='layer9')
          
        self.bottleneck = layers.Dense(encoded_dim * 2, name='bottleneck')
        self.pre_reshape = layers.Dense(2*2*512, name='pre_reshape')
        self.reshape = layers.Reshape(target_shape=(2, 2, 512), name = 'reshape')
        self.output_layer = layers.Conv2DTranspose(filters = 3, kernel_size=1, strides=1, activation='sigmoid' ,padding='valid', name='outputs')
        self.upsample = layers.UpSampling2D(4)

    def call(self, input):
        #input = self.bottleneck(input)
        input = self.pre_reshape(input)
        input = self.reshape(input)
        input = self.layer5(input)
        input = self.layer6(input)
        input = self.layer7(input)
        input = self.layer8(input)
        input = self.layer9(input)
        input = self.output_layer(input)
        out = self.upsample(input)
        return out

    def get_config(self):
        return super().get_config()

class DecoderResNet18(DecoderResNet):
    def __init__(self, encoded_dim):
        super().__init__(DecoderResBlock, [2, 2, 2, 2], encoded_dim)

    def call(self, input):
        return super().call(input)

    def model(self, input_shape):
        x = keras.Input(input_shape, name='input')
        return keras.models.Model(x, self.call(x))