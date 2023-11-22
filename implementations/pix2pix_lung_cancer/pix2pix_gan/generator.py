from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model
from keras.layers import (
    Input,
    Conv2D,
    Conv2DTranspose,
    LeakyReLU,
    Activation,
    Concatenate,
    BatchNormalization,
    Dropout,
)
from typing import Tuple


# define an encoder block
def create_encoder_block(layer_in, n_filters, batchnorm=True):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # add downsampling layer
    g = Conv2D(
        n_filters, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init
    )(layer_in)
    # conditionally add batch normalization
    if batchnorm:
        g = BatchNormalization()(g, training=True)
    # leaky relu activation
    g = LeakyReLU(alpha=0.2)(g)
    return g


# define a decoder block
def create_decoder_block(layer_in, skip_in, n_filters, dropout=True):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # add upsampling layer
    g = Conv2DTranspose(
        n_filters, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init
    )(layer_in)
    # add batch normalization
    g = BatchNormalization()(g, training=True)
    # conditionally add dropout
    if dropout:
        g = Dropout(0.5)(g, training=True)
    # merge with skip connection
    g = Concatenate()([g, skip_in])
    # relu activation
    g = Activation("relu")(g)
    return g


# define the standalone generator model
def create_generator(
    image_shape: Tuple[int, int, int] | Tuple[int, int] = (256, 256, 3)
) -> "Model":
    """
    Encoder: C64-C128-C256-C512-C512-C512-C512-C512
    Decoder: CD512-CD1024-CD1024-C1024-C1024-C512-C256-C128
    """

    # weight initialization
    init = RandomNormal(stddev=0.02)
    # image input
    in_image = Input(shape=image_shape)
    # encoder model: C64-C128-C256-C512-C512-C512-C512-C512
    e1 = create_encoder_block(in_image, 64, batchnorm=False)
    e2 = create_encoder_block(e1, 128)
    e3 = create_encoder_block(e2, 256)
    e4 = create_encoder_block(e3, 512)
    e5 = create_encoder_block(e4, 512)
    e6 = create_encoder_block(e5, 512)
    e7 = create_encoder_block(e6, 512)
    # bottleneck, no batch norm and relu
    b = Conv2D(512, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init)(e7)
    b = Activation("relu")(b)
    # decoder model: CD512-CD1024-CD1024-C1024-C1024-C512-C256-C128
    d1 = create_decoder_block(b, e7, 512)
    d2 = create_decoder_block(d1, e6, 512)
    d3 = create_decoder_block(d2, e5, 512)
    d4 = create_decoder_block(d3, e4, 512, dropout=False)
    d5 = create_decoder_block(d4, e3, 256, dropout=False)
    d6 = create_decoder_block(d5, e2, 128, dropout=False)
    d7 = create_decoder_block(d6, e1, 64, dropout=False)
    # output
    g = Conv2DTranspose(
        3, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init
    )(d7)
    out_image = Activation("tanh")(g)
    # define model
    model = Model(in_image, out_image)
    return model
