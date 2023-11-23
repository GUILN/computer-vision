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
)

from typing import Tuple


def create_discriminator(
    image_shape: Tuple[int, int, int] | Tuple[int, int]
) -> "Model":
    init = RandomNormal(stddev=0.02)
    in_src_image = Input(shape=image_shape)
    in_target_image = Input(shape=image_shape)
    # Concatenetate images channel-wise
    merged = Concatenate()([in_src_image, in_target_image])
    # C64
    d = Conv2D(64, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init)(
        merged
    )
    d = LeakyReLU(alpha=0.2)(d)
    # C128
    d = Conv2D(128, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C256
    d = Conv2D(256, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C512
    d = Conv2D(512, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # Second last output layer
    d = Conv2D(512, (4, 4), padding="same", kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # Patch output
    d = Conv2D(1, (4, 4), padding="same", kernel_initializer=init)(d)
    patch_out = Activation("sigmoid")(d)
    # Define model
    model = Model([in_src_image, in_target_image], patch_out)
    # Compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss="binary_crossentropy", optimizer=opt, loss_weights=[0.5])
    return model
