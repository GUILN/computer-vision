from keras.optimizers import Adam
from keras.models import Model
from typing import Tuple
from keras.layers import BatchNormalization, Input


def create_gan(
    generator_model: "Model",
    discriminator_model: "Model",
    image_shape: Tuple[int, int, int] | Tuple[int, int] = (256, 256, 3),
) -> "Model":
    for layer in discriminator_model.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = False
    # define source image
    in_src = Input(shape=image_shape)
    # connect the source image to the generator input
    gen_out = generator_model(in_src)
    # connect the source input and generator output to the discriminator input
    dis_out = discriminator_model([in_src, gen_out])
    # src image as input, generated image and classification output
    model = Model(in_src, [dis_out, gen_out])
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(
        loss=["binary_crossentropy", "mae"],
        loss_weights=[1, 100],
        optimizer=opt,
    )
    return model
