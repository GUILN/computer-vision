"""
1. Applies random jittering and mirroring to the training dataset.
2. Randomly flips the image horizontally.
3. Normalizes the images to [-1, 1].
https://www.tensorflow.org/tutorials/generative/pix2pix
"""
from typing import Tuple
import tensorflow as tf
import logging
from test_data_pipeline import TestImageTuple

ORIGINAL_SIZE = (256, 256)


def resize(
    test_image_tuple: TestImageTuple,
    resize: Tuple[int, int] = (286, 286),
) -> TestImageTuple:
    input_image = tf.image.resize(
        test_image_tuple.input_image,
        resize,
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
    )
    real_image = tf.image.resize(
        test_image_tuple.image, resize, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    return TestImageTuple(input_image=input_image, image=real_image)


def random_crop(
    test_image_tuple: TestImageTuple,
    size: Tuple[int, int] = ORIGINAL_SIZE,
) -> TestImageTuple:
    stacked_image = tf.stack(
        [test_image_tuple.input_image, test_image_tuple.image], axis=0
    )
    height, width = size
    cropped_image = tf.image.random_crop(stacked_image, size=[2, height, width, 3])
    return TestImageTuple(input_image=cropped_image[0], image=cropped_image[1])


@tf.function()
def random_jittering(
    test_image_tuple: TestImageTuple,
    resize: Tuple[int, int] = (286, 286),
    original_size: Tuple[int, int] = ORIGINAL_SIZE,
) -> TestImageTuple:
    logging.debug("Applying random jittering...")
    logging.debug("Resizing to %s...", str(resize))
    test_image_tuple = resize(test_image_tuple, resize)
    logging.debug("Randomly cropping back to original size %s...", str(original_size))
    test_image_tuple = random_crop(test_image_tuple, original_size)
    logging.debug("Randomly mirroring...")
    if tf.random.uniform(()) > 0.5:
        test_image_tuple = TestImageTuple(
            input_image=tf.image.flip_left_right(test_image_tuple.input_image),
            image=tf.image.flip_left_right(test_image_tuple.image),
        )
    return test_image_tuple
