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


def resize_image(
    test_image_tuple: TestImageTuple,
    resize: Tuple[int, int] = (286, 286),
) -> TestImageTuple:
    input_image = tf.image.resize(
        test_image_tuple.input_image,
        [resize[0], resize[1]],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
    )
    real_image = tf.image.resize(
        test_image_tuple.image,
        [resize[0], resize[1]],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
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


def normalize(test_image_tuple: TestImageTuple) -> TestImageTuple:
    input_image = (test_image_tuple.input_image / 127.5) - 1
    real_image = (test_image_tuple.image / 127.5) - 1
    return TestImageTuple(input_image=input_image, image=real_image)


@tf.function()
def random_jittering(
    test_image_tuple: TestImageTuple,
    resize: Tuple[int, int] = (286, 286),
    original_size: Tuple[int, int] = ORIGINAL_SIZE,
) -> TestImageTuple:
    logging.debug("Applying random jittering...")
    logging.debug("Resizing to %s...", str(resize))
    test_image_tuple = resize_image(test_image_tuple, resize)
    logging.debug("Randomly cropping back to original size %s...", str(original_size))
    test_image_tuple = random_crop(test_image_tuple, original_size)
    logging.debug("Randomly mirroring...")
    if tf.random.uniform(()) > 0.5:
        test_image_tuple = TestImageTuple(
            input_image=tf.image.flip_left_right(test_image_tuple.input_image),
            image=tf.image.flip_left_right(test_image_tuple.image),
        )
    return test_image_tuple


def load_image(image_file) -> TestImageTuple:
    logging.debug("Loading images...")
    image = tf.io.read_file(image_file)
    image = tf.image.decode_png(image)
    
    logging.debug("Splitting input and real images...")
    w = tf.shape(image)[1]
    w = w // 2
    input_image = image[:, :w, :]
    real_image = image[:, w:, :]
    
    logging.debug("Converting to float32...")
    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32) 
    
    return TestImageTuple(input_image=input_image, image=real_image)


def load_train_image(
    real_image_file: str,
    resize: Tuple[int, int] = (286, 286),
    original_size: Tuple[int, int] = ORIGINAL_SIZE,
) -> TestImageTuple:
    test_image_tuple = load_image(real_image_file)
    logging.debug("Treating train image...")
    logging.debug("Applying random jittering...")
    test_image_tuple = random_jittering(test_image_tuple, resize, original_size)
    logging.debug("Normalizing...")
    test_image_tuple = normalize(test_image_tuple)
    return test_image_tuple


def load_test_image(
    real_image_file: str,
    resize: Tuple[int, int] = (256, 256),
) -> TestImageTuple:
    test_image_tuple = load_image(real_image_file)
    test_image_tuple = resize_image(test_image_tuple, resize)
    test_image_tuple = normalize(test_image_tuple)
    return test_image_tuple


def get_train_dataset(input_data_dir: str, buffer_size: int = 400, batch_size: int = 1):
    logging.info("Getting train dataset...")
    train_dataset = tf.data.Dataset.list_files(input_data_dir + "/*.png")
    train_dataset = train_dataset.map(load_train_image, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.shuffle(buffer_size)
    train_dataset = train_dataset.batch(batch_size)
    return train_dataset


def get_test_dataset(input_data_dir: str, batch_size: int = 1):
    logging.info("Getting test dataset...")
    test_dataset = tf.data.Dataset.list_files(input_data_dir + "/*.png")
    test_dataset = test_dataset.map(load_test_image, num_parallel_calls=tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size)
    return test_dataset
