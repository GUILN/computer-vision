import tensorflow as tf

# The facade training set consist of 400 images
BUFFER_SIZE = 400
# The batch size of 1 produced better results for the U-Net in the
# original pix2pix experiment
BATCH_SIZE = 1
# Each image is 256x256 in size
IMG_WIDTH = 256
IMG_HEIGHT = 256


def load_image(image_file: str) -> str:
    """
    This function returns the path to the image.
    """
    # Read and decode an image file to a uint8 tensor
    image = tf.io.read_file(image_file)
    image = tf.io.decode_jpeg(image)

    # Split each image tensor into two tensors:
    # - one with a real building facade image
    # - one with an architecture label image
    w = tf.shape(image)[1]
    w = w // 2
    input_image = image[:, w:, :]
    real_image = image[:, :w, :]

    # Convert both images to float32 tensors
    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)

    return input_image, real_image


def resize(input_image: str, real_image: str, height: int, width: int) -> str:
    """
    This function resizes the images to the desired height and width.
    """
    input_image = tf.image.resize(
        input_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    real_image = tf.image.resize(
        real_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )

    return input_image, real_image


def random_crop(input_image: str, real_image: str) -> str:
    """
    This function crops the images to the desired height and width.
    """
    # Stack the images together
    stacked_image = tf.stack([input_image, real_image], axis=0)
    # Crop to the desired height and width
    cropped_image = tf.image.random_crop(
        stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3]
    )

    return cropped_image[0], cropped_image[1]


def normalize(input_image: str, real_image: str) -> str:
    """
    This function normalizes the images to the range [-1, 1].
    """
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1

    return input_image, real_image


@tf.function()
def random_jitter_preprocess(input_image: str, real_image: str) -> str:
    """
    This function applies random jitter to the images.
    """
    # Resize the image to a bigger height and width
    input_image, real_image = resize(input_image, real_image, 286, 286)
    # Randomly crop the image to the desired height and width
    input_image, real_image = random_crop(input_image, real_image)
    # Randomly flip the image horizontally
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)

    return input_image, real_image
