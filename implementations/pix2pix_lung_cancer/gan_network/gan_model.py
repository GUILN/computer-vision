import os
from typing import Tuple
import tensorflow as tf

from gan_network.discriminator import Discriminator
from gan_network.generator import Generator


class GanModel:
    def __init__(
        self,
        checkpoint_dir: str,
        save_image_dir: str,
        log_dir: str = "logs/",
        input_size: Tuple[int, int] = (256, 256),
    ):
        self._generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self._discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self._generator = Generator()
        self._discriminator = Discriminator()
        self._log_dir = log_dir
        self._save_image_dir = save_image_dir
        self._checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        self._checkpoint = tf.train.Checkpoint(
            generator_optimizer=self._generator_optimizer,
            discriminator_optimizer=self._discriminator_optimizer,
        )

    def generate_images(
        self,
        test_input: tf.Tensor,
        target: tf.Tensor,
        image_name: str = "generated_image.png",
    ) -> None:
        """
        Generates images.
        :param test_input: the test input
        :param target: the target
        """
        prediction = self._generator(test_input, training=True)
        # save test_input, tar and prediction

        tf.keras.preprocessing.image.save_img(
            os.path.join(self._save_image_dir, "prediction_" + image_name),
            prediction[0].numpy(),
        )
        tf.keras.preprocessing.image.save_img(
            os.path.join(self._save_image_dir, "target_" + image_name),
            target[0].numpy(),
        )
        tf.keras.preprocessing.image.save_img(
            os.path.join(self._save_image_dir, "test_input_" + image_name),
            test_input[0].numpy(),
        )
